# pylint: disable=import-outside-toplevel,inconsistent-quotes,too-many-lines
"""
Cross-area quality summary report.

Parses solver logs for all areas and produces a summary showing:
  - Solve status and timing
  - Target accuracy (hit rate, mean/max error)
  - Weight distortion (RMSE, percentiles)
  - Aggregate multiplier distribution (old vs new weight comparison)
  - Violated targets by variable
  - Weight exhaustion and cross-area aggregation diagnostics
  - Bystander checks (untargeted variables + per-bin analysis)

Usage:
    python -m tmd.areas.quality_report
    python -m tmd.areas.quality_report --scope cds
    python -m tmd.areas.quality_report --scope cds --output
    python -m tmd.areas.quality_report --scope CA,WY -o report.txt
    python -m tmd.areas.quality_report --scope MN01,MN02
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    CD_TARGET_DIR,
    CD_WEIGHT_DIR,
    STATE_TARGET_DIR,
    STATE_WEIGHT_DIR,
)
from tmd.imputation_assumptions import TAXYEAR

_WT_COL = f"WT{TAXYEAR}"

# Decode raw constraint descriptions into human-readable labels
_CNT_LABELS = {0: "amt", 1: "returns", 2: "nz-count"}
_FS_LABELS = {0: "all", 1: "single", 2: "MFJ", 4: "HoH"}


def _humanize_desc(desc: str) -> str:
    """
    Turn 'c00100/cnt=1/scope=1/agi=[500000.0,1000000.0)/fs=4'
    into 'c00100 returns HoH $500K-$1M'.
    """
    parts = desc.split("/")
    varname = parts[0]
    attrs = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            attrs[k] = v

    cnt = int(attrs.get("cnt", -1))
    fs = int(attrs.get("fs", 0))
    agi_raw = attrs.get("agi", "")

    cnt_label = _CNT_LABELS.get(cnt, f"cnt{cnt}")
    fs_label = _FS_LABELS.get(fs, f"fs{fs}")

    # Parse AGI range like [500000.0,1000000.0)
    agi_label = ""
    m = re.match(r"\[([^,]+),([^)]+)\)", agi_raw)
    if m:
        lo_s, hi_s = m.group(1), m.group(2)
        lo = float(lo_s)
        hi = float(hi_s)
        if lo < -1e10:
            agi_label = f"<${hi / 1000:.0f}K"
        elif hi > 1e10:
            agi_label = f"${lo / 1000:.0f}K+"
        else:
            agi_label = f"${lo / 1000:.0f}K-${hi / 1000:.0f}K"

    pieces = [varname, cnt_label]
    if fs != 0:
        pieces.append(fs_label)
    if agi_label:
        pieces.append(agi_label)
    return " ".join(pieces)


def _fmt_agi_bin(lo, hi):
    """Format an AGI bin range for display."""
    if lo < -1e10:
        return "<$0"
    if hi > 1e10:
        return f"${lo / 1000:.0f}K+"
    return f"${lo / 1000:.0f}K-${hi / 1000:.0f}K"


def parse_log(logpath: Path) -> dict:
    """Parse a single area solver log file into a summary dict."""
    if not logpath.exists():
        return {"status": "NO LOG"}
    log = logpath.read_text()

    result = {"status": "UNKNOWN"}

    # Solve status
    m = re.search(r"Solver status: (\S+)", log)
    if m:
        result["status"] = m.group(1)
    if "PrimalInfeasible" in log or "FAILED" in log:
        result["status"] = "FAILED"

    # Population share
    m = re.search(r"pop_share = ([\d.]+) / ([\d.]+) = ([\d.]+)", log)
    if m:
        result["pop_share"] = float(m.group(3))

    # Solve time
    m = re.search(r"Solve time: ([\d.]+)s", log)
    if m:
        result["solve_time"] = float(m.group(1))

    # Target accuracy
    m = re.search(r"mean \|relative error\|: ([\d.]+)", log)
    if m:
        result["mean_err"] = float(m.group(1))
    m = re.search(r"max  \|relative error\|: ([\d.]+)", log)
    if m:
        result["max_err"] = float(m.group(1))
    m = re.search(r"targets hit: (\d+)/(\d+)", log)
    if m:
        result["targets_hit"] = int(m.group(1))
        result["targets_total"] = int(m.group(2))
    m_viol = re.search(r"VIOLATED: (\d+) targets", log)
    result["n_violated"] = int(m_viol.group(1)) if m_viol else 0

    # Weight distortion
    m = re.search(
        r"min=([\d.]+), p5=([\d.]+), median=([\d.]+), "
        r"p95=([\d.]+), max=([\d.]+)",
        log,
    )
    if m:
        result["w_min"] = float(m.group(1))
        result["w_p5"] = float(m.group(2))
        result["w_median"] = float(m.group(3))
        result["w_p95"] = float(m.group(4))
        result["w_max"] = float(m.group(5))
    m = re.search(r"RMSE from 1.0: ([\d.]+)", log)
    if m:
        result["w_rmse"] = float(m.group(1))

    # Weight distribution histogram
    dist_bins = {}
    for line in log.splitlines():
        m_bin = re.match(
            r"\s+\[\s*([\d.]+),\s*([\d.]+)\):\s+(\d+)\s+\(\s*([\d.]+)%\)",
            line,
        )
        if m_bin:
            lo, hi = float(m_bin.group(1)), float(m_bin.group(2))
            cnt, pct = int(m_bin.group(3)), float(m_bin.group(4))
            dist_bins[(lo, hi)] = {"count": cnt, "pct": pct}
    result["dist_bins"] = dist_bins

    m_n = re.search(r"distribution \(n=(\d+)\)", log)
    if m_n:
        result["n_records"] = int(m_n.group(1))

    # Violated target details
    violated = []
    in_violated = False
    for line in log.splitlines():
        if "VIOLATED:" in line and "targets" in line:
            in_violated = True
            continue
        if in_violated:
            m_det = re.match(
                r"\s+([\d.]+)%\s*\|\s*target=\s*([\d.]+)\s*\|"
                r"\s*achieved=\s*([\d.]+)\s*\|"
                r"\s*(\S+/cnt=\d+/scope=\d+/agi=.*?/fs=\d+)",
                line,
            )
            if m_det:
                violated.append(
                    {
                        "pct_err": float(m_det.group(1)),
                        "target": float(m_det.group(2)),
                        "achieved": float(m_det.group(3)),
                        "desc": m_det.group(4),
                    }
                )
            else:
                in_violated = False
    result["violated_details"] = violated

    return result


def _list_areas_from_logs(weight_dir):
    """Return sorted list of area codes from log files."""
    logs = sorted(weight_dir.glob("*.log"))
    return [lp.stem for lp in logs]


def _infer_target_dir(weight_dir):
    """Infer target directory from weight directory."""
    # weight_dir is like .../weights/states or .../weights/cds
    return weight_dir.parent.parent / "targets" / weight_dir.name


def _aggregate_multiplier_histogram(solved_df):
    """
    Aggregate per-area multiplier distributions into a combined view.

    The multiplier x = area_weight / (pop_share * national_weight).
    x = 1.0 means the record keeps its population-proportional share.
    This is the side-by-side comparison of initial (x=1) vs optimized
    weights.
    """
    lines = []

    # Canonical bin order matching the solver's histogram
    # Note: exact zeros are logged as [0.0000, 0.0000) → key (0.0, 0.0)
    canonical_bins = [
        (0.0, 0.0),
        (0.0, 0.1),
        (0.1, 0.5),
        (0.5, 0.8),
        (0.8, 0.9),
        (0.9, 0.95),
        (0.95, 1.0),
        (1.0, 1.05),
        (1.05, 1.1),
        (1.1, 1.2),
        (1.2, 1.5),
        (1.5, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
        (10.0, 100.0),
        (100.0, np.inf),
    ]
    # Aggregate counts from all areas
    agg = {}
    total_records = 0
    n_areas = 0
    for _, row in solved_df.iterrows():
        dist = row.get("dist_bins", {})
        n_rec = row.get("n_records", 0)
        if not dist or n_rec == 0:
            continue
        n_areas += 1
        total_records += n_rec
        for key, val in dist.items():
            agg[key] = agg.get(key, 0) + val["count"]

    if total_records == 0:
        return lines

    n_per_area = total_records // n_areas if n_areas > 0 else 0
    lines.append(
        "AGGREGATE MULTIPLIER DISTRIBUTION"
        f" ({n_areas} areas x {n_per_area:,}"
        f" records = {total_records:,} area-record pairs):"
    )
    lines.append(
        "  The multiplier x = area_weight / (pop_share * national_weight)."
    )
    lines.append(
        "  x = 1.0 means the record keeps its population-proportional"
        " share."
    )
    lines.append(
        "  Initial (pre-optimization): all x = 1.0."
        " This histogram shows the optimized distribution."
    )

    bin_labels = {
        (0.0, 0.0): "x = 0 (excluded)",
        (0.0, 0.1): "(0, 0.1)",
        (0.1, 0.5): "[0.1, 0.5)",
        (0.5, 0.8): "[0.5, 0.8)",
        (0.8, 0.9): "[0.8, 0.9)",
        (0.9, 0.95): "[0.9, 0.95)",
        (0.95, 1.0): "[0.95, 1.0)",
        (1.0, 1.05): "[1.0, 1.05)",
        (1.05, 1.1): "[1.05, 1.1)",
        (1.1, 1.2): "[1.1, 1.2)",
        (1.2, 1.5): "[1.2, 1.5)",
        (1.5, 2.0): "[1.5, 2.0)",
        (2.0, 5.0): "[2.0, 5.0)",
        (5.0, 10.0): "[5.0, 10.0)",
        (10.0, 100.0): "[10.0, 100.0)",
        (100.0, np.inf): "[100.0, inf)",
    }

    lines.append(f"  {'Bin':<20} {'Count':>12} {'Pct':>8} {'CumPct':>8}")
    lines.append("  " + "-" * 50)
    cum_pct = 0.0
    for bkey in canonical_bins:
        cnt = agg.get(bkey, 0)
        if cnt == 0:
            continue
        pct = 100.0 * cnt / total_records
        cum_pct += pct
        label = bin_labels.get(bkey, str(bkey))
        lines.append(f"  {label:<20} {cnt:>12,} {pct:>7.1f}% {cum_pct:>7.1f}%")

    # Summary stats
    n_near_one = sum(agg.get(b, 0) for b in [(0.95, 1.0), (1.0, 1.05)])
    n_within_20 = sum(
        agg.get(b, 0)
        for b in [
            (0.8, 0.9),
            (0.9, 0.95),
            (0.95, 1.0),
            (1.0, 1.05),
            (1.05, 1.1),
            (1.1, 1.2),
        ]
    )
    n_excluded = agg.get((0.0, 0.0), 0)
    n_extreme = sum(
        agg.get(b, 0) for b in [(5.0, 10.0), (10.0, 100.0), (100.0, np.inf)]
    )
    lines.append("")
    lines.append(
        f"  Within +/-5% of 1.0:"
        f" {n_near_one:,} ({100 * n_near_one / total_records:.1f}%)"
    )
    lines.append(
        f"  Within +/-20% of 1.0:"
        f" {n_within_20:,} ({100 * n_within_20 / total_records:.1f}%)"
    )
    lines.append(
        f"  Excluded (x=0):"
        f" {n_excluded:,} ({100 * n_excluded / total_records:.1f}%)"
    )
    if n_extreme > 0:
        lines.append(
            f"  Extreme (x>=5.0):"
            f" {n_extreme:,} ({100 * n_extreme / total_records:.1f}%)"
        )
    lines.append("")

    return lines


def generate_report(
    areas=None,
    weight_dir=None,
    target_dir=None,
    scope_label=None,
):
    """Generate cross-area quality summary report."""
    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR
    if target_dir is None:
        target_dir = _infer_target_dir(weight_dir)
    if areas is None:
        areas = _list_areas_from_logs(weight_dir)

    rows = []
    for st in areas:
        logpath = weight_dir / f"{st.lower()}.log"
        info = parse_log(logpath)
        info["area"] = st
        rows.append(info)

    df = pd.DataFrame(rows)

    # Summary statistics
    solved = df[df["status"].isin(["Solved", "AlmostSolved"])]
    failed = df[df["status"] == "FAILED"]
    n_solved = len(solved)
    n_failed = len(failed)
    n_violated_states = (solved["n_violated"] > 0).sum()
    total_violated = solved["n_violated"].sum()

    tol_pct = AREA_CONSTRAINT_TOL * 100

    lines = []
    lines.append("=" * 80)
    header = "CROSS-AREA QUALITY SUMMARY REPORT"
    if scope_label:
        header += f"  [{scope_label}]"
    lines.append(header)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Overall
    n_areas = len(df)
    lines.append(f"Areas: {n_areas}")
    lines.append(f"Solved: {n_solved}")
    lines.append(f"Failed: {n_failed}")
    if n_failed > 0:
        lines.append(f"  Failed: {', '.join(failed['area'].tolist())}")
    lines.append(
        f"Areas with violated targets: {n_violated_states}/{n_solved}"
    )
    if not solved.empty and "targets_total" in solved.columns:
        tpt = int(solved["targets_total"].iloc[0])
        tpt_sum = int(solved["targets_total"].sum())
    else:
        tpt, tpt_sum = "?", "?"
    lines.append(f"Total targets: {n_solved} areas \u00d7 {tpt} = {tpt_sum}")
    lines.append(f"Total violated targets: {int(total_violated)}")
    if not solved.empty and "solve_time" in solved.columns:
        cum_time = solved["solve_time"].sum()
        avg_time = solved["solve_time"].mean()
        lines.append(
            f"Cumulative solve time: {cum_time:.0f}s"
            f" (avg {avg_time:.1f}s per area;"
            f" ~{cum_time / 16:.0f}s wall @ 16 workers)"
        )
    lines.append("")

    # Target accuracy
    if not solved.empty and "mean_err" in solved.columns:
        lines.append("TARGET ACCURACY:")
        lines.append(
            "  Error = |achieved - target| / |target|."
            "  A target is 'hit' if error < tolerance."
        )
        lines.append(
            f"  Per-area mean error:"
            f"avg={solved['mean_err'].mean():.4f}, "
            f"worst={solved['mean_err'].max():.4f}"
        )
        lines.append(
            f"  Per-area max error: "
            f"avg={solved['max_err'].mean():.4f}, "
            f"worst={solved['max_err'].max():.4f}"
        )
        if "targets_hit" in solved.columns:
            total_t = solved["targets_total"].iloc[0]
            hit_pcts = solved["targets_hit"] / solved["targets_total"] * 100
            lines.append(
                f"  Hit rate:  "
                f"avg={hit_pcts.mean():.1f}%, "
                f"min={hit_pcts.min():.1f}% "
                f"(out of {total_t} targets, "
                f"tolerance: +/-{tol_pct:.1f}% + eps)"
            )
        lines.append("")

    # Weight distortion
    if not solved.empty and "w_rmse" in solved.columns:
        lines.append("WEIGHT DISTORTION (multiplier from 1.0):")
        lines.append(
            "  Each record's multiplier x = area_weight"
            " / (pop_share * national_weight)."
        )
        lines.append(
            "  x=1.0 means population-proportional."
            "  RMSE measures overall departure from x=1."
        )
        lines.append(
            f"  RMSE:   avg={solved['w_rmse'].mean():.3f}, "
            f"max={solved['w_rmse'].max():.3f}"
        )
        lines.append(
            f"  Min:    avg={solved['w_min'].mean():.3f}, "
            f"min={solved['w_min'].min():.3f}"
        )
        lines.append(
            f"  P05:    avg={solved['w_p5'].mean():.3f}, "
            f"min={solved['w_p5'].min():.3f}"
        )
        lines.append(
            f"  Median: avg={solved['w_median'].mean():.3f}, "
            f"range=[{solved['w_median'].min():.3f}, "
            f"{solved['w_median'].max():.3f}]"
        )
        lines.append(
            f"  P95:    avg={solved['w_p95'].mean():.3f}, "
            f"max={solved['w_p95'].max():.3f}"
        )
        lines.append(
            f"  Max:    avg={solved['w_max'].mean():.1f}, "
            f"max={solved['w_max'].max():.1f}"
        )
        lines.append("")

    # Aggregate multiplier distribution (old vs new weight comparison)
    if not solved.empty:
        lines.extend(_aggregate_multiplier_histogram(solved))

    # Near-zero weight summary
    if not solved.empty:
        zero_pcts = []
        lt01_pcts = []
        for _, row in solved.iterrows():
            dist = row.get("dist_bins", {})
            n_rec = row.get("n_records", 0)
            if not dist or n_rec == 0:
                continue
            n_zero = dist.get((0.0, 0.0), {}).get("count", 0)
            n_lt01 = n_zero + dist.get((0.0, 0.1), {}).get("count", 0)
            zero_pcts.append(100 * n_zero / n_rec)
            lt01_pcts.append(100 * n_lt01 / n_rec)
        if zero_pcts:
            lines.append("NEAR-ZERO WEIGHT MULTIPLIERS (% of records):")
            lines.append(
                "  Records with x near 0 are effectively"
                " excluded from the area."
            )
            lines.append(
                f"  Exact zero (x=0):  "
                f"avg={np.mean(zero_pcts):.1f}%, "
                f"max={np.max(zero_pcts):.1f}%"
            )
            lines.append(
                f"  Below 0.1 (x<0.1): "
                f"avg={np.mean(lt01_pcts):.1f}%, "
                f"max={np.max(lt01_pcts):.1f}%"
            )
            lines.append("")

    # Load TMD data and weight files (once, shared by all diagnostics)
    report_data = _load_report_data(areas, weight_dir)
    if report_data is not None:
        tmd, s006, state_weights, n_loaded = report_data
        # Weight distribution by AGI stub (old vs new)
        lines.extend(
            _weight_distribution_by_stub(
                tmd, s006, state_weights, n_loaded, target_dir
            )
        )

    # Per-area table
    # For small area counts (states), show all areas.
    # For large area counts (CDs, counties), show top 20 by max error.
    _DETAIL_CUTOFF = 60
    show_all = n_areas <= _DETAIL_CUTOFF

    if show_all:
        lines.append("PER-AREA DETAIL:")
        display_df = df
    else:
        lines.append(
            "PER-AREA DETAIL (top 20 by violations / weight distortion):"
        )
        # Always include failed areas, then sort solved by
        # violations desc, then weight RMSE desc
        failed_rows = df[df["status"] == "FAILED"]
        solved_rows = df[df["status"].isin(["Solved", "AlmostSolved"])].copy()
        solved_rows["_viol"] = solved_rows["n_violated"].fillna(0)
        solved_rows["_rmse"] = solved_rows["w_rmse"].fillna(0)
        top_solved = solved_rows.sort_values(
            ["_viol", "_rmse"], ascending=[False, False]
        ).head(20)
        top_solved = top_solved.drop(columns=["_viol", "_rmse"])
        display_df = pd.concat([failed_rows, top_solved]).drop_duplicates(
            subset=["area"]
        )

    lines.append(
        "  MeanErr/MaxErr = |relative error| (fraction)."
        "  wRMSE = root-mean-square deviation of"
        " multipliers from 1.0"
    )
    lines.append(
        "  (higher = more distortion)."
        "  wP05/wMed/wP95/wMax = multiplier percentiles."
        "  %zero = records excluded."
    )
    max_id = max(
        (len(str(row["area"])) for _, row in display_df.iterrows()),
        default=4,
    )
    id_w = max(max_id + 1, 5)
    header = (
        f"{'Area':<{id_w}} {'Status':<14} {'Hit':>5} {'Tot':>5} "
        f"{'Viol':>5} {'MeanErr':>8} {'MaxErr':>8} "
        f"{'wRMSE':>7} {'wP05':>7} {'wMed':>7} "
        f"{'wP95':>7} {'wMax':>8} {'%zero':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in display_df.iterrows():
        hit = int(row.get("targets_hit", 0))
        tot = int(row.get("targets_total", 0))
        viol = int(row.get("n_violated", 0))
        me = row.get("mean_err", 0)
        mx = row.get("max_err", 0)
        rmse = row.get("w_rmse", 0)
        p5 = row.get("w_p5", 0)
        med = row.get("w_median", 0)
        p95 = row.get("w_p95", 0)
        wmax = row.get("w_max", 0)
        dist = row.get("dist_bins", {})
        n_rec = row.get("n_records", 0)
        n_zero = 0
        if isinstance(dist, dict):
            n_zero = dist.get((0.0, 0.0), {}).get("count", 0)
        pct_zero = 100 * n_zero / n_rec if n_rec > 0 else 0
        area_str = str(row["area"])
        lines.append(
            f"{area_str:<{id_w}} {row['status']:<14} {hit:>5} {tot:>5} "
            f"{viol:>5} {me:>8.4f} {mx:>8.4f} "
            f"{rmse:>7.3f} {p5:>7.3f} {med:>7.3f} "
            f"{p95:>7.3f} {wmax:>8.1f} {pct_zero:>5.1f}%"
        )
    if not show_all:
        n_omitted = n_areas - len(display_df)
        lines.append(f"  ... {n_omitted} areas omitted (all within tolerance)")
    lines.append("")

    # Violated targets by variable
    all_violated = []
    for _, row in df.iterrows():
        for v in row.get("violated_details", []):
            desc = v["desc"]
            varname = desc.split("/")[0]
            cnt_m = re.search(r"cnt=(\d+)", desc)
            cnt_type = int(cnt_m.group(1)) if cnt_m else -1
            abs_miss = abs(v["achieved"] - v["target"])
            all_violated.append(
                {
                    "area": row["area"],
                    "varname": varname,
                    "cnt_type": cnt_type,
                    "pct_err": v["pct_err"],
                    "target": v["target"],
                    "achieved": v["achieved"],
                    "abs_miss": abs_miss,
                    "desc": desc,
                }
            )
    if all_violated:
        vdf = pd.DataFrame(all_violated)
        var_counts = vdf["varname"].value_counts()
        lines.append("VIOLATIONS BY VARIABLE:")
        for var, cnt in var_counts.items():
            areas_with = sorted(vdf[vdf["varname"] == var]["area"].unique())
            lines.append(
                f"  {var}: {cnt} violations across " f"{len(areas_with)} areas"
            )
        lines.append("")

        area_counts = vdf["area"].value_counts().head(10)
        lines.append("AREAS WITH MOST VIOLATIONS:")
        for st, cnt in area_counts.items():
            lines.append(f"  {st}: {cnt} violated")
        lines.append("")

        amt_viol = vdf[vdf["cnt_type"] == 0].sort_values(
            ["pct_err", "abs_miss"], ascending=[False, False]
        )
        lines.append("WORST 5 AMOUNT VIOLATIONS (by % error):")
        if amt_viol.empty:
            lines.append("  (none — all amount targets met)")
        else:
            for _, r in amt_viol.head(5).iterrows():
                lines.append(
                    f"  {r['area']:<4} {r['pct_err']:.3f}% "
                    f"target=${r['target']:>15,.0f}  "
                    f"achieved=${r['achieved']:>15,.0f}  "
                    f"miss=${r['abs_miss']:>12,.0f}  "
                    f"{_humanize_desc(r['desc'])}"
                )
        lines.append("")

        cnt_viol = vdf[vdf["cnt_type"].isin([1, 2])].sort_values(
            ["pct_err", "abs_miss"], ascending=[False, False]
        )
        lines.append("WORST 5 COUNT VIOLATIONS (by % error):")
        if cnt_viol.empty:
            lines.append("  (none — all count targets met)")
        else:
            for _, r in cnt_viol.head(5).iterrows():
                lines.append(
                    f"  {r['area']:<4} {r['pct_err']:.3f}% "
                    f"target={r['target']:>12,.0f}  "
                    f"achieved={r['achieved']:>12,.0f}  "
                    f"miss={r['abs_miss']:>8,.0f}  "
                    f"{_humanize_desc(r['desc'])}"
                )
        lines.append("")

    # Weight diagnostics (uses pre-loaded data if available)
    if report_data is not None:
        tmd, s006, state_weights, n_loaded = report_data
        lines.extend(
            _weight_diagnostics(
                areas,
                weight_dir,
                target_dir,
                tmd,
                s006,
                state_weights,
                n_loaded,
            )
        )

    report = "\n".join(lines)
    return report


def _load_report_data(areas, weight_dir):
    """
    Load TMD data and area weight files for diagnostics.

    Called once from generate_report(); results passed to all
    diagnostic functions to avoid reloading.

    Returns (tmd, s006, state_weights, n_loaded) or None if
    no weight files found.
    """
    from tmd.storage import STORAGE_FOLDER

    tmd_path = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    tmd_cols = [
        "RECID",
        "s006",
        "MARS",
        "XTOT",
        "data_source",
        "e00200",
        "e00300",
        "e00400",
        "e00600",
        "e00650",
        "e00900",
        "e01400",
        "e01500",
        "e01700",
        "e02000",
        "e02300",
        "e02400",
        "e17500",
        "e19200",
        "e26270",
        "n24",
        "p22250",
        "p23250",
    ]
    tmd = pd.read_csv(tmd_path, usecols=tmd_cols)
    tmd["capgains_net"] = tmd["p22250"] + tmd["p23250"]
    s006 = tmd["s006"].values

    agi_path = STORAGE_FOLDER / "output" / "cached_c00100.npy"
    if agi_path.exists():
        tmd["c00100"] = np.load(agi_path)

    allvars_path = STORAGE_FOLDER / "output" / "cached_allvars.csv"
    if allvars_path.exists():
        needed_tc = [
            "c04470",
            "c07100",
            "c09600",
            "c18300",
            "c19200",
            "c19700",
            "iitax",
            "payrolltax",
            "standard",
        ]
        avail = pd.read_csv(allvars_path, nrows=0).columns
        load_tc = [c for c in needed_tc if c in avail]
        if load_tc:
            allvars = pd.read_csv(allvars_path, usecols=load_tc)
            for col in load_tc:
                tmd[col] = allvars[col].values

    n_records = len(tmd)
    weight_sum = np.zeros(n_records)
    state_weights = {}
    for st in areas:
        wpath = weight_dir / f"{st.lower()}_tmd_weights.csv.gz"
        if not wpath.exists():
            continue
        w = pd.read_csv(wpath, usecols=[_WT_COL])[_WT_COL].values
        weight_sum += w
        state_weights[st] = w

    n_loaded = len(state_weights)
    if n_loaded == 0:
        return None
    return tmd, s006, state_weights, n_loaded


def _weight_distribution_by_stub(
    tmd, s006, state_weights, n_loaded, target_dir
):
    """
    Show weighted return counts and AGI by AGI stub:
    national (old) vs sum-of-areas (new) and change.
    """
    lines = []

    if "c00100" not in tmd.columns:
        return lines

    # Get AGI bins from target file
    first_area = next(iter(state_weights.keys()), None)
    if first_area is None:
        return lines
    tgt_path = target_dir / f"{first_area}_targets.csv"
    if not tgt_path.exists():
        return lines
    tgt_df = pd.read_csv(tgt_path, comment="#")

    agi_pairs = set()
    for _, row in tgt_df.iterrows():
        lo, hi = float(row.agilo), float(row.agihi)
        if lo < -1e10 and hi > 1e10:
            continue
        agi_pairs.add((lo, hi))
    agi_bins = sorted(agi_pairs)
    if not agi_bins:
        return lines

    c00100 = tmd["c00100"].values
    lines.append(f"WEIGHT DISTRIBUTION BY AGI STUB ({n_loaded} areas):")
    lines.append(
        "  National = s006-weighted totals."
        " Sum-of-Areas = area-weight-weighted totals."
    )

    # Header
    lines.append(
        f"  {'AGI Stub':<18}"
        f" {'Natl Returns':>14} {'Area Returns':>14} {'Chg%':>7}"
        f" {'Natl AGI ($B)':>14} {'Area AGI ($B)':>14} {'Chg%':>7}"
    )
    lines.append("  " + "-" * 92)

    # Running totals
    tot_nat_ret = 0.0
    tot_area_ret = 0.0
    tot_nat_agi = 0.0
    tot_area_agi = 0.0

    for lo, hi in agi_bins:
        in_bin = (c00100 >= lo) & (c00100 < hi)
        mask = in_bin.astype(float)

        # Returns
        nat_ret = float((s006 * mask).sum())
        area_ret = sum(float((w * mask).sum()) for w in state_weights.values())
        ret_chg = (area_ret / nat_ret - 1) * 100 if nat_ret else 0

        # AGI
        agi_vals = c00100 * mask
        nat_agi = float((s006 * agi_vals).sum())
        area_agi = sum(
            float((w * agi_vals).sum()) for w in state_weights.values()
        )
        agi_chg = (area_agi / nat_agi - 1) * 100 if nat_agi else 0

        tot_nat_ret += nat_ret
        tot_area_ret += area_ret
        tot_nat_agi += nat_agi
        tot_area_agi += area_agi

        bin_label = _fmt_agi_bin(lo, hi)
        flag = " *" if abs(ret_chg) > 2 or abs(agi_chg) > 2 else ""
        lines.append(
            f"  {bin_label:<18}"
            f" {nat_ret:>14,.0f} {area_ret:>14,.0f}"
            f" {ret_chg:>+6.2f}%"
            f" {nat_agi / 1e9:>14.1f} {area_agi / 1e9:>14.1f}"
            f" {agi_chg:>+6.2f}%{flag}"
        )

    # Total row
    tot_ret_chg = (tot_area_ret / tot_nat_ret - 1) * 100 if tot_nat_ret else 0
    tot_agi_chg = (tot_area_agi / tot_nat_agi - 1) * 100 if tot_nat_agi else 0
    lines.append("  " + "-" * 92)
    lines.append(
        f"  {'TOTAL':<18}"
        f" {tot_nat_ret:>14,.0f} {tot_area_ret:>14,.0f}"
        f" {tot_ret_chg:>+6.2f}%"
        f" {tot_nat_agi / 1e9:>14.1f} {tot_area_agi / 1e9:>14.1f}"
        f" {tot_agi_chg:>+6.2f}%"
    )

    n_flagged = 0
    for lo, hi in agi_bins:
        in_bin = (c00100 >= lo) & (c00100 < hi)
        mask = in_bin.astype(float)
        nat_ret = float((s006 * mask).sum())
        area_ret = sum(float((w * mask).sum()) for w in state_weights.values())
        ret_chg = (area_ret / nat_ret - 1) * 100 if nat_ret else 0
        agi_vals = c00100 * mask
        nat_agi = float((s006 * agi_vals).sum())
        area_agi = sum(
            float((w * agi_vals).sum()) for w in state_weights.values()
        )
        agi_chg = (area_agi / nat_agi - 1) * 100 if nat_agi else 0
        if abs(ret_chg) > 2 or abs(agi_chg) > 2:
            n_flagged += 1
    if n_flagged:
        lines.append(f"  * = {n_flagged} stubs with >2% change")
    lines.append("")

    return lines


def _weight_diagnostics(
    _areas, _weight_dir, target_dir, tmd, s006, state_weights, n_loaded
):
    """
    Combined weight diagnostics: exhaustion + national aggregation.

    Uses pre-loaded TMD data and weight files (from _load_report_data).
    """
    lines = []
    n_records = len(tmd)

    # Weight exhaustion
    weight_sum = np.zeros(n_records)
    for w in state_weights.values():
        weight_sum += w
    usage = weight_sum / s006

    _sub = "sum of area weights / national weight"
    lines.append(f"WEIGHT EXHAUSTION ({_sub}):")
    lines.append(
        f"  A ratio of 1.0 means the record's national"
        f" weight is fully allocated across"
        f" {n_loaded} areas."
    )
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    quantiles = np.percentile(usage, pcts)
    parts = []
    for p, q in zip(pcts, quantiles):
        label = {0: "min", 50: "median", 100: "max"}.get(p, f"p{p}")
        parts.append(f"{label}={q:.4f}")
    lines.append("  " + ", ".join(parts))
    lines.append(f"  Mean: {usage.mean():.4f}, Std: {usage.std():.4f}")

    n_over = int((usage > 1.10).sum())
    n_under = int((usage < 0.90).sum())
    lines.append(
        f"  Over-used (>1.10):"
        f" {n_over} ({100 * n_over / n_records:.1f}%)  "
        f"Under-used (<0.90):"
        f" {n_under} ({100 * n_under / n_records:.1f}%)"
    )
    for thresh in [2, 5, 10]:
        ct = int((usage > thresh).sum())
        if ct > 0:
            lines.append(f"  Exhaustion > {thresh}x: {ct} records")
    lines.append("")

    # Most exhausted records — profile and top states
    _mars = {1: "Single", 2: "MFJ", 3: "MFS", 4: "HoH", 5: "Wid"}
    _ds = {0: "CPS", 1: "PUF"}
    top_idx = np.argsort(usage)[::-1][:5]
    lines.append("MOST EXHAUSTED RECORDS (top 5):")
    for rank, idx in enumerate(top_idx, 1):
        r = tmd.iloc[idx]
        exh = usage[idx]
        mars = _mars.get(int(r.MARS), f"fs{int(r.MARS)}")
        ds = _ds.get(int(r.data_source), "?")
        agi = tmd["c00100"].iloc[idx] if "c00100" in tmd else 0
        # Top 3 states by weight for this record
        st_wts = []
        for st, w in state_weights.items():
            if w[idx] > 0:
                st_wts.append((st, w[idx]))
        st_wts.sort(key=lambda x: -x[1])
        top3 = ", ".join(f"{st}={wt:.1f}" for st, wt in st_wts[:3])
        recid = int(r.RECID)
        lines.append(
            f"  {rank}. RECID {recid}: exh={exh:.1f}x,"
            f" s006={r.s006:.1f},"
            f" {ds} {mars},"
            f" AGI=${agi:,.0f}"
        )
        lines.append(
            f"     wages=${r.e00200:,.0f},"
            f" int=${r.e00300:,.0f},"
            f" div=${r.e00600:,.0f},"
            f" ptshp=${r.e26270:,.0f}"
        )
        lines.append(
            f"     top areas: {top3}" f" ({len(st_wts)} nonzero of {n_loaded})"
        )
    lines.append("")

    # Cross-state aggregation vs national totals
    check_vars = [
        ("Returns (s006)", "s006", True),
        ("AGI (c00100)", "c00100", False),
        ("Wages (e00200)", "e00200", False),
        ("Capital gains (capgains_net)", "capgains_net", False),
        ("SALT ded (c18300)", "c18300", False),
        ("Income tax (iitax)", "iitax", False),
    ]

    national = {}
    for label, var, is_count in check_vars:
        if var not in tmd.columns:
            national[var] = None
            continue
        if var == "s006":
            national[var] = float(s006.sum())
        else:
            national[var] = float((s006 * tmd[var].values).sum())

    state_sums = {var: 0.0 for _, var, _ in check_vars}
    for _st, w in state_weights.items():
        for _label, var, _is_count in check_vars:
            if var not in tmd.columns:
                continue
            if var == "s006":
                state_sums[var] += float(w.sum())
            else:
                state_sums[var] += float((w * tmd[var].values).sum())

    lines.append(
        f"CROSS-AREA AGGREGATION vs NATIONAL TOTALS"
        f" for SELECTED VARIABLES ({n_loaded} areas):"
    )
    lines.append(
        "  Do area weights preserve national totals?  Diff% near 0 = good."
    )
    lines.append(
        f"  {'Variable':<30} {'National':>16}"
        f" {'Sum-of-Areas':>16} {'Diff':>14} {'Diff%':>8}"
    )
    lines.append("  " + "-" * 86)
    for label, var, is_count in check_vars:
        nat = national[var]
        sos = state_sums[var]
        if nat is None or nat == 0:
            continue
        diff = sos - nat
        diff_pct = (sos / nat - 1) * 100
        if is_count:
            lines.append(
                f"  {label:<30} {nat:>16,.0f}"
                f" {sos:>16,.0f}"
                f" {diff:>+14,.0f}"
                f" {diff_pct:>+7.2f}%"
            )
        else:
            lines.append(
                f"  {label:<30}"
                f" ${nat / 1e9:>14.1f}B"
                f" ${sos / 1e9:>14.1f}B"
                f" ${diff / 1e9:>+12.2f}B"
                f" {diff_pct:>+7.2f}%"
            )
    lines.append("")

    # Bystander check: untargeted variables (aggregate)
    lines.extend(_bystander_check(tmd, s006, state_weights, n_loaded))

    # Bystander check: per-bin analysis for targeted + key untargeted vars
    lines.extend(
        _bystander_by_bin(tmd, s006, state_weights, n_loaded, target_dir)
    )

    return lines


def _bystander_check(tmd, s006, state_weights, n_loaded):
    """
    Check untargeted variables for cross-state aggregation
    distortion. These are 'innocent bystanders' that may be
    jerked around by weight adjustments aimed at targeted
    variables.
    """
    lines = []

    # Untargeted variables to check, grouped by category
    # Format: (label, varname, is_count)
    bystander_vars = [
        # Tax liability / credits
        ("Income tax (iitax)", "iitax", False),
        ("Payroll tax", "payrolltax", False),
        ("AMT (c09600)", "c09600", False),
        ("Total credits (c07100)", "c07100", False),
        # Deductions
        ("Medical expenses (e17500)", "e17500", False),
        ("Student loan int (e19200)", "e19200", False),
        ("Itemized ded (c04470)", "c04470", False),
        ("Standard deduction", "standard", False),
        ("Mortgage int (c19200)", "c19200", False),
        ("Charitable (c19700)", "c19700", False),
        # Income not directly targeted
        ("Tax-exempt int (e00400)", "e00400", False),
        ("Qual dividends (e00650)", "e00650", False),
        ("Sch C income (e00900)", "e00900", False),
        ("IRA distrib (e01400)", "e01400", False),
        ("Taxable pensions (e01700)", "e01700", False),
        ("Sch E net (e02000)", "e02000", False),
        ("Unemployment (e02300)", "e02300", False),
        # Demographics
        ("Total persons (XTOT)", "XTOT", True),
        ("Children <17 (n24)", "n24", True),
    ]

    # Compute national and sum-of-states for each
    results = []
    for label, var, is_count in bystander_vars:
        if var not in tmd.columns:
            continue
        nat = float((s006 * tmd[var].values).sum())
        if abs(nat) < 1:
            continue
        sos = 0.0
        for _st, w in state_weights.items():
            sos += float((w * tmd[var].values).sum())
        diff_pct = (sos / nat - 1) * 100
        results.append((label, var, is_count, nat, sos, diff_pct))

    # Sort by absolute distortion
    results.sort(key=lambda x: -abs(x[5]))

    lines.append(
        "BYSTANDER CHECK: UNTARGETED VARIABLES" f" ({n_loaded} areas):"
    )
    lines.append(
        "  Variables NOT directly targeted — distortion"
        " from weight adjustments."
    )
    lines.append(
        f"  {'Variable':<30} {'National':>16}"
        f" {'Sum-of-Areas':>16} {'Diff':>14} {'Diff%':>8}"
    )
    lines.append("  " + "-" * 86)
    for label, _var, is_count, nat, sos, diff_pct in results:
        diff = sos - nat
        flag = " ***" if abs(diff_pct) > 2 else ""
        if is_count:
            lines.append(
                f"  {label:<30} {nat:>16,.0f}"
                f" {sos:>16,.0f}"
                f" {diff:>+14,.0f}"
                f" {diff_pct:>+7.2f}%{flag}"
            )
        else:
            lines.append(
                f"  {label:<30}"
                f" ${nat / 1e9:>14.1f}B"
                f" ${sos / 1e9:>14.1f}B"
                f" ${diff / 1e9:>+12.2f}B"
                f" {diff_pct:>+7.2f}%{flag}"
            )

    n_flagged = sum(1 for *_, d in results if abs(d) > 2)
    lines.append("")
    if n_flagged:
        lines.append(
            f"  *** = {n_flagged} variables with"
            f" >2% aggregation distortion"
        )
    else:
        lines.append(
            "  All untargeted variables within" + " 2% aggregation tolerance."
        )
    lines.append("")

    return lines


def _bystander_by_bin(tmd, s006, state_weights, n_loaded, target_dir):
    """
    Per-AGI-bin bystander analysis.

    For both targeted and untargeted variable-bin combinations,
    computes cross-area aggregation distortion. Shows whether
    dropped targets (variable-bin combos excluded from the recipe)
    are still well-behaved or drifting.
    """
    lines = []

    # Read one target file to identify what's targeted
    first_area = next(iter(state_weights.keys()), None)
    if first_area is None:
        return lines
    tgt_path = target_dir / f"{first_area}_targets.csv"
    if not tgt_path.exists():
        return lines
    tgt_df = pd.read_csv(tgt_path, comment="#")

    # Build set of targeted (varname, count, agilo, agihi, fstatus)
    targeted_set = set()
    for _, row in tgt_df.iterrows():
        targeted_set.add(
            (
                row.varname,
                int(row["count"]),
                float(row.agilo),
                float(row.agihi),
                int(row.fstatus),
            )
        )

    # Get AGI bins from target file (exclude the all-bins row)
    agi_pairs = set()
    for _, row in tgt_df.iterrows():
        lo, hi = float(row.agilo), float(row.agihi)
        if lo < -1e10 and hi > 1e10:
            continue  # skip all-bins
        agi_pairs.add((lo, hi))
    agi_bins = sorted(agi_pairs)

    if not agi_bins or "c00100" not in tmd.columns:
        return lines

    # Pre-compute AGI bin masks
    agi_masks = {}
    for lo, hi in agi_bins:
        agi_masks[(lo, hi)] = (tmd["c00100"].values >= lo) & (
            tmd["c00100"].values < hi
        )

    # Variables to check: recipe vars + key untargeted
    # (label, varname, count_type)
    check_vars = [
        # Targeted amount variables
        ("AGI", "c00100", 0),
        ("Wages", "e00200", 0),
        ("Interest", "e00300", 0),
        ("Pensions tot", "e01500", 0),
        ("Social Security", "e02400", 0),
        ("SALT ded", "c18300", 0),
        ("Partnership", "e26270", 0),
        # Count targets
        ("Returns", "c00100", 1),
        ("Wage nz-count", "e00200", 2),
        # Key untargeted (for bystander effect)
        ("Income tax", "iitax", 0),
        ("Cap gains net", "capgains_net", 0),
        ("Sch C income", "e00900", 0),
        ("IRA distrib", "e01400", 0),
        ("Pensions txbl", "e01700", 0),
        ("Sch E net", "e02000", 0),
        ("Tax-exempt int", "e00400", 0),
    ]

    results = []
    for label, varname, cnt in check_vars:
        if varname not in tmd.columns:
            continue
        var_vals = tmd[varname].values

        for lo, hi in agi_bins:
            in_bin = agi_masks[(lo, hi)]

            # Build the value array for this variable-bin-count combo
            if cnt == 0:
                vals = var_vals * in_bin
            elif cnt == 1:
                vals = in_bin.astype(float)
            elif cnt == 2:
                vals = ((var_vals != 0) & in_bin).astype(float)
            else:
                continue

            # National total
            nat = float((s006 * vals).sum())
            if abs(nat) < 1:
                continue

            # Sum-of-areas total
            sos = 0.0
            for w in state_weights.values():
                sos += float((w * vals).sum())
            diff_pct = (sos / nat - 1) * 100

            # Check if this combo is targeted (fstatus=0 for amounts/counts)
            is_targeted = (varname, cnt, lo, hi, 0) in targeted_set

            results.append(
                {
                    "label": label,
                    "varname": varname,
                    "cnt": cnt,
                    "lo": lo,
                    "hi": hi,
                    "nat": nat,
                    "sos": sos,
                    "diff_pct": diff_pct,
                    "targeted": is_targeted,
                }
            )

    if not results:
        return lines

    # Sort by absolute distortion
    results.sort(key=lambda x: -abs(x["diff_pct"]))

    # Show top 30 by distortion
    n_show = min(30, len(results))
    n_total = len(results)
    cnt_labels = {0: "amt", 1: "returns", 2: "nz-count"}
    lines.append(f"PER-BIN DISTORTION ANALYSIS ({n_loaded} areas):")
    lines.append(
        f"  T = targeted, U = untargeted."
        f" Sorted by |distortion|,"
        f" top {n_show} out of {n_total} combinations."
    )
    col_hdr = (
        f"  {'':>1} {'Variable':<18} {'Type':<10}"
        f" {'AGI Bin':<16}"
        f" {'National':>14} {'Sum-Areas':>14}"
        f" {'Diff':>14} {'Diff%':>8}"
    )
    lines.append(col_hdr)
    lines.append("  " + "-" * (len(col_hdr) - 2))

    for r in results[:n_show]:
        mark = "T" if r["targeted"] else "U"
        ct_label = cnt_labels.get(r["cnt"], f"cnt{r['cnt']}")
        bin_label = _fmt_agi_bin(r["lo"], r["hi"])
        flag = " ***" if abs(r["diff_pct"]) > 5 else ""
        diff = r["sos"] - r["nat"]

        if r["cnt"] == 0:
            nat_s = f"${r['nat'] / 1e9:.2f}B"
            sos_s = f"${r['sos'] / 1e9:.2f}B"
            diff_s = f"${diff / 1e9:+.2f}B"
        else:
            nat_s = f"{r['nat']:,.0f}"
            sos_s = f"{r['sos']:,.0f}"
            diff_s = f"{diff:+,.0f}"

        lines.append(
            f"  {mark} {r['label']:<18} {ct_label:<10}"
            f" {bin_label:<16}"
            f" {nat_s:>14} {sos_s:>14}"
            f" {diff_s:>14}"
            f" {r['diff_pct']:>+7.2f}%{flag}"
        )

    # Summary
    untargeted = [r for r in results if not r["targeted"]]
    targeted_results = [r for r in results if r["targeted"]]
    lines.append("")
    lines.append(
        f"  Targeted: {len(targeted_results)}"
        f"   Untargeted: {len(untargeted)}"
    )
    if untargeted:
        worst_d = max(abs(r["diff_pct"]) for r in untargeted)
        mean_d = np.mean([abs(r["diff_pct"]) for r in untargeted])
        n_bad = sum(1 for r in untargeted if abs(r["diff_pct"]) > 5)
        lines.append(
            f"  Untargeted distortion:"
            f" mean={mean_d:.2f}%, worst={worst_d:.2f}%"
        )
        if n_bad:
            lines.append(
                f"  *** = {n_bad} untargeted combos" f" with >5% distortion"
            )
        else:
            lines.append("  All untargeted combos within 5% distortion.")
    if targeted_results:
        worst_t = max(abs(r["diff_pct"]) for r in targeted_results)
        mean_t = np.mean([abs(r["diff_pct"]) for r in targeted_results])
        lines.append(
            f"  Targeted-bin distortion:"
            f" mean={mean_t:.2f}%, worst={worst_t:.2f}%"
        )
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Cross-area quality summary report",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help=(
            "'states', 'cds', or comma-separated area codes"
            " (default: all states)"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Save report to file. No argument = auto-generate"
            " filename in weight directory. Or specify a path."
        ),
    )
    args = parser.parse_args()

    areas = None
    weight_dir = None
    target_dir = None
    scope_label = "states"
    if args.scope:
        scope_lower = args.scope.lower().strip()
        if scope_lower == "cds":
            weight_dir = CD_WEIGHT_DIR
            target_dir = CD_TARGET_DIR
            scope_label = "cds"
        elif scope_lower == "states":
            weight_dir = STATE_WEIGHT_DIR
            target_dir = STATE_TARGET_DIR
            scope_label = "states"
        else:
            codes = [s.strip().upper() for s in args.scope.split(",")]
            # Detect CDs vs states by code length
            if codes and len(codes[0]) > 2:
                weight_dir = CD_WEIGHT_DIR
                target_dir = CD_TARGET_DIR
                scope_label = f"cds ({len(codes)} selected)"
            else:
                scope_label = f"states ({len(codes)} selected)"
            areas = codes

    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR
        target_dir = STATE_TARGET_DIR

    report = generate_report(
        areas,
        weight_dir=weight_dir,
        target_dir=target_dir,
        scope_label=scope_label,
    )
    print(report)

    # Save to file
    if args.output is not None:
        if args.output == "auto":
            output_path = weight_dir / "quality_report.txt"
        else:
            output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
