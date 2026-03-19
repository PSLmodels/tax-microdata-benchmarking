# pylint: disable=import-outside-toplevel,inconsistent-quotes
"""
Cross-state quality summary report.

Parses solver logs for all states and produces a summary showing:
  - Solve status and timing
  - Target accuracy (hit rate, mean/max error)
  - Weight distortion (RMSE, percentiles)
  - Violated targets by variable
  - Weight exhaustion and cross-state aggregation diagnostics

Usage:
    python -m tmd.areas.quality_report
    python -m tmd.areas.quality_report --scope CA,WY
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    STATE_WEIGHT_DIR,
)
from tmd.areas.prepare.constants import ALL_STATES
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


def generate_report(areas=None, weight_dir=None):
    """Generate cross-state quality summary report."""
    if areas is None:
        areas = ALL_STATES
    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR

    rows = []
    for st in areas:
        logpath = weight_dir / f"{st.lower()}.log"
        info = parse_log(logpath)
        info["state"] = st
        rows.append(info)

    df = pd.DataFrame(rows)

    # Summary statistics
    solved = df[df["status"].isin(["Solved", "AlmostSolved"])]
    failed = df[df["status"] == "FAILED"]
    n_states = len(df)
    n_solved = len(solved)
    n_failed = len(failed)
    n_violated_states = (solved["n_violated"] > 0).sum()
    total_violated = solved["n_violated"].sum()

    tol_pct = AREA_CONSTRAINT_TOL * 100

    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-STATE QUALITY SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overall
    lines.append(f"States: {n_states}")
    lines.append(f"Solved: {n_solved}")
    lines.append(f"Failed: {n_failed}")
    if n_failed > 0:
        lines.append(f"  Failed: {', '.join(failed['state'].tolist())}")
    lines.append(
        f"States with violated targets: {n_violated_states}/{n_solved}"
    )
    if not solved.empty and "targets_total" in solved.columns:
        tpt = int(solved["targets_total"].iloc[0])
        tpt_sum = int(solved["targets_total"].sum())
    else:
        tpt, tpt_sum = "?", "?"
    lines.append(f"Total targets: {n_solved} states \u00d7 {tpt} = {tpt_sum}")
    lines.append(f"Total violated targets: {int(total_violated)}")
    lines.append("")

    # Target accuracy
    if not solved.empty and "mean_err" in solved.columns:
        lines.append("TARGET ACCURACY:")
        lines.append(
            f"  Per-state mean error: "
            f"avg across states={solved['mean_err'].mean():.4f}, "
            f"worst state={solved['mean_err'].max():.4f}"
        )
        lines.append(
            f"  Per-state max error:  "
            f"avg across states={solved['max_err'].mean():.4f}, "
            f"worst state={solved['max_err'].max():.4f}"
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

    # Per-state table
    lines.append("PER-STATE DETAIL:")
    lines.append(
        "  Err cols = |relative error| (fraction); "
        "weight cols = multiplier on national weight (1.0 = unchanged)"
    )
    header = (
        f"{'St':<4} {'Status':<14} {'Hit':>5} {'Tot':>5} "
        f"{'Viol':>5} {'MeanErr':>8} {'MaxErr':>8} "
        f"{'wRMSE':>7} {'wP05':>7} {'wMed':>7} "
        f"{'wP95':>7} {'wMax':>8} {'%zero':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.iterrows():
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
        lines.append(
            f"{row['state']:<4} {row['status']:<14} {hit:>5} {tot:>5} "
            f"{viol:>5} {me:>8.4f} {mx:>8.4f} "
            f"{rmse:>7.3f} {p5:>7.3f} {med:>7.3f} "
            f"{p95:>7.3f} {wmax:>8.1f} {pct_zero:>5.1f}%"
        )
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
                    "state": row["state"],
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
            states_with = sorted(vdf[vdf["varname"] == var]["state"].unique())
            lines.append(
                f"  {var}: {cnt} violations across "
                f"{len(states_with)} states"
            )
        lines.append("")

        state_counts = vdf["state"].value_counts().head(10)
        lines.append("STATES WITH MOST VIOLATIONS:")
        for st, cnt in state_counts.items():
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
                    f"  {r['state']:<4} {r['pct_err']:.3f}% "
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
                    f"  {r['state']:<4} {r['pct_err']:.3f}% "
                    f"target={r['target']:>12,.0f}  "
                    f"achieved={r['achieved']:>12,.0f}  "
                    f"miss={r['abs_miss']:>8,.0f}  "
                    f"{_humanize_desc(r['desc'])}"
                )
        lines.append("")

    # Weight diagnostics
    lines.extend(_weight_diagnostics(areas, weight_dir))

    report = "\n".join(lines)
    return report


def _weight_diagnostics(areas, weight_dir=None):
    """
    Combined weight diagnostics: exhaustion + national aggregation.

    Loads TMD data and state weight files once, reuses for both.
    Only reads the specific columns needed (not the full allvars).
    """
    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR

    from tmd.storage import STORAGE_FOLDER

    lines = []

    # Load national data
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
    n_records = len(tmd)
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

    # Load all state weights once
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
        return lines

    # Weight exhaustion
    usage = weight_sum / s006

    _sub = "sum of state weights / national weight"
    lines.append(f"WEIGHT EXHAUSTION ({_sub}):")
    lines.append(
        f"  A ratio of 1.0 means the record's national"
        f" weight is fully allocated across"
        f" {n_loaded} states."
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
            f"     top states: {top3}"
            f" ({len(st_wts)} nonzero of {n_loaded})"
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
        f"CROSS-STATE AGGREGATION vs NATIONAL TOTALS"
        f" for SELECTED VARIABLES ({n_loaded} states):"
    )
    lines.append(
        f"  {'Variable':<30} {'National':>16}"
        f" {'Sum-of-States':>16} {'Diff%':>8}"
    )
    lines.append("  " + "-" * 72)
    for label, var, is_count in check_vars:
        nat = national[var]
        sos = state_sums[var]
        if nat is None or nat == 0:
            continue
        diff_pct = (sos / nat - 1) * 100
        if is_count:
            lines.append(
                f"  {label:<30} {nat:>16,.0f}"
                f" {sos:>16,.0f} {diff_pct:>+7.2f}%"
            )
        else:
            lines.append(
                f"  {label:<30}"
                f" ${nat / 1e9:>14.1f}B"
                f" ${sos / 1e9:>14.1f}B"
                f" {diff_pct:>+7.2f}%"
            )
    lines.append("")

    # Bystander check: untargeted variables
    lines.extend(_bystander_check(tmd, s006, state_weights, n_loaded))

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
        if var == "XTOT":
            nat = float((s006 * tmd[var].values).sum())
        elif is_count:
            nat = float((s006 * tmd[var].values).sum())
        else:
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
        "BYSTANDER CHECK: UNTARGETED VARIABLES" f" ({n_loaded} states):"
    )
    lines.append(
        "  Variables NOT directly targeted — distortion"
        " from weight adjustments."
    )
    lines.append(
        f"  {'Variable':<30} {'National':>16}"
        f" {'Sum-of-States':>16} {'Diff%':>8}"
    )
    lines.append("  " + "-" * 72)
    for label, _var, is_count, nat, sos, diff_pct in results:
        flag = " ***" if abs(diff_pct) > 2 else ""
        if is_count:
            lines.append(
                f"  {label:<30} {nat:>16,.0f}"
                f" {sos:>16,.0f}"
                f" {diff_pct:>+7.2f}%{flag}"
            )
        else:
            lines.append(
                f"  {label:<30}"
                f" ${nat / 1e9:>14.1f}B"
                f" ${sos / 1e9:>14.1f}B"
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


def main():
    parser = argparse.ArgumentParser(
        description="Cross-state quality summary report",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Comma-separated state codes (default: all states)",
    )
    args = parser.parse_args()

    areas = None
    if args.scope:
        areas = [s.strip().upper() for s in args.scope.split(",")]

    report = generate_report(areas)
    print(report)


if __name__ == "__main__":
    main()
