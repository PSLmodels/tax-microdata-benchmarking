# pylint: disable=import-outside-toplevel,inconsistent-quotes
"""
Developer mode — iterative LP/QP auto-relaxation for area weights.

Runs an automated relaxation cascade on each area to find the
least-invasive parameter adjustments that make the solver succeed.
Produces a per-area override YAML file committed to the repo.

Relaxation cascade (least invasive first):
  Level 0: Full spec, default params
  Level 1: Drop unreachable targets (automatic in solver)
  Level 2: Reduce slack penalty on problematic targets
  Level 3: Drop specific targets identified by LP feasibility
  Level 4: Raise multiplier cap
  Level 5: Raise constraint tolerance

Usage:
    python -m tmd.areas.developer_tools --scope cds --workers 16
    python -m tmd.areas.developer_tools --scope cds --lp-only
    python -m tmd.areas.developer_tools --scope NY12 --verbose
"""

import argparse
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    AREA_MULTIPLIER_MIN,
    AREA_SLACK_PENALTY,
    CD_MULTIPLIER_MAX,
    CD_TARGET_DIR,
    CD_WEIGHT_DIR,
    STATE_TARGET_DIR,
    STATE_WEIGHT_DIR,
    _assign_slack_penalties,
    _build_constraint_matrix,
    _check_feasibility,
    _drop_impossible_targets,
    _load_taxcalc_data,
    _solve_area_qp,
)
from tmd.areas.solver_overrides import write_overrides

# --- Configuration ---

_RECIPES = Path(__file__).parent / "prepare" / "recipes"
_CD_OVERRIDES = _RECIPES / "cd_solver_overrides.yaml"
_STATE_OVERRIDES = _RECIPES / "state_solver_overrides.yaml"

# Max targets to drop per area before giving up
_MAX_DROPS = 10

# Relaxation cascade parameters
_LEVEL_LABELS = {
    0: "default",
    1: "auto-drop unreachable",
    2: "reduced slack",
    3: "drop targets",
    4: "raise multiplier cap",
    5: "raise tolerance",
}


def _run_lp_feasibility(
    area,
    vdf,
    target_dir,
    multiplier_max,
    constraint_tol=AREA_CONSTRAINT_TOL,
):
    """
    Run LP feasibility check for one area.

    Returns dict with:
        area, feasible, n_targets, n_infeasible,
        worst_labels [(label, slack, rel_slack)]
    """
    out = io.StringIO()
    B_csc, targets, labels, _pop = _build_constraint_matrix(
        area, vdf, out, target_dir=target_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc,
        targets,
        labels,
        out,
        multiplier_max=multiplier_max,
        constraint_tol=constraint_tol,
    )
    n_records = B_csc.shape[1]
    info = _check_feasibility(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        multiplier_min=AREA_MULTIPLIER_MIN,
        multiplier_max=multiplier_max,
        out=out,
    )
    return {
        "area": area,
        "feasible": info["feasible"],
        "n_targets": len(targets),
        "worst_labels": info.get("worst_labels", []),
        "log": out.getvalue(),
    }


def _solve_with_params(
    area,
    vdf,
    target_dir,
    multiplier_max,
    constraint_tol=AREA_CONSTRAINT_TOL,
    slack_penalty=AREA_SLACK_PENALTY,
    drop_labels=None,
):
    """
    Solve one area with given parameters.

    Returns dict with solve results.
    """
    out = io.StringIO()
    B_csc, targets, labels, _pop_share = _build_constraint_matrix(
        area, vdf, out, target_dir=target_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc,
        targets,
        labels,
        out,
        multiplier_max=multiplier_max,
        constraint_tol=constraint_tol,
    )

    # Apply explicit drops
    if drop_labels:
        from tmd.areas.solver_overrides import should_drop_target

        keep = [not should_drop_target(lbl, drop_labels) for lbl in labels]
        n_drop = sum(1 for k in keep if not k)
        if n_drop > 0:
            from scipy.sparse import csc_matrix

            keep_arr = np.array(keep)
            B_dense = B_csc.toarray()
            B_csc = csc_matrix(B_dense[keep_arr, :])
            targets = targets[keep_arr]
            labels = [lbl for lbl, k in zip(labels, keep) if k]

    n_records = B_csc.shape[1]

    # Assign slack penalties
    per_constraint_penalties = _assign_slack_penalties(
        labels, default_penalty=slack_penalty
    )

    # Solve QP
    x_opt, _s_lo, _s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        slack_penalties=per_constraint_penalties,
        multiplier_min=AREA_MULTIPLIER_MIN,
        multiplier_max=multiplier_max,
        out=out,
    )

    # Compute violations
    achieved = np.asarray(B_csc @ x_opt).ravel()
    rel_errors = np.abs(achieved - targets) / np.maximum(np.abs(targets), 1.0)
    eps = 1e-9
    viol_mask = rel_errors > constraint_tol + eps
    n_violated = int(viol_mask.sum())
    max_viol = float(rel_errors.max()) if len(rel_errors) > 0 else 0

    # Identify violated labels
    violated_labels = []
    if n_violated > 0:
        for idx in np.where(viol_mask)[0]:
            violated_labels.append((labels[idx], float(rel_errors[idx])))
        violated_labels.sort(key=lambda x: -x[1])

    rmse = float(np.sqrt(np.mean((x_opt - 1.0) ** 2)))

    return {
        "area": area,
        "status": info["status"],
        "n_targets": len(targets),
        "n_violated": n_violated,
        "max_viol": max_viol,
        "rmse": rmse,
        "violated_labels": violated_labels,
        "log": out.getvalue(),
    }


def _relaxation_cascade(
    area,
    vdf,
    target_dir,
    multiplier_max,
    constraint_tol=AREA_CONSTRAINT_TOL,
    verbose=False,
):
    """
    Run relaxation cascade for one area.

    Tries increasingly aggressive relaxations until the area
    solves with acceptable violations.

    Returns (level, overrides_dict, solve_result).
    """
    # Level 0: default params
    result = _solve_with_params(
        area,
        vdf,
        target_dir,
        multiplier_max=multiplier_max,
        constraint_tol=constraint_tol,
    )

    if result["n_violated"] == 0 and "Solved" in result["status"]:
        return 0, {}, result

    if verbose:
        print(
            f"  {area}: Level 0 — {result['n_violated']} violations,"
            f" status={result['status']}"
        )

    # Level 2: LP feasibility to identify problematic constraints,
    # then drop them one at a time
    lp_info = _run_lp_feasibility(
        area, vdf, target_dir, multiplier_max, constraint_tol
    )

    drop_labels = []
    if not lp_info["feasible"] or result["n_violated"] > 0:
        # Get worst constraints from LP
        worst = lp_info.get("worst_labels", [])
        # Also add violated labels from QP
        for lbl, _err in result.get("violated_labels", []):
            if lbl not in [w[0] for w in worst]:
                worst.append((lbl, 0, 0))

        # Level 3: Drop targets iteratively
        for i, (lbl, *_) in enumerate(worst):
            if i >= _MAX_DROPS:
                break
            drop_labels.append(lbl)
            result = _solve_with_params(
                area,
                vdf,
                target_dir,
                multiplier_max=multiplier_max,
                constraint_tol=constraint_tol,
                drop_labels=drop_labels,
            )
            if verbose:
                print(
                    f"  {area}: Level 3 — dropped {len(drop_labels)},"
                    f" {result['n_violated']} violations"
                )
            if result["n_violated"] == 0 and "Solved" in result["status"]:
                overrides = {"drop_targets": list(drop_labels)}
                return 3, overrides, result

    # Level 4: Raise multiplier cap
    higher_cap = min(multiplier_max * 2, 200)
    result = _solve_with_params(
        area,
        vdf,
        target_dir,
        multiplier_max=higher_cap,
        constraint_tol=constraint_tol,
        drop_labels=drop_labels,
    )
    if result["n_violated"] == 0 and "Solved" in result["status"]:
        overrides = {"multiplier_max": higher_cap}
        if drop_labels:
            overrides["drop_targets"] = list(drop_labels)
        return 4, overrides, result
    if verbose:
        print(
            f"  {area}: Level 4 — cap={higher_cap},"
            f" {result['n_violated']} violations"
        )

    # Level 5: Raise tolerance
    higher_tol = constraint_tol * 2
    result = _solve_with_params(
        area,
        vdf,
        target_dir,
        multiplier_max=higher_cap,
        constraint_tol=higher_tol,
        drop_labels=drop_labels,
    )
    overrides = {
        "multiplier_max": higher_cap,
        "constraint_tol": higher_tol,
    }
    if drop_labels:
        overrides["drop_targets"] = list(drop_labels)

    level = 5 if result["n_violated"] == 0 else 5
    if verbose:
        status = "OK" if result["n_violated"] == 0 else "STILL FAILING"
        print(
            f"  {area}: Level 5 — tol={higher_tol},"
            f" {result['n_violated']} violations ({status})"
        )

    return level, overrides, result


# Module-level cache for worker processes
_WORKER_VDF = None
_WORKER_TARGET_DIR = None
_WORKER_MULTIPLIER_MAX = None
_WORKER_CONSTRAINT_TOL = None
_WORKER_VERBOSE = False


def _init_worker(target_dir, multiplier_max, constraint_tol, verbose):
    """Initialize worker process with TMD data."""
    # pylint: disable=global-statement
    global _WORKER_VDF, _WORKER_TARGET_DIR
    global _WORKER_MULTIPLIER_MAX, _WORKER_CONSTRAINT_TOL
    global _WORKER_VERBOSE
    # pylint: enable=global-statement
    _WORKER_TARGET_DIR = Path(target_dir)
    _WORKER_MULTIPLIER_MAX = multiplier_max
    _WORKER_CONSTRAINT_TOL = constraint_tol
    _WORKER_VERBOSE = verbose
    if _WORKER_VDF is None:
        _WORKER_VDF = _load_taxcalc_data()


def _process_area(area):
    """Process one area through the relaxation cascade."""
    _init_worker(
        _WORKER_TARGET_DIR,
        _WORKER_MULTIPLIER_MAX,
        _WORKER_CONSTRAINT_TOL,
        _WORKER_VERBOSE,
    )
    level, overrides, result = _relaxation_cascade(
        area,
        _WORKER_VDF,
        _WORKER_TARGET_DIR,
        multiplier_max=_WORKER_MULTIPLIER_MAX,
        constraint_tol=_WORKER_CONSTRAINT_TOL,
        verbose=_WORKER_VERBOSE,
    )
    return (
        area,
        level,
        overrides,
        {
            "status": result["status"],
            "n_targets": result["n_targets"],
            "n_violated": result["n_violated"],
            "max_viol": result["max_viol"],
            "rmse": result["rmse"],
        },
    )


def _lp_only_area(area):
    """Run LP feasibility check only for one area."""
    _init_worker(
        _WORKER_TARGET_DIR,
        _WORKER_MULTIPLIER_MAX,
        _WORKER_CONSTRAINT_TOL,
        False,
    )
    return _run_lp_feasibility(
        area,
        _WORKER_VDF,
        _WORKER_TARGET_DIR,
        _WORKER_MULTIPLIER_MAX,
        _WORKER_CONSTRAINT_TOL,
    )


def run_developer_tools(
    scope="cds",
    num_workers=1,
    lp_only=False,
    verbose=False,
):
    """
    Run developer mode: iterative relaxation for all areas.

    Parameters
    ----------
    scope : str
        'cds', 'states', or comma-separated area codes.
    num_workers : int
        Number of parallel workers.
    lp_only : bool
        If True, only run LP feasibility (no QP solve).
    verbose : bool
        Print per-area progress.
    """
    scope_lower = scope.lower().strip()
    first_code = scope.split(",")[0].strip()
    is_cd = scope_lower == "cds" or len(first_code) > 2

    if is_cd:
        target_dir = CD_TARGET_DIR
        weight_dir = CD_WEIGHT_DIR
        override_path = _CD_OVERRIDES
        multiplier_max = CD_MULTIPLIER_MAX
    else:
        target_dir = STATE_TARGET_DIR
        weight_dir = STATE_WEIGHT_DIR
        override_path = _STATE_OVERRIDES
        multiplier_max = 25.0

    constraint_tol = AREA_CONSTRAINT_TOL

    # List areas from target files
    areas = sorted(
        p.name.split("_")[0] for p in target_dir.glob("*_targets.csv")
    )
    if scope_lower not in ("cds", "states", "all"):
        codes = [c.strip().lower() for c in scope.split(",")]
        areas = [a for a in areas if a in codes]

    n_areas = len(areas)
    print(f"Developer mode: {n_areas} areas, {num_workers} workers")
    print(f"Target dir: {target_dir}")
    t_start = time.time()

    if lp_only:
        # LP feasibility only
        print("Running LP feasibility checks...")
        results = []
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(
                str(target_dir),
                multiplier_max,
                constraint_tol,
                False,
            ),
        ) as executor:
            futures = {executor.submit(_lp_only_area, a): a for a in areas}
            for future in as_completed(futures):
                results.append(future.result())

        feasible = sum(1 for r in results if r["feasible"])
        infeasible = [r for r in results if not r["feasible"]]
        elapsed = time.time() - t_start

        print(
            f"\nLP Feasibility: {feasible}/{n_areas} feasible"
            f" ({elapsed:.1f}s)"
        )
        if infeasible:
            print(f"\nInfeasible areas ({len(infeasible)}):")
            for r in sorted(infeasible, key=lambda x: x["area"]):
                worst = r["worst_labels"][:3]
                worst_str = ", ".join(
                    f"{lbl} ({rel:.1%})" for lbl, _, rel in worst
                )
                print(f"  {r['area']}: {worst_str}")
        return

    # Full relaxation cascade
    print("Running relaxation cascade...")
    area_results = []
    area_overrides = {}
    level_counts = {i: 0 for i in range(6)}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            str(target_dir),
            multiplier_max,
            constraint_tol,
            verbose,
        ),
    ) as executor:
        futures = {executor.submit(_process_area, a): a for a in areas}
        completed = 0
        for future in as_completed(futures):
            area, level, overrides, stats = future.result()
            completed += 1
            area_results.append((area, level, overrides, stats))
            level_counts[level] = level_counts.get(level, 0) + 1
            if overrides:
                area_overrides[area] = overrides

            # Progress
            if completed % 10 == 0 or completed == n_areas:
                elapsed = time.time() - t_start
                sys.stdout.write(
                    f"\r  {completed}/{n_areas}" f" [{elapsed:.0f}s]"
                )
                sys.stdout.flush()

    sys.stdout.write("\n")
    elapsed = time.time() - t_start

    # Write override file
    defaults = {
        "multiplier_max": multiplier_max,
        "constraint_tol": constraint_tol,
    }
    write_overrides(override_path, defaults, area_overrides)

    # Summary
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"Override file: {override_path}")
    print("\nRelaxation levels:")
    for level in sorted(level_counts.keys()):
        cnt = level_counts[level]
        if cnt > 0:
            label = _LEVEL_LABELS.get(level, f"level {level}")
            print(f"  Level {level} ({label}): {cnt} areas")

    n_overrides = len(area_overrides)
    if n_overrides > 0:
        print(f"\n{n_overrides} areas need overrides:")
        for area, level, overrides, stats in sorted(
            area_results, key=lambda x: -x[1]
        ):
            if not overrides:
                continue
            drops = len(overrides.get("drop_targets", []))
            extra = []
            if drops:
                extra.append(f"{drops} dropped")
            if "multiplier_max" in overrides:
                extra.append(f"cap={overrides['multiplier_max']}")
            if "constraint_tol" in overrides:
                extra.append(f"tol={overrides['constraint_tol']}")
            detail = ", ".join(extra)
            print(
                f"  {area}: level {level},"
                f" {stats['n_violated']} violations,"
                f" RMSE={stats['rmse']:.3f}"
                f" ({detail})"
            )

    # Write developer report
    report_path = weight_dir / "developer_report.txt"
    _write_report(
        report_path,
        area_results,
        level_counts,
        n_areas,
        elapsed,
        override_path,
    )
    print(f"Report: {report_path}")


def _write_report(
    report_path, area_results, level_counts, n_areas, elapsed, override_path
):
    """Write detailed developer mode report."""
    from datetime import datetime

    lines = []
    lines.append("=" * 70)
    lines.append("DEVELOPER MODE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Areas: {n_areas}, Time: {elapsed:.0f}s")
    lines.append(f"Override file: {override_path}")
    lines.append("=" * 70)
    lines.append("")

    # Level summary
    lines.append("RELAXATION LEVELS:")
    for level in sorted(level_counts.keys()):
        cnt = level_counts[level]
        if cnt > 0:
            label = _LEVEL_LABELS.get(level, f"level {level}")
            lines.append(f"  Level {level} ({label}): {cnt}")
    lines.append("")

    # Per-area detail (only non-level-0)
    problem_areas = [
        (a, lv, ov, st) for a, lv, ov, st in area_results if lv > 0
    ]
    if problem_areas:
        lines.append("AREAS REQUIRING RELAXATION:")
        lines.append(
            f"  {'Area':<6} {'Lvl':>3} {'Viol':>4}"
            f" {'RMSE':>7} {'Drops':>5} {'Details'}"
        )
        lines.append("  " + "-" * 60)
        for area, level, overrides, stats in sorted(
            problem_areas, key=lambda x: (-x[1], x[0])
        ):
            drops = len(overrides.get("drop_targets", []))
            details = []
            for dt in overrides.get("drop_targets", [])[:3]:
                details.append(f"drop:{dt}")
            if "multiplier_max" in overrides:
                details.append(f"cap={overrides['multiplier_max']}")
            if "constraint_tol" in overrides:
                details.append(f"tol={overrides['constraint_tol']}")
            detail_str = "; ".join(details)
            lines.append(
                f"  {area:<6} {level:>3}"
                f" {stats['n_violated']:>4}"
                f" {stats['rmse']:>7.3f}"
                f" {drops:>5}"
                f" {detail_str}"
            )
        lines.append("")

    # All areas summary
    lines.append("ALL AREAS:")
    lines.append(
        f"  {'Area':<6} {'Lvl':>3} {'Tgts':>4}"
        f" {'Viol':>4} {'RMSE':>7} {'Status'}"
    )
    lines.append("  " + "-" * 40)
    for area, level, _ov, stats in sorted(area_results, key=lambda x: x[0]):
        lines.append(
            f"  {area:<6} {level:>3}"
            f" {stats['n_targets']:>4}"
            f" {stats['n_violated']:>4}"
            f" {stats['rmse']:>7.3f}"
            f" {stats['status']}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def target_difficulty(area, target_dir=None):
    """
    Compute target difficulty for a single area.

    For each target, compares the proportionate value (pop_share ×
    national total) against the actual target.  Large gaps indicate
    targets that force the solver to distort weights heavily.

    Returns a DataFrame sorted by |gap_pct| descending, and prints
    a formatted report.
    """
    if target_dir is None:
        target_dir = CD_TARGET_DIR

    vardf = _load_taxcalc_data()
    s006 = vardf["s006"].values
    n = len(vardf)

    targets = pd.read_csv(target_dir / f"{area.lower()}_targets.csv")

    # Get pop_share from XTOT row
    xtot_row = targets[targets["varname"] == "XTOT"].iloc[0]
    national_pop = float((s006 * vardf["XTOT"].values).sum())
    pop_share = xtot_row["target"] / national_pop

    results = []
    for _, row in targets.iterrows():
        if row["varname"] == "XTOT":
            results.append(
                {
                    "varname": "XTOT",
                    "count": 0,
                    "fstatus": 0,
                    "agilo": row["agilo"],
                    "agihi": row["agihi"],
                    "national": national_pop,
                    "proportionate": xtot_row["target"],
                    "target": row["target"],
                    "gap": 0,
                    "gap_pct": 0,
                }
            )
            continue

        # Build mask (same logic as _build_constraint_matrix)
        mask = np.ones(n, dtype=float)
        if row["scope"] == 1:
            mask *= (vardf["data_source"] == 1).values.astype(float)
        in_bin = (vardf["c00100"] >= row["agilo"]) & (
            vardf["c00100"] < row["agihi"]
        )
        mask *= in_bin.values.astype(float)
        if row["fstatus"] > 0:
            mask *= (vardf["MARS"] == row["fstatus"]).values.astype(float)

        if row["count"] == 0:
            var_vals = vardf[row["varname"]].astype(float).values
        elif row["count"] == 1:
            var_vals = np.ones(n, dtype=float)
        elif row["count"] == 2:
            var_vals = (vardf[row["varname"]] != 0).astype(float).values
        else:
            var_vals = np.ones(n, dtype=float)

        nat_val = float((s006 * mask * var_vals).sum())
        prop_val = pop_share * nat_val
        tgt_val = row["target"]
        gap = tgt_val - prop_val
        gap_pct = (tgt_val / prop_val - 1) * 100 if abs(prop_val) > 1 else 0

        results.append(
            {
                "varname": row["varname"],
                "count": int(row["count"]),
                "fstatus": int(row["fstatus"]),
                "agilo": row["agilo"],
                "agihi": row["agihi"],
                "national": nat_val,
                "proportionate": prop_val,
                "target": tgt_val,
                "gap": gap,
                "gap_pct": gap_pct,
            }
        )

    rdf = pd.DataFrame(results)
    rdf["abs_gap_pct"] = rdf["gap_pct"].abs()
    rdf = rdf.sort_values("abs_gap_pct", ascending=False)

    # Print formatted report
    cnt_labels = {0: "amt", 1: "returns", 2: "nz-count"}
    fs_labels = {0: "", 1: " single", 2: " MFJ", 4: " HoH"}

    def _fmt_agi(lo, hi):
        if lo < -1e10 and hi > 1e10:
            return "all"
        if lo < -1e10:
            return f"<${hi / 1000:.0f}K"
        if hi > 1e10:
            return f"${lo / 1000:.0f}K+"
        return f"${lo / 1000:.0f}K-${hi / 1000:.0f}K"

    print(f"\nTARGET DIFFICULTY: {area.upper()} (pop_share={pop_share:.6f})")
    print("Proportionate = what area would get if it looked like the nation")
    print(
        f"\n{'Target':<45} {'Proportionate':>14}"
        f" {'Target':>14} {'Gap%':>8}"
    )
    print("-" * 83)

    for _, r in rdf.iterrows():
        cnt = cnt_labels.get(r["count"], f"c{r['count']}")
        fs = fs_labels.get(r["fstatus"], "")
        agi = _fmt_agi(r["agilo"], r["agihi"])
        label = f"{r['varname']} {cnt}{fs} {agi}"

        if r["count"] == 0:
            print(
                f"{label:<45}"
                f" ${r['proportionate'] / 1e6:>11.1f}M"
                f" ${r['target'] / 1e6:>11.1f}M"
                f" {r['gap_pct']:>+7.1f}%"
            )
        else:
            print(
                f"{label:<45}"
                f" {r['proportionate']:>13,.0f}"
                f" {r['target']:>13,.0f}"
                f" {r['gap_pct']:>+7.1f}%"
            )

    n_easy = (rdf["abs_gap_pct"] < 5).sum()
    n_mod = ((rdf["abs_gap_pct"] >= 5) & (rdf["abs_gap_pct"] < 20)).sum()
    n_hard = ((rdf["abs_gap_pct"] >= 20) & (rdf["abs_gap_pct"] < 50)).sum()
    n_vhard = (rdf["abs_gap_pct"] >= 50).sum()
    print(
        f"\nDifficulty: {n_easy} easy (<5%),"
        f" {n_mod} moderate (5-20%),"
        f" {n_hard} hard (20-50%),"
        f" {n_vhard} very hard (>50%)"
    )
    print(
        f"Mean |gap%|: {rdf['abs_gap_pct'].mean():.1f}%,"
        f" median: {rdf['abs_gap_pct'].median():.1f}%"
    )

    return rdf


def dual_analysis(area, target_dir=None):
    """
    Solve a single area and print dual (shadow price) analysis.

    High dual values identify constraints that are binding and
    expensive — even targets with moderate gap% may have high duals
    because they conflict with other targets.

    Usage:
        python -m tmd.areas.developer_tools --dual AL01
    """
    if target_dir is None:
        target_dir = CD_TARGET_DIR

    vardf = _load_taxcalc_data()
    n_records = len(vardf)
    out = io.StringIO()

    B_csc, targets, labels, _pop_share = _build_constraint_matrix(
        area.lower(), vardf, out, target_dir=target_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc,
        targets,
        labels,
        out,
        multiplier_max=CD_MULTIPLIER_MAX,
    )

    m = len(targets)
    slack_pens = _assign_slack_penalties(labels)

    x_opt, s_lo, s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        multiplier_max=CD_MULTIPLIER_MAX,
        slack_penalties=slack_pens,
        out=out,
    )

    dual = info.get("dual")
    if dual is None:
        print("No dual variables available from solver.")
        return

    z_upper = dual[:m]
    z_lower = dual[m : 2 * m]
    z_combined = np.maximum(np.abs(z_upper), np.abs(z_lower))

    Bx = B_csc @ x_opt
    rel_err = np.where(
        np.abs(targets) > 1,
        (Bx - targets) / np.abs(targets) * 100,
        0,
    )

    rdf = pd.DataFrame(
        [
            {
                "label": labels[j],
                "z": z_combined[j],
                "err": rel_err[j],
                "slack": max(s_lo[j], s_hi[j]),
            }
            for j in range(m)
        ]
    )
    rdf = rdf.sort_values("z", ascending=False)

    print(f"\nDUAL ANALYSIS: {area.upper()}")
    print(
        f"Status: {info['status']},"
        f" {info['iterations']} iters,"
        f" {info['solve_time']:.1f}s"
    )
    print("\nShadow prices: higher = more expensive to satisfy.")
    print("Targets at ±0.500% error are binding (at tolerance boundary).")

    print(f"\n{'Label':<55} {'|Dual|':>10} {'Err%':>8}")
    print("-" * 75)
    for _, r in rdf.iterrows():
        binding = " *" if abs(r["err"]) >= 0.499 else ""
        print(
            f"{r['label'][:55]:<55}"
            f" {r['z']:>10.2f}"
            f" {r['err']:>+7.3f}%{binding}"
        )

    n_binding = (rdf["err"].abs() >= 0.499).sum()
    print(f"\n{n_binding}/{m} constraints binding (at tolerance boundary)")


def main():
    parser = argparse.ArgumentParser(
        description="Developer mode — auto-relaxation for area weights",
    )
    parser.add_argument(
        "--scope",
        default="cds",
        help="'cds', 'states', or comma-separated area codes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--lp-only",
        action="store_true",
        help="Only run LP feasibility check (no QP solve)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-area relaxation progress",
    )
    parser.add_argument(
        "--difficulty",
        metavar="AREA",
        help="Print target difficulty table for a single area",
    )
    parser.add_argument(
        "--dual",
        metavar="AREA",
        help="Solve a single area and print dual (shadow price) analysis",
    )
    args = parser.parse_args()

    scope_lower = args.scope.lower()
    if scope_lower in ("cds", "cd"):
        tdir = CD_TARGET_DIR
    elif scope_lower in ("states", "state"):
        tdir = STATE_TARGET_DIR
    else:
        tdir = CD_TARGET_DIR

    if args.difficulty:
        target_difficulty(args.difficulty, target_dir=tdir)
        return

    if args.dual:
        dual_analysis(args.dual, target_dir=tdir)
        return

    run_developer_tools(
        scope=args.scope,
        num_workers=args.workers,
        lp_only=args.lp_only,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
