# pylint: disable=import-outside-toplevel,global-statement
"""
Batch area weight optimization — parallel processing for all areas.

TMD data loaded once per worker (not once per area).
Progress reporting with ETA.
Uses concurrent.futures for clean parallel execution.

Usage:
    python -m tmd.areas.batch_weights --scope states --workers 8
    python -m tmd.areas.batch_weights --scope MN,CA,TX --workers 4
"""

import argparse
import io
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csc_matrix

# Module-level cache for TMD data (one per worker process)
_WORKER_VDF = None
_WORKER_POP = None
_WORKER_TARGET_DIR = None
_WORKER_WEIGHT_DIR = None
_WORKER_MULTIPLIER_MAX = None
_WORKER_OVERRIDES = None


def _init_worker(
    target_dir=None,
    weight_dir=None,
    multiplier_max=None,
    override_path=None,
):
    """Load TMD data once per worker process."""
    global _WORKER_VDF, _WORKER_POP
    global _WORKER_TARGET_DIR, _WORKER_WEIGHT_DIR
    global _WORKER_MULTIPLIER_MAX, _WORKER_OVERRIDES
    if target_dir is not None:
        _WORKER_TARGET_DIR = Path(target_dir)
    if weight_dir is not None:
        _WORKER_WEIGHT_DIR = Path(weight_dir)
    if multiplier_max is not None:
        _WORKER_MULTIPLIER_MAX = multiplier_max
    if override_path is not None:
        from tmd.areas.solver_overrides import load_overrides

        _WORKER_OVERRIDES = load_overrides(Path(override_path))
    if _WORKER_VDF is not None:
        return
    from tmd.areas.create_area_weights import (
        POPFILE_PATH,
        _load_taxcalc_data,
    )

    _WORKER_VDF = _load_taxcalc_data()
    with open(POPFILE_PATH, "r", encoding="utf-8") as pf:
        _WORKER_POP = yaml.safe_load(pf.read())


def _solve_one_area(area):
    """
    Solve area weights for one area using cached worker data.

    Returns (area, elapsed, n_targets, n_violated, status, max_viol_pct).
    """
    _init_worker()
    from tmd.areas.create_area_weights import (
        AREA_CONSTRAINT_TOL,
        AREA_MAX_ITER,
        AREA_MULTIPLIER_MAX,
        AREA_MULTIPLIER_MIN,
        AREA_SLACK_PENALTY,
        FIRST_YEAR,
        LAST_YEAR,
        _assign_slack_penalties,
        _build_constraint_matrix,
        _check_feasibility,
        _drop_impossible_targets,
        _print_multiplier_diagnostics,
        _print_slack_diagnostics,
        _print_target_diagnostics,
        _read_params,
        _solve_area_qp,
    )

    t0 = time.time()
    out = io.StringIO()

    # Build and solve
    vdf = _WORKER_VDF
    tgt_dir = _WORKER_TARGET_DIR
    wgt_dir = _WORKER_WEIGHT_DIR
    params = _read_params(area, out, target_dir=tgt_dir)

    # Apply solver overrides (centralized file takes precedence)
    if _WORKER_OVERRIDES is not None:
        from tmd.areas.solver_overrides import (
            get_area_overrides,
            get_drop_targets,
        )

        area_ovr = get_area_overrides(_WORKER_OVERRIDES, area)
        # Override params with centralized overrides
        for key in (
            "constraint_tol",
            "slack_penalty",
            "max_iter",
            "multiplier_min",
            "multiplier_max",
        ):
            if key in area_ovr:
                params[key] = area_ovr[key]
        drop_patterns = get_drop_targets(_WORKER_OVERRIDES, area)
    else:
        drop_patterns = []

    constraint_tol = params.get(
        "constraint_tol",
        params.get("target_ratio_tolerance", AREA_CONSTRAINT_TOL),
    )
    slack_penalty = params.get("slack_penalty", AREA_SLACK_PENALTY)
    max_iter = params.get("max_iter", AREA_MAX_ITER)
    multiplier_min = params.get("multiplier_min", AREA_MULTIPLIER_MIN)
    default_max = (
        _WORKER_MULTIPLIER_MAX
        if _WORKER_MULTIPLIER_MAX is not None
        else AREA_MULTIPLIER_MAX
    )
    multiplier_max = params.get("multiplier_max", default_max)

    B_csc, targets, labels, pop_share = _build_constraint_matrix(
        area, vdf, out, target_dir=tgt_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc,
        targets,
        labels,
        out,
        multiplier_max=multiplier_max,
        constraint_tol=constraint_tol,
    )

    # Apply per-area target drops from override file
    if drop_patterns:
        from tmd.areas.solver_overrides import should_drop_target

        keep = [not should_drop_target(lbl, drop_patterns) for lbl in labels]
        n_drop = sum(1 for k in keep if not k)
        if n_drop > 0:
            keep_arr = np.array(keep)
            B_dense = B_csc.toarray()
            B_csc = csc_matrix(B_dense[keep_arr, :])
            targets = targets[keep_arr]
            labels = [lbl2 for lbl2, k in zip(labels, keep) if k]
            out.write(
                f"OVERRIDE: dropped {n_drop} targets,"
                f" {len(targets)} remaining\n"
            )

    n_records = B_csc.shape[1]

    # Pre-solve feasibility check
    _check_feasibility(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        multiplier_min=multiplier_min,
        multiplier_max=multiplier_max,
        out=out,
    )

    # Assign per-constraint slack penalties
    per_constraint_penalties = _assign_slack_penalties(
        labels, default_penalty=slack_penalty
    )
    n_reduced = int((per_constraint_penalties < slack_penalty).sum())
    if n_reduced > 0:
        out.write(
            f"SLACK PENALTIES: {n_reduced}/{len(labels)}"
            f" constraints have reduced penalty\n"
        )

    # Check for per-record multiplier caps (from exhaustion limiting)
    caps_path = wgt_dir / f"{area}_record_caps.npy"
    if caps_path.exists():
        record_caps = np.load(caps_path)
        multiplier_max = np.minimum(multiplier_max, record_caps)
        n_capped = int((record_caps < AREA_MULTIPLIER_MAX).sum())
        out.write(
            f"USING PER-RECORD MULTIPLIER CAPS"
            f" ({n_capped} records capped)\n"
        )

    x_opt, s_lo, s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        slack_penalties=per_constraint_penalties,
        max_iter=max_iter,
        multiplier_min=multiplier_min,
        multiplier_max=multiplier_max,
        out=out,
    )

    # Diagnostics
    n_violated = _print_target_diagnostics(
        x_opt, B_csc, targets, labels, constraint_tol, out
    )
    _print_multiplier_diagnostics(x_opt, out)
    _print_slack_diagnostics(s_lo, s_hi, targets, labels, out)

    # Compute max violation percentage for summary
    achieved = np.asarray(B_csc @ x_opt).ravel()
    rel_errors = np.abs(achieved - targets) / np.maximum(np.abs(targets), 1.0)
    eps = 1e-9
    viol_mask = rel_errors > constraint_tol + eps
    max_viol_pct = (
        float(rel_errors[viol_mask].max() * 100) if viol_mask.any() else 0.0
    )

    # Write log
    logpath = wgt_dir / f"{area}.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with open(logpath, "w", encoding="utf-8") as f:
        f.write(out.getvalue())

    # Write weights file
    w0 = pop_share * vdf.s006.values
    wght_area = x_opt * w0

    wdict = {f"WT{FIRST_YEAR}": wght_area}
    cum_pop_growth = 1.0
    pop = _WORKER_POP
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        annual_pop_growth = pop[year] / pop[year - 1]
        cum_pop_growth *= annual_pop_growth
        wdict[f"WT{year}"] = wght_area * cum_pop_growth

    wdf = pd.DataFrame.from_dict(wdict)
    awpath = wgt_dir / f"{area}_tmd_weights.csv.gz"
    wdf.to_csv(
        awpath,
        index=False,
        float_format="%.5f",
        compression="gzip",
    )

    elapsed = time.time() - t0
    return (
        area,
        elapsed,
        len(targets),
        n_violated,
        info["status"],
        max_viol_pct,
    )


def _list_target_areas(target_dir=None):
    """Return sorted list of area codes with target files."""
    from tmd.areas.create_area_weights import STATE_TARGET_DIR

    tfolder = target_dir or STATE_TARGET_DIR
    tpaths = sorted(tfolder.glob("*_targets.csv"))
    areas = [tpath.name.split("_")[0] for tpath in tpaths]
    return areas


def _filter_areas(areas, area_filter):
    """Filter areas by type: 'states', 'cds', or 'all'."""
    if area_filter == "all":
        return areas
    if area_filter == "states":
        return [a for a in areas if len(a) == 2 and not re.match(r"[x-z]", a)]
    if area_filter == "cds":
        return [a for a in areas if len(a) > 2]
    # Treat as comma-separated list
    requested = [a.strip() for a in area_filter.split(",")]
    return [a for a in areas if a in requested]


def run_batch(
    num_workers=1,
    area_filter="all",
    force=False,
    target_dir=None,
    weight_dir=None,
    multiplier_max=None,
    override_path=None,
):
    """
    Run area weight optimization for multiple areas in parallel.

    Parameters
    ----------
    num_workers : int
        Number of parallel worker processes.
    area_filter : str
        'states', 'cds', 'all', or comma-separated area codes.
    force : bool
        If True, recompute all areas even if up-to-date.
    target_dir : Path, optional
        Directory containing target CSVs.
    weight_dir : Path, optional
        Directory for weight output.
    """
    from tmd.areas.create_area_weights import (
        STATE_TARGET_DIR,
        STATE_WEIGHT_DIR,
    )

    if target_dir is None:
        target_dir = STATE_TARGET_DIR
    if weight_dir is None:
        weight_dir = STATE_WEIGHT_DIR
    all_areas = _list_target_areas(target_dir=target_dir)
    areas = _filter_areas(all_areas, area_filter)

    if not areas:
        print("No areas to process.")
        return

    # Filter to out-of-date areas unless force=True
    if not force:
        from tmd.areas.make_all import time_of_newest_other_dependency

        newest_dep = time_of_newest_other_dependency()
        todo = []
        for area in areas:
            wpath = weight_dir / f"{area}_tmd_weights.csv.gz"
            tpath = target_dir / f"{area}_targets.csv"
            if wpath.exists():
                wtime = wpath.stat().st_mtime
                ttime = tpath.stat().st_mtime
                if wtime > max(newest_dep, ttime):
                    continue
            todo.append(area)
        areas = todo

    if not areas:
        print("All areas up-to-date. Use --force to recompute.")
        return

    # Count targets from first area's CSV
    first_tpath = target_dir / f"{areas[0]}_targets.csv"
    n_targets = (
        sum(
            1
            for line in first_tpath.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
        - 1
    )  # subtract header
    n = len(areas)
    max_total = n * n_targets
    print(
        f"Processing {n} areas"
        f" (up to {n_targets} targets each,"
        f" {max_total:,} max total)"
        f" with {num_workers} workers..."
    )
    print(
        "(Areas shown in completion order, which varies with"
        " parallel workers.)"
    )

    # Ensure weights directory exists
    weight_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    completed = 0
    violated_areas = []
    worst_viol_pct = 0.0
    max_id_width = max(len(a) for a in areas)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            str(target_dir),
            str(weight_dir),
            multiplier_max,
            str(override_path) if override_path else None,
        ),
    ) as executor:
        futures = {
            executor.submit(_solve_one_area, area): area for area in areas
        }
        for future in as_completed(futures):
            area = futures[future]
            try:
                area_code, _, _, n_viol, _, mv_pct = future.result()
                completed += 1

                if n_viol > 0:
                    violated_areas.append((area_code, n_viol))
                worst_viol_pct = max(worst_viol_pct, mv_pct)

                # Start new line every 10 areas with count prefix
                if (completed - 1) % 10 == 0:
                    sys.stdout.write(f"\n{completed:4d} ")
                sys.stdout.write(f" {area_code.ljust(max_id_width)}")
                sys.stdout.flush()

                # After every 10th area (or the last), print elapsed
                if completed % 10 == 0 or completed == n:
                    elapsed_total = time.time() - t_start
                    sys.stdout.write(f"  [{elapsed_total:.0f}s elapsed]")
                    sys.stdout.flush()

            except Exception:
                completed += 1
                sys.stdout.write(f" {area}:FAIL")
                sys.stdout.flush()
        sys.stdout.write("\n")

    total = time.time() - t_start
    m, s = divmod(total, 60)
    print(
        f"\nCompleted {completed}/{n} areas in"
        f" {total:.1f}s ({int(m)}m {s:.0f}s)"
    )
    if violated_areas:
        total_viol = sum(v for _, v in violated_areas)
        print(
            f"{len(violated_areas)} areas had violated targets"
            f" ({total_viol} targets total)."
            f" Largest violation: {worst_viol_pct:.2f}%."
        )
        cmd = "python -m tmd.areas.quality_report"
        print(f"Run: {cmd} for full details.")
    else:
        print("All targets hit within tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch area weight optimization"
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="states",
        help="'states', 'cds', 'all', or comma-separated codes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all areas even if up-to-date",
    )
    args = parser.parse_args()
    run_batch(
        num_workers=args.workers,
        area_filter=args.scope,
        force=args.force,
    )
