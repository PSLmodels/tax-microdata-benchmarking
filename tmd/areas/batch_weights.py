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

from tmd.areas.create_area_weights import valid_area

# Module-level cache for TMD data (one per worker process)
_WORKER_VDF = None
_WORKER_POP = None
_WORKER_TARGET_DIR = None
_WORKER_WEIGHT_DIR = None


def _init_worker(target_dir=None, weight_dir=None):
    """Load TMD data once per worker process."""
    global _WORKER_VDF, _WORKER_POP
    global _WORKER_TARGET_DIR, _WORKER_WEIGHT_DIR
    if target_dir is not None:
        _WORKER_TARGET_DIR = Path(target_dir)
    if weight_dir is not None:
        _WORKER_WEIGHT_DIR = Path(weight_dir)
    if _WORKER_VDF is not None:
        return
    from tmd.areas.create_area_weights_clarabel import (
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
    from tmd.areas.create_area_weights_clarabel import (
        AREA_CONSTRAINT_TOL,
        AREA_MAX_ITER,
        AREA_MULTIPLIER_MAX,
        AREA_MULTIPLIER_MIN,
        AREA_SLACK_PENALTY,
        FIRST_YEAR,
        LAST_YEAR,
        _build_constraint_matrix,
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
    constraint_tol = params.get(
        "constraint_tol",
        params.get("target_ratio_tolerance", AREA_CONSTRAINT_TOL),
    )
    slack_penalty = params.get("slack_penalty", AREA_SLACK_PENALTY)
    max_iter = params.get("max_iter", AREA_MAX_ITER)
    multiplier_min = params.get("multiplier_min", AREA_MULTIPLIER_MIN)
    multiplier_max = params.get("multiplier_max", AREA_MULTIPLIER_MAX)

    B_csc, targets, labels, pop_share = _build_constraint_matrix(
        area, vdf, out, target_dir=tgt_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc, targets, labels, out
    )

    n_records = B_csc.shape[1]
    x_opt, s_lo, s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=constraint_tol,
        slack_penalty=slack_penalty,
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
    from tmd.areas.create_area_weights_clarabel import STATE_TARGET_DIR

    tfolder = target_dir or STATE_TARGET_DIR
    tpaths = sorted(tfolder.glob("*_targets.csv"))
    areas = []
    for tpath in tpaths:
        area = tpath.name.split("_")[0]
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        ok = valid_area(area)
        sys.stderr = old_stderr
        if ok:
            areas.append(area)
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
    from tmd.areas.create_area_weights_clarabel import (
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

    n = len(areas)
    print(f"Processing {n} areas with {num_workers} workers...")
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
        initargs=(str(target_dir), str(weight_dir)),
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
    print(f"\nCompleted {completed}/{n} areas in {total:.1f}s")
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
