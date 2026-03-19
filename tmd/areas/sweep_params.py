#!/usr/bin/env python3
"""
Parameter sweep for state weight solver.

Tests combinations of multiplier_max and weight_penalty,
solving all 51 states for each combo and reporting:
  - Target violations
  - Weight exhaustion
  - Solve time

Usage:
    python -m tmd.areas.sweep_params
"""

import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    AREA_MAX_ITER,
    AREA_MULTIPLIER_MIN,
    AREA_SLACK_PENALTY,
    POPFILE_PATH,
    STATE_TARGET_DIR,
    _build_constraint_matrix,
    _drop_impossible_targets,
    _load_taxcalc_data,
    _solve_area_qp,
)
from tmd.areas.prepare.constants import ALL_STATES
from tmd.imputation_assumptions import TAXYEAR

_WT_COL = f"WT{TAXYEAR}"

# --- Parameter grid ---
GRID_MULTIPLIER_MAX = [10, 15, 25, 100]
GRID_WEIGHT_PENALTY = [1.0, 10.0, 100.0]

NUM_WORKERS = 8

# Module-level cache
_VDF = None
_POP = None


def _init():
    """Load data once."""
    global _VDF, _POP  # pylint: disable=global-statement
    if _VDF is not None:
        return
    _VDF = _load_taxcalc_data()
    with open(POPFILE_PATH, "r", encoding="utf-8") as pf:
        _POP = yaml.safe_load(pf.read())


def _solve_one(args):
    """Solve one state with given params. Returns stats dict."""
    area, mult_max, wt_pen = args
    _init()
    vdf = _VDF
    out = io.StringIO()

    B_csc, targets, labels, pop_share = _build_constraint_matrix(
        area, vdf, out, target_dir=STATE_TARGET_DIR
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc, targets, labels, out
    )

    n_records = B_csc.shape[1]
    t0 = time.time()
    x_opt, _s_lo, _s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        constraint_tol=AREA_CONSTRAINT_TOL,
        slack_penalty=AREA_SLACK_PENALTY,
        max_iter=AREA_MAX_ITER,
        multiplier_min=AREA_MULTIPLIER_MIN,
        multiplier_max=mult_max,
        weight_penalty=wt_pen,
        out=out,
    )
    elapsed = time.time() - t0

    # Compute stats
    achieved = np.asarray(B_csc @ x_opt).ravel()
    rel_errors = np.abs(achieved - targets) / np.maximum(np.abs(targets), 1.0)
    eps = 1e-9
    n_violated = int((rel_errors > AREA_CONSTRAINT_TOL + eps).sum())
    max_viol = float(rel_errors.max() * 100)

    # Weight stats
    w0 = pop_share * vdf.s006.values
    final_weights = x_opt * w0

    return {
        "area": area,
        "mult_max": mult_max,
        "wt_pen": wt_pen,
        "n_targets": len(targets),
        "n_violated": n_violated,
        "max_viol_pct": max_viol,
        "x_max": float(x_opt.max()),
        "x_rmse": float(np.sqrt(np.mean((x_opt - 1.0) ** 2))),
        "pct_zero": float((x_opt < 1e-6).mean() * 100),
        "status": info["status"],
        "elapsed": elapsed,
        "pop_share": pop_share,
        "final_weights": final_weights,
    }


def run_sweep():
    """Run parameter sweep and print results."""
    print("Loading TMD data...")
    _init()

    areas = [s.lower() for s in ALL_STATES]
    combos = [
        (mm, wp) for mm in GRID_MULTIPLIER_MAX for wp in GRID_WEIGHT_PENALTY
    ]

    s006 = _VDF.s006.values
    n_records = len(s006)

    print(
        f"Sweep: {len(combos)} parameter combos"
        f" x {len(areas)} states"
        f" = {len(combos) * len(areas)} solves"
    )
    print(
        f"Grid: mult_max={GRID_MULTIPLIER_MAX},"
        f" weight_penalty={GRID_WEIGHT_PENALTY}"
    )
    print(f"Workers: {NUM_WORKERS}")
    print()

    results = []

    for combo_idx, (mm, wp) in enumerate(combos):
        label = f"mult_max={mm:>3}, wt_pen={wp:>5.1f}"
        sys.stdout.write(f"[{combo_idx + 1}/{len(combos)}] {label}...")
        sys.stdout.flush()

        t0 = time.time()
        tasks = [(area, mm, wp) for area in areas]

        combo_results = []
        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_init,
        ) as executor:
            futures = {
                executor.submit(_solve_one, task): task for task in tasks
            }
            for future in as_completed(futures):
                combo_results.append(future.result())

        elapsed = time.time() - t0

        # Aggregate stats
        total_violated = sum(r["n_violated"] for r in combo_results)
        max_viol = max(r["max_viol_pct"] for r in combo_results)
        avg_rmse = np.mean([r["x_rmse"] for r in combo_results])
        avg_zero = np.mean([r["pct_zero"] for r in combo_results])

        # Compute exhaustion
        weight_sum = np.zeros(n_records)
        for r in combo_results:
            weight_sum += r["final_weights"]
        exhaustion = weight_sum / s006
        max_exh = float(exhaustion.max())
        p99_exh = float(np.percentile(exhaustion, 99))
        n_over5 = int((exhaustion > 5).sum())
        n_over10 = int((exhaustion > 10).sum())

        summary = {
            "mult_max": mm,
            "wt_pen": wp,
            "total_violated": total_violated,
            "max_viol_pct": max_viol,
            "avg_rmse": avg_rmse,
            "avg_pct_zero": avg_zero,
            "max_exhaustion": max_exh,
            "p99_exhaustion": p99_exh,
            "n_over5x": n_over5,
            "n_over10x": n_over10,
            "elapsed": elapsed,
        }
        results.append(summary)

        sys.stdout.write(
            f" {elapsed:.0f}s |"
            f" viol={total_violated:>4}"
            f" maxV={max_viol:.2f}%"
            f" wRMSE={avg_rmse:.3f}"
            f" %zero={avg_zero:.1f}%"
            f" maxExh={max_exh:.1f}"
            f" >5x={n_over5}"
            f" >10x={n_over10}\n"
        )

    # Summary table
    print("\n" + "=" * 100)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 100)
    cols = [
        "mMax",
        "wPen",
        "Viol",
        "MaxV%",
        "wRMSE",
        "%zero",
        "MaxExh",
        "p99Exh",
        "O5x",
        "O10x",
        "Time",
    ]
    widths = [5, 6, 5, 6, 6, 6, 7, 7, 5, 5, 6]
    hdr = " ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        mm = r["mult_max"]
        wp = r["wt_pen"]
        tv = r["total_violated"]
        mv = r["max_viol_pct"]
        rm = r["avg_rmse"]
        pz = r["avg_pct_zero"]
        me = r["max_exhaustion"]
        pe = r["p99_exhaustion"]
        o5 = r["n_over5x"]
        o10 = r["n_over10x"]
        et = r["elapsed"]
        print(
            f"{mm:>5} {wp:>6.1f}"
            f" {tv:>5} {mv:>6.2f}"
            f" {rm:>6.3f} {pz:>6.1f}"
            f" {me:>7.1f} {pe:>7.3f}"
            f" {o5:>5} {o10:>5}"
            f" {et:>5.0f}s"
        )
    print()
    print("Baseline: mult_max=100, wt_pen=1.0")


if __name__ == "__main__":
    run_sweep()
