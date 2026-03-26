# pylint: disable=import-outside-toplevel
"""
Solve for area weights using Clarabel QP optimizer.

Reads per-area target CSV files (produced by prepare_targets.py)
and runs the Clarabel constrained QP solver to find weight
multipliers that hit area-specific targets within tolerance.

Optional exhaustion limiting (--max-exhaustion) runs a two-pass
solve: first unconstrained, then with per-record multiplier caps
to keep cross-area weight exhaustion within bounds.

Usage:
    # All states, 8 parallel workers:
    python -m tmd.areas.solve_weights --scope states --workers 8

    # All congressional districts:
    python -m tmd.areas.solve_weights --scope cds --workers 16

    # Specific areas:
    python -m tmd.areas.solve_weights --scope MN,CA,TX --workers 4
    python -m tmd.areas.solve_weights --scope MN01,CA52 --workers 4
"""

import argparse
import time

import numpy as np
import pandas as pd

from tmd.areas.create_area_weights import (
    AREA_MULTIPLIER_MAX,
    CD_MULTIPLIER_MAX,
    CD_TARGET_DIR,
    CD_WEIGHT_DIR,
    STATE_TARGET_DIR,
    STATE_WEIGHT_DIR,
)
from tmd.imputation_assumptions import TAXYEAR

_WT_COL = f"WT{TAXYEAR}"
_MAX_EXHAUST_ITERATIONS = 5


def _fmt_time(seconds):
    """Format seconds as '1034.4s (17m 14s)'."""
    m, s = divmod(seconds, 60)
    return f"{seconds:.1f}s ({int(m)}m {s:.0f}s)"


def solve_state_weights(
    scope="states",
    num_workers=1,
    force=True,
    max_exhaustion=None,
):
    """
    Run the Clarabel solver for the specified areas.

    Parameters
    ----------
    scope : str
        'states' or comma-separated state codes.
    num_workers : int
        Number of parallel worker processes.
    force : bool
        Recompute all areas even if weight files are up-to-date.
    max_exhaustion : float or None
        If set, limit per-record cross-state weight exhaustion
        to this multiple of the national weight. Runs iterative
        two-pass solve.
    """
    from tmd.areas.batch_weights import run_batch

    specific = _parse_scope(scope)
    if specific:
        area_filter = ",".join(a.lower() for a in specific)
    else:
        area_filter = "states"

    t0 = time.time()

    print("Solving state weights...")
    run_batch(
        num_workers=num_workers,
        area_filter=area_filter,
        force=force,
        target_dir=STATE_TARGET_DIR,
        weight_dir=STATE_WEIGHT_DIR,
    )

    if max_exhaustion is None:
        elapsed = time.time() - t0
        print(f"Total solve time: {_fmt_time(elapsed)}")
        return

    # --- Exhaustion-limited iterative passes ---
    for iteration in range(1, _MAX_EXHAUST_ITERATIONS + 1):
        exhaustion, state_weights = _compute_exhaustion(STATE_WEIGHT_DIR)
        n_over = int((exhaustion > max_exhaustion).sum())
        max_exh = exhaustion.max()
        print(
            f"\nExhaustion check (pass {iteration}):"
            f" max={max_exh:.2f},"
            f" {n_over} records > {max_exhaustion}x"
        )
        if n_over == 0:
            print("All records within exhaustion limit.")
            break

        # Compute and write per-record caps
        affected = _write_exhaustion_caps(
            exhaustion,
            state_weights,
            max_exhaustion,
            STATE_WEIGHT_DIR,
            STATE_TARGET_DIR,
        )
        if not affected:
            break

        # Re-solve affected states
        af = ",".join(affected)
        print(
            f"Pass {iteration + 1}: re-solving"
            f" {len(affected)} states with caps..."
        )
        run_batch(
            num_workers=num_workers,
            area_filter=af,
            force=True,
            target_dir=STATE_TARGET_DIR,
            weight_dir=STATE_WEIGHT_DIR,
        )

        # Clean up cap files for this iteration
        _cleanup_caps(STATE_WEIGHT_DIR, affected)
    else:
        exhaustion, _ = _compute_exhaustion(STATE_WEIGHT_DIR)
        n_still = int((exhaustion > max_exhaustion).sum())
        if n_still > 0:
            print(
                f"Warning: {n_still} records still exceed"
                f" {max_exhaustion}x after"
                f" {_MAX_EXHAUST_ITERATIONS} iterations"
                f" (max={exhaustion.max():.2f}x)"
            )

    elapsed = time.time() - t0
    print(f"Total solve time: {_fmt_time(elapsed)}")


def solve_cd_weights(
    scope="cds",
    num_workers=1,
    force=True,
):
    """
    Run the Clarabel solver for congressional districts.

    Parameters
    ----------
    scope : str
        'cds' or comma-separated CD codes (e.g., 'MN01,CA52').
    num_workers : int
        Number of parallel worker processes.
    force : bool
        Recompute all areas even if weight files are up-to-date.
    """
    from tmd.areas.batch_weights import run_batch

    specific = _parse_cd_scope(scope)
    if specific:
        area_filter = ",".join(a.lower() for a in specific)
    else:
        area_filter = "cds"

    t0 = time.time()
    print("Solving CD weights...")
    run_batch(
        num_workers=num_workers,
        area_filter=area_filter,
        force=force,
        target_dir=CD_TARGET_DIR,
        weight_dir=CD_WEIGHT_DIR,
        multiplier_max=CD_MULTIPLIER_MAX,
    )

    elapsed = time.time() - t0
    print(f"Total solve time: {_fmt_time(elapsed)}")


def _compute_exhaustion(weight_dir):
    """
    Compute per-record exhaustion across all state weight files.

    Returns (exhaustion_array, state_weights_dict).
    """
    from tmd.storage import STORAGE_FOLDER

    s006 = pd.read_csv(
        STORAGE_FOLDER / "output" / "tmd.csv.gz",
        usecols=["s006"],
    )["s006"].values
    n = len(s006)

    weight_sum = np.zeros(n)
    state_weights = {}
    for wpath in sorted(weight_dir.glob("*_tmd_weights.csv.gz")):
        area = wpath.name.split("_")[0]
        w = pd.read_csv(wpath, usecols=[_WT_COL])[_WT_COL].values
        weight_sum += w
        state_weights[area] = w

    exhaustion = weight_sum / s006
    return exhaustion, state_weights


def _write_exhaustion_caps(
    exhaustion,
    state_weights,
    max_exhaustion,
    weight_dir,
    target_dir,
):
    """
    Compute per-record multiplier caps and write cap files.

    For over-exhausted records, scale each state's multiplier
    cap proportionally to current usage so total exhaustion
    equals max_exhaustion.

    Returns list of affected area codes.
    """
    from tmd.storage import STORAGE_FOLDER

    s006 = pd.read_csv(
        STORAGE_FOLDER / "output" / "tmd.csv.gz",
        usecols=["s006"],
    )["s006"].values
    nat_pop = pd.read_csv(
        STORAGE_FOLDER / "output" / "tmd.csv.gz",
        usecols=["s006", "XTOT"],
    )
    nat_pop = (nat_pop["s006"] * nat_pop["XTOT"]).sum()

    over_mask = exhaustion > max_exhaustion
    if not over_mask.any():
        return []

    # Scale factor per record: how much to shrink
    scale = np.ones_like(exhaustion)
    scale[over_mask] = max_exhaustion / exhaustion[over_mask]

    affected_areas = set()
    for area, weights in state_weights.items():
        # Get pop_share for this area
        tpath = target_dir / f"{area}_targets.csv"
        if not tpath.exists():
            continue
        targets_df = pd.read_csv(tpath, comment="#", nrows=1)
        xtot_target = targets_df.iloc[0]["target"]
        pop_share = xtot_target / nat_pop

        # Current multiplier for each record
        w0 = pop_share * s006
        with np.errstate(divide="ignore", invalid="ignore"):
            current_x = np.where(w0 > 0, weights / w0, 0.0)

        # New cap = current_x * scale (only tighten, never loosen)
        new_caps = current_x * scale
        # Don't exceed global max
        new_caps = np.minimum(new_caps, AREA_MULTIPLIER_MAX)

        # Only write if some records are actually capped
        n_capped = int((new_caps < AREA_MULTIPLIER_MAX - 1e-6).sum())
        if n_capped > 0:
            caps_path = weight_dir / f"{area}_record_caps.npy"
            np.save(caps_path, new_caps)
            affected_areas.add(area)

    return sorted(affected_areas)


def _cleanup_caps(weight_dir, areas):
    """Remove cap files after a solve pass."""
    for area in areas:
        caps_path = weight_dir / f"{area}_record_caps.npy"
        caps_path.unlink(missing_ok=True)


def _parse_scope(scope):
    """Parse scope string into list of state codes or None."""
    _EXCLUDE = {"US", "PR", "OA"}
    scope_lower = scope.lower().strip()
    if scope_lower in ("states", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    return [c for c in codes if len(c) == 2 and c not in _EXCLUDE]


def _parse_cd_scope(scope):
    """Parse scope string into list of CD codes or None."""
    scope_lower = scope.lower().strip()
    if scope_lower in ("cds", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    return [c for c in codes if len(c) > 2]


def _is_cd_scope(scope):
    """Return True if the scope refers to CDs rather than states."""
    scope_lower = scope.lower().strip()
    if scope_lower == "cds":
        return True
    if scope_lower == "states":
        return False
    # Comma-separated: check first code length
    first = scope.split(",")[0].strip()
    return len(first) > 2


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=("Solve for area weights using Clarabel QP optimizer"),
    )
    parser.add_argument(
        "--scope",
        default="states",
        help=(
            "'states', 'cds', or comma-separated area codes"
            " (e.g., 'MN,CA,TX' or 'MN01,CA52')"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel solver workers (default: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=True,
        help="Recompute all areas even if up-to-date",
    )
    parser.add_argument(
        "--max-exhaustion",
        type=float,
        default=None,
        help=(
            "Max per-record cross-area weight exhaustion"
            " (e.g., 5.0). Runs iterative solve to enforce."
        ),
    )
    args = parser.parse_args()

    if _is_cd_scope(args.scope):
        solve_cd_weights(
            scope=args.scope,
            num_workers=args.workers,
            force=args.force,
        )
    else:
        solve_state_weights(
            scope=args.scope,
            num_workers=args.workers,
            force=args.force,
            max_exhaustion=args.max_exhaustion,
        )


if __name__ == "__main__":
    main()
