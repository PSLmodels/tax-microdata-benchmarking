# pylint: disable=import-outside-toplevel
"""
Solve for state weights using Clarabel QP optimizer.

Reads per-state target CSV files (produced by prepare_targets.py)
and runs the Clarabel constrained QP solver to find weight
multipliers that hit area-specific targets within tolerance.

Usage:
    # All states, 8 parallel workers:
    python -m tmd.areas.solve_weights --scope states --workers 8

    # Specific states:
    python -m tmd.areas.solve_weights --scope MN,CA,TX --workers 4

    # Force recompute even if up-to-date:
    python -m tmd.areas.solve_weights --scope states --workers 8 --force

    # Single state, no parallelism:
    python -m tmd.areas.solve_weights --scope MN
"""

import argparse
import time

from tmd.areas.create_area_weights_clarabel import (
    STATE_TARGET_DIR,
    STATE_WEIGHT_DIR,
)


def solve_state_weights(
    scope="states",
    num_workers=1,
    force=True,
):
    """
    Run the Clarabel solver for the specified areas.

    Parameters
    ----------
    scope : str
        'states' or comma-separated state codes (e.g., 'MN,CA,TX').
    num_workers : int
        Number of parallel worker processes.
    force : bool
        Recompute all areas even if weight files are up-to-date.
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
    elapsed = time.time() - t0
    print(f"Total solve time: {elapsed:.1f}s")


def _parse_scope(scope):
    """Parse scope string into a list of state codes or None for all."""
    _EXCLUDE = {"US", "PR", "OA"}
    scope_lower = scope.lower().strip()
    if scope_lower in ("states", "all"):
        return None
    codes = [c.strip().upper() for c in scope.split(",") if c.strip()]
    return [c for c in codes if len(c) == 2 and c not in _EXCLUDE]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Solve for state weights using Clarabel QP optimizer",
    )
    parser.add_argument(
        "--scope",
        default="states",
        help="'states' or comma-separated state codes (e.g., 'MN,CA,TX')",
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
        help="Recompute all areas even if up-to-date (default: True)",
    )
    args = parser.parse_args()

    solve_state_weights(
        scope=args.scope,
        num_workers=args.workers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
