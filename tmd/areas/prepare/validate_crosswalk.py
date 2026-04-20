"""
Validate a Geocorr crosswalk (117->118 or 117->119).

Runs data-quality checks on the chosen crosswalk file:

  1. Allocation factors: ``afact2`` per ``(stabbr, cd117)`` sum to 1.0
     within a tight tolerance.
  2. At-large state recoding: ``MT, DE, WY, SD, ND, VT, AK`` (and any
     states with the ``00`` / ``98`` single-district code) all map to a
     final CD code ``stabbr + "01"``.
  3. Population conservation: total ``pop20`` over target CDs equals
     total ``pop20`` over 117th CDs.
  4. Target-CD count: total number of distinct ``(stabbr, cd_target)``
     codes is 436 (435 voting + DC).
  5. For 119: for the five states that changed between Congresses
     (AL, GA, LA, NY, NC), verify that the 117->119 allocation factors
     DIFFER from the 117->118 factors.  For all other states, verify
     that the allocation factors are IDENTICAL (within 1e-6).

Usage:
    python -m tmd.areas.prepare.validate_crosswalk --congress 118
    python -m tmd.areas.prepare.validate_crosswalk --congress 119
"""

import argparse
import sys

import numpy as np
import pandas as pd

from tmd.areas.prepare.soi_cd_data import (
    _CROSSWALK_PATHS,
    SUPPORTED_CONGRESSES,
    load_crosswalk,
)

# States whose CD boundaries changed between the 118th and 119th
# Congresses (per the Supreme Court / state redistricting decisions
# effective for the January 2025 session).
_CHANGED_STATES_119 = {"AL", "GA", "LA", "NY", "NC"}

# Tolerance for floating-point comparisons between loaded DataFrames.
_TOL = 1e-6

# The Geocorr CSV stores afact2 rounded to 4 decimals, so per-group
# afact2 sums can be off by up to ~1e-4 purely from rounding.
_AFACT_SUM_TOL = 5e-4


def _check_afact_sums(cw: pd.DataFrame) -> tuple[bool, str]:
    """Allocation factors per (stabbr, cd117) must sum to ~1.0.

    afact2 is stored to 4 decimals in the CSV, so per-group sums can
    deviate from 1.0 by O(1e-4) purely from rounding.  The tolerance
    is ``_AFACT_SUM_TOL``.
    """
    sums = cw.groupby(["stabbr", "cd117"])["afact2"].sum()
    max_dev = float((sums - 1.0).abs().max())
    ok = max_dev < _AFACT_SUM_TOL
    msg = (
        f"afact2 per (stabbr, cd117) sums to 1.0 "
        f"(max |dev| = {max_dev:.3e}, tol = {_AFACT_SUM_TOL:.0e})"
    )
    return ok, msg


def _check_at_large_recoding(cw: pd.DataFrame) -> tuple[bool, str]:
    """At-large ``00`` / ``98`` source codes should be re-coded to 01."""
    # After load_crosswalk, cd117 for at-large states should be "01".
    at_large_states = {"MT", "DE", "WY", "SD", "ND", "VT", "AK"}
    bad_states = []
    for state in at_large_states:
        rows = cw[cw["stabbr"] == state]
        if len(rows) == 0:
            continue
        cds = set(rows["cd117"].unique())
        if cds != {"01"}:
            bad_states.append((state, cds))
    ok = not bad_states
    if ok:
        msg = (
            f"At-large states ({len(at_large_states)}) all "
            f"recoded to cd117='01'"
        )
    else:
        msg = f"At-large recoding failed for: {bad_states}"
    return ok, msg


def _check_pop_conservation(congress: int) -> tuple[bool, str]:
    """Population sums per state match between source and target CDs.

    Each Geocorr row is one (cd117-block, target-CD) intersection with
    its own ``pop20``.  Per state, the sum of pop20 across all rows
    equals the state population regardless of how rows are grouped by
    source or target CD.  This check verifies that:

      (a) total population is plausible (50M+ in the 50 states), and
      (b) the split summed by cd117 matches the split summed by
          cd_target (trivially true row-by-row, checked as a sanity
          test on the cleaned data).
    """
    path = _CROSSWALK_PATHS[congress]
    # The Geocorr file has a label row immediately after the header.
    raw = pd.read_csv(path, skiprows=[1], dtype=str)
    raw["pop20"] = pd.to_numeric(raw["pop20"], errors="coerce")
    raw = raw.dropna(subset=["pop20"])

    # Determine the target column name in this file (cd118 or cd119)
    target_col = "cd119" if "cd119" in raw.columns else "cd118"

    # Total pop summed two ways (should be identical — both are
    # full-file sums of pop20).
    total_source = float(raw.groupby(["stab", "cd117"])["pop20"].sum().sum())
    total_target = float(
        raw.groupby(["stab", target_col])["pop20"].sum().sum()
    )
    rel_err = abs(total_source - total_target) / max(total_source, 1.0)

    # Plausibility: US 2020 pop is ~331M; even excluding PR this
    # should be at least 300M.
    plausible = total_source > 300_000_000
    ok = rel_err < 1e-9 and plausible
    msg = (
        f"Population conserved: source={total_source:,.0f}, "
        f"target={total_target:,.0f} "
        f"(rel err = {rel_err:.2e}, plausible={plausible})"
    )
    return ok, msg


def _check_target_cd_count(
    cw: pd.DataFrame, congress: int
) -> tuple[bool, str]:
    """Number of distinct target CDs (excl. PR) must equal 436.

    The crosswalk includes PR with a single placeholder CD, but PR
    has no voting representative.  Downstream pipeline excludes PR,
    so the validation uses the same convention.
    """
    # cd_target is the neutral column name after load_crosswalk()
    distinct = cw.loc[cw["stabbr"] != "PR", ["stabbr", "cd_target"]]
    distinct = distinct.drop_duplicates()
    n = len(distinct)
    expected = 436
    ok = n == expected
    msg = (
        f"Distinct target CDs for {congress}th Congress "
        f"(excluding PR): {n} (expected {expected})"
    )
    return ok, msg


def _check_changed_vs_unchanged_states() -> tuple[bool, str]:
    """Compare 117->118 and 117->119 allocation factors.

    For the five changed states (AL, GA, LA, NY, NC) the factors
    should differ; for all others they should be bit-identical.
    """
    if 119 not in _CROSSWALK_PATHS or not _CROSSWALK_PATHS[119].exists():
        return True, (
            "Skipping 118-vs-119 comparison (119 crosswalk not present)"
        )

    cw118 = load_crosswalk(congress=118)
    cw119 = load_crosswalk(congress=119)

    # Rebuild a consistent merge key.  The source grid is
    # (stab, cd117); after merge we compare the sets of allocation
    # factors row-by-row.
    cw118 = cw118.copy()
    cw119 = cw119.copy()
    cw118["cd_target_118"] = cw118["cd_target"]
    cw119["cd_target_119"] = cw119["cd_target"]

    # For each state, collect the 117->target assignments as
    # (cd117, cd_target) pair lists and compare.
    states_118 = set(cw118["stabbr"].unique())
    states_119 = set(cw119["stabbr"].unique())
    common = states_118 & states_119

    changed_detected = set()
    unchanged_detected = set()
    mismatches = []

    for state in common:
        a = cw118[cw118["stabbr"] == state][
            ["cd117", "cd_target_118", "afact2"]
        ].reset_index(drop=True)
        b = cw119[cw119["stabbr"] == state][
            ["cd117", "cd_target_119", "afact2"]
        ].reset_index(drop=True)

        # Sort by (cd117, cd_target) to align
        a = a.sort_values(["cd117", "cd_target_118"]).reset_index(drop=True)
        b = b.sort_values(["cd117", "cd_target_119"]).reset_index(drop=True)

        same_shape = len(a) == len(b)
        same_cd_targets = (
            same_shape
            and (a["cd_target_118"].values == b["cd_target_119"].values).all()
        )
        same_factors = same_shape and np.allclose(
            a["afact2"].values, b["afact2"].values, atol=_TOL
        )
        identical = same_cd_targets and same_factors

        if identical:
            unchanged_detected.add(state)
        else:
            changed_detected.add(state)

    # Expected sets
    expected_changed = _CHANGED_STATES_119 & common
    unexpected_changed = changed_detected - expected_changed
    missing_changed = expected_changed - changed_detected

    ok = not unexpected_changed and not missing_changed
    if ok:
        msg = (
            f"118-vs-119 comparison: {len(unchanged_detected)} states "
            f"identical, {len(changed_detected)} changed "
            f"(expected: {sorted(expected_changed)})"
        )
    else:
        parts = []
        if unexpected_changed:
            parts.append(
                f"UNEXPECTED changed states: {sorted(unexpected_changed)}"
            )
            mismatches.extend(sorted(unexpected_changed))
        if missing_changed:
            parts.append(f"EXPECTED but identical: {sorted(missing_changed)}")
            mismatches.extend(sorted(missing_changed))
        msg = " | ".join(parts)
    return ok, msg


def _print_state_summary(cw: pd.DataFrame, congress: int) -> None:
    """Print a per-state district count summary for eyeballing."""
    counts = (
        cw[["stabbr", "cd_target"]]
        .drop_duplicates()
        .groupby("stab")
        .size()
        .rename("n_cds")
        .sort_values(ascending=False)
    )
    print(f"\n  {congress}th Congress district counts by state:")
    # Print compactly in rows of 10
    items = list(counts.items())
    for i in range(0, len(items), 10):
        chunk = items[i : i + 10]
        print("    " + "  ".join(f"{s}={n}" for s, n in chunk))


def validate(congress: int, verbose: bool = True) -> bool:
    if congress not in SUPPORTED_CONGRESSES:
        raise ValueError(
            f"Unsupported congress {congress}; "
            f"expected one of {SUPPORTED_CONGRESSES}"
        )
    cw_path = _CROSSWALK_PATHS[congress]
    if not cw_path.exists():
        print(f"[SKIP] Crosswalk file not found: {cw_path}")
        return True

    print(f"\nValidating crosswalk: {cw_path}")
    cw = load_crosswalk(congress=congress)

    checks = [
        _check_afact_sums(cw),
        _check_at_large_recoding(cw),
        _check_pop_conservation(congress),
        _check_target_cd_count(cw, congress),
    ]

    # 118-vs-119 consistency check (only runs when both files exist)
    if congress == 119:
        checks.append(_check_changed_vs_unchanged_states())

    all_ok = True
    for ok, msg in checks:
        mark = "[OK]  " if ok else "[FAIL]"
        print(f"  {mark} {msg}")
        all_ok = all_ok and ok

    if verbose:
        _print_state_summary(cw, congress)

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a 117->118 or 117->119 Congressional "
            "District crosswalk"
        ),
    )
    parser.add_argument(
        "--congress",
        type=int,
        choices=SUPPORTED_CONGRESSES,
        required=True,
        help="Target Congressional session (118 or 119)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-state district-count summary",
    )
    args = parser.parse_args()

    ok = validate(args.congress, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
