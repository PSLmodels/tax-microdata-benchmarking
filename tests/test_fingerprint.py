"""
On-demand reproducibility fingerprint test for area weight results.

NOT run by `make test` (excluded in Makefile). A full
``solve_weights`` run (~55 min for CDs at 16 workers) must have
populated weight files before this test is useful.

Run manually after a full solve:

    pytest tests/test_fingerprint.py -v --update-fingerprint
    pytest tests/test_fingerprint.py -v
    pytest tests/test_fingerprint.py -v -k states
    pytest tests/test_fingerprint.py -v -k cd_118
    pytest tests/test_fingerprint.py -v -k cd_119

``--update-fingerprint`` writes or overwrites the reference
``<scope>_fingerprint.json`` file for every scope whose weight
directory contains weight files, then skips assertions. Running
without the flag compares current weight files against the
committed reference JSONs. Any scope whose weight directory is
empty is skipped with an informative reason, so a fresh clone
without any solve produces only skips, not failures.

Fingerprint method: for each area, compute five summary statistics
of the ``WT{TAXYEAR}`` column (record count, positive-weight count,
weight sum, weight standard deviation, and maximum weight). Exact
integer match is required for the two counts; relative tolerance
``rtol=1e-3`` is used for the three float statistics. This design
replaces the previous integer-rounded sum + SHA-256 hash approach
(issue #477), which was insensitive to small areas, distribution-
blind, and produced rounding-boundary false positives across
machines.
"""

# pylint: disable=too-few-public-methods
# Each test class holds one test method by design (one test per scope);
# the pytest class wrapper exists only for the @pytest.mark.skipif guard.

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from tmd.areas import AREAS_FOLDER
from tmd.areas.prepare.constants import ALL_STATES
from tmd.imputation_assumptions import TAXYEAR

FINGERPRINT_DIR = AREAS_FOLDER / "fingerprints"
STATE_WEIGHT_DIR = AREAS_FOLDER / "weights" / "states"
CD_118_WEIGHT_DIR = AREAS_FOLDER / "weights" / "cds_118"
CD_119_WEIGHT_DIR = AREAS_FOLDER / "weights" / "cds_119"

# Records with weight above this threshold are counted as "positive".
# Weights below this are treated as effectively zero (solver noise or
# records outside the area). Picking the same cutoff across machines
# avoids sparsity-count drift from sub-noise weight values.
POSITIVE_WEIGHT_THRESHOLD = 0.01

# Float-statistic tolerance. Cross-machine noise in aggregate
# statistics is O(1e-6) to O(1e-4) relative; real code/data changes
# are typically > 1%. 1e-3 sits cleanly between noise and signal.
RELATIVE_TOLERANCE = 1e-3

# Which statistics are compared exactly vs. with relative tolerance.
EXACT_STATS = ("n_records", "n_positive")
FLOAT_STATS = ("weight_sum", "weight_std", "max_weight")


def _discover_cd_areas(weight_dir):
    """Discover CD area codes from existing weight files.

    Returns an alphabetical list of uppercase codes (e.g. "CA05").
    Empty list if the directory does not exist or contains no
    weight files.
    """
    if not weight_dir.exists():
        return []
    return sorted(
        p.name.replace("_tmd_weights.csv.gz", "").upper()
        for p in weight_dir.glob("*_tmd_weights.csv.gz")
    )


def _compute_area_stats(weight_path, taxyear):
    """Compute the 5 fingerprint statistics for one area's weight file.

    Reads the ``WT{taxyear}`` column and returns a dict with:
        n_records    (int)   total rows in the weight file
        n_positive   (int)   rows with weight > POSITIVE_WEIGHT_THRESHOLD
        weight_sum   (float) sum of all weights
        weight_std   (float) population standard deviation of weights
        max_weight   (float) maximum weight
    """
    wdf = pd.read_csv(weight_path)
    col = f"WT{taxyear}"
    if col not in wdf.columns:
        raise KeyError(
            f"Weight file {weight_path} has no column {col!r} "
            f"(columns: {list(wdf.columns)})"
        )
    wt = wdf[col].values
    return {
        "n_records": int(len(wt)),
        "n_positive": int((wt > POSITIVE_WEIGHT_THRESHOLD).sum()),
        "weight_sum": float(wt.sum()),
        "weight_std": float(wt.std()),
        "max_weight": float(wt.max()),
    }


def _compute_fingerprint(areas, weight_dir, scope, taxyear):
    """Build the fingerprint dict for one scope from its weight files.

    Skips any area whose weight file does not exist (so a partially
    completed solve produces a fingerprint over just the areas that
    were solved). The returned dict has two top-level keys:

        "metadata" — scope, taxyear, area count, generated date
        "areas"    — dict keyed by area code (e.g. "AK", "CA05")

    Area keys are stored in the JSON in insertion order; json.dump
    with sort_keys=True gives a stable alphabetical on-disk order.
    """
    per_area = {}
    for area in areas:
        wpath = weight_dir / f"{area.lower()}_tmd_weights.csv.gz"
        if not wpath.exists():
            continue
        per_area[area] = _compute_area_stats(wpath, taxyear)

    return {
        "metadata": {
            "scope": scope,
            "taxyear": taxyear,
            "n_areas": len(per_area),
            "generated": date.today().isoformat(),
        },
        "areas": per_area,
    }


def _fingerprint_path(scope):
    return FINGERPRINT_DIR / f"{scope}_fingerprint.json"


def _save_fingerprint(scope, fp):
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _fingerprint_path(scope)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, sort_keys=True)
    return path


def _load_fingerprint(scope):
    path = _fingerprint_path(scope)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compare_fingerprints(reference, current):
    """Return a list of human-readable mismatch strings.

    Empty list means reference and current match within tolerance.
    Each element in the returned list names one area and one
    statistic that diverged, with both values and the relative
    difference for float statistics.
    """
    ref_areas = reference.get("areas", {})
    cur_areas = current.get("areas", {})

    mismatches = []

    missing = sorted(set(ref_areas) - set(cur_areas))
    extra = sorted(set(cur_areas) - set(ref_areas))
    for a in missing:
        mismatches.append(f"{a}: missing from current run")
    for a in extra:
        mismatches.append(f"{a}: unexpected (not in reference)")

    for area in sorted(set(ref_areas) & set(cur_areas)):
        ref = ref_areas[area]
        cur = cur_areas[area]
        for stat in EXACT_STATS:
            if ref.get(stat) != cur.get(stat):
                mismatches.append(
                    f"{area}.{stat}: {ref.get(stat)} -> {cur.get(stat)}"
                )
        for stat in FLOAT_STATS:
            ref_v = ref.get(stat)
            cur_v = cur.get(stat)
            if ref_v is None or cur_v is None:
                mismatches.append(
                    f"{area}.{stat}: {ref_v} -> {cur_v} (missing value)"
                )
                continue
            if not np.isclose(cur_v, ref_v, rtol=RELATIVE_TOLERANCE):
                denom = abs(ref_v) if ref_v != 0 else 1.0
                rel = abs(cur_v - ref_v) / denom
                mismatches.append(
                    f"{area}.{stat}: {ref_v:.6g} -> {cur_v:.6g} "
                    f"(rel diff {rel:.2e}, rtol={RELATIVE_TOLERANCE:.0e})"
                )

    return mismatches


@pytest.fixture
def update_mode(request):
    return request.config.getoption("--update-fingerprint")


def _has_weight_files(weight_dir, areas):
    if not weight_dir.exists() or not areas:
        return False
    for a in areas:
        if (weight_dir / f"{a.lower()}_tmd_weights.csv.gz").exists():
            return True
    return False


def _run_fingerprint_test(scope, areas, weight_dir, update):
    """Shared save-or-compare logic for all three scopes."""
    current = _compute_fingerprint(areas, weight_dir, scope, TAXYEAR)

    if update:
        path = _save_fingerprint(scope, current)
        pytest.skip(f"Saved to {path} — re-run without --update to test")

    reference = _load_fingerprint(scope)
    if reference is None:
        path = _save_fingerprint(scope, current)
        pytest.skip(f"No reference found. Saved to {path} — re-run")

    ref_n = reference.get("metadata", {}).get("n_areas")
    cur_n = current["metadata"]["n_areas"]
    assert cur_n == ref_n, f"Area count: reference={ref_n}, current={cur_n}"

    mismatches = _compare_fingerprints(reference, current)
    assert not mismatches, (
        f"{len(mismatches)} statistic mismatch(es) in {scope} fingerprint:\n"
        + "\n".join(mismatches)
    )


# --- State tests ---


@pytest.mark.skipif(
    not _has_weight_files(STATE_WEIGHT_DIR, ALL_STATES),
    reason=(
        "No state weight files — run "
        "`python -m tmd.areas.solve_weights --scope states` first"
    ),
)
class TestStatesFingerprint:
    """Reproducibility fingerprint for state area weights."""

    # pylint: disable=redefined-outer-name
    def test_states_weights_match_reference(self, update_mode):
        _run_fingerprint_test(
            "states", ALL_STATES, STATE_WEIGHT_DIR, update_mode
        )


# --- CD 118 tests ---

_CD_118_AREAS = _discover_cd_areas(CD_118_WEIGHT_DIR)


@pytest.mark.skipif(
    not _has_weight_files(CD_118_WEIGHT_DIR, _CD_118_AREAS),
    reason=(
        "No cds_118 weight files — run "
        "`python -m tmd.areas.solve_weights --scope cds --congress 118` first"
    ),
)
class TestCd118Fingerprint:
    """Reproducibility fingerprint for 118th Congress CD weights."""

    # pylint: disable=redefined-outer-name
    def test_cd_118_weights_match_reference(self, update_mode):
        _run_fingerprint_test(
            "cds_118", _CD_118_AREAS, CD_118_WEIGHT_DIR, update_mode
        )


# --- CD 119 tests ---

_CD_119_AREAS = _discover_cd_areas(CD_119_WEIGHT_DIR)


@pytest.mark.skipif(
    not _has_weight_files(CD_119_WEIGHT_DIR, _CD_119_AREAS),
    reason=(
        "No cds_119 weight files — run "
        "`python -m tmd.areas.solve_weights --scope cds --congress 119` first"
    ),
)
class TestCd119Fingerprint:
    """Reproducibility fingerprint for 119th Congress CD weights."""

    # pylint: disable=redefined-outer-name
    def test_cd_119_weights_match_reference(self, update_mode):
        _run_fingerprint_test(
            "cds_119", _CD_119_AREAS, CD_119_WEIGHT_DIR, update_mode
        )
