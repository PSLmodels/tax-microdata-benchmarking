"""
On-demand fingerprint test for area weight results.

NOT run by `make test` (excluded in Makefile).
Run manually after a full solve:

    pytest tests/test_fingerprint.py -v
    pytest tests/test_fingerprint.py -v --update-fingerprint

The first run saves a reference fingerprint.
Subsequent runs compare against it.

Fingerprint method: for each area, round weights to nearest integer,
sum them, and hash the per-area sums. This is simple, fast, and
catches any meaningful change in results.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tmd.areas import AREAS_FOLDER

FINGERPRINT_DIR = AREAS_FOLDER / "fingerprints"
STATE_WEIGHT_DIR = AREAS_FOLDER / "weights" / "states"

ALL_STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def _compute_fingerprint(areas, weight_dir):
    """Compute fingerprint from weight files.

    For each area, reads the first WT column, rounds weights to
    nearest integer, and records the sum. The collection of integer
    sums is hashed for a single comparison value.
    """
    per_area = {}
    for area in areas:
        code = area.lower()
        wpath = weight_dir / f"{code}_tmd_weights.csv.gz"
        if not wpath.exists():
            continue
        wdf = pd.read_csv(wpath)
        wt_cols = [c for c in wdf.columns if c.startswith("WT")]
        wt = wdf[wt_cols[0]].values
        int_sum = int(np.round(wt).sum())
        per_area[area] = int_sum

    # Hash of all per-area integer sums
    hash_str = "|".join(f"{a}:{per_area[a]}" for a in sorted(per_area.keys()))
    hash_val = hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    return {
        "n_areas": len(per_area),
        "weight_hash": hash_val,
        "per_area_int_sums": per_area,
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


@pytest.fixture
def update_mode(request):
    return request.config.getoption("--update-fingerprint")


def _has_weight_files(weight_dir, areas):
    for a in areas:
        wpath = weight_dir / f"{a.lower()}_tmd_weights.csv.gz"
        if wpath.exists():
            return True
    return False


@pytest.mark.skipif(
    not _has_weight_files(STATE_WEIGHT_DIR, ALL_STATES),
    reason="No state weight files — run solve_weights first",
)
class TestStateFingerprint:
    """Fingerprint tests for state weights."""

    # pylint: disable=redefined-outer-name
    def test_state_weights_match_reference(self, update_mode):
        """Compare weight integer sums against saved reference."""
        current = _compute_fingerprint(ALL_STATES, STATE_WEIGHT_DIR)

        if update_mode:
            path = _save_fingerprint("states", current)
            pytest.skip(f"Saved to {path} — re-run to test")

        reference = _load_fingerprint("states")
        if reference is None:
            path = _save_fingerprint("states", current)
            pytest.skip(f"No reference found. Saved to {path} — re-run")

        ref_n = reference["n_areas"]
        cur_n = current["n_areas"]
        assert cur_n == ref_n, f"Area count: {ref_n} -> {cur_n}"

        assert (
            current["weight_hash"] == reference["weight_hash"]
        ), "Weight hash mismatch — results changed"

    def test_per_area_sums_match(self, update_mode):
        """Identify which areas changed."""
        if update_mode:
            pytest.skip("Update mode")

        reference = _load_fingerprint("states")
        if reference is None:
            pytest.skip("No reference fingerprint")

        current = _compute_fingerprint(ALL_STATES, STATE_WEIGHT_DIR)
        ref_sums = reference.get("per_area_int_sums", {})
        cur_sums = current.get("per_area_int_sums", {})

        mismatches = []
        for area in sorted(ref_sums.keys()):
            if area not in cur_sums:
                mismatches.append(f"{area}: missing")
                continue
            if ref_sums[area] != cur_sums[area]:
                mismatches.append(
                    f"{area}: {ref_sums[area]}" f" -> {cur_sums[area]}"
                )

        assert (
            not mismatches
        ), f"{len(mismatches)} areas changed:\n" + "\n".join(mismatches)
