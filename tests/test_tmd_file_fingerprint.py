"""
Reproducibility fingerprint test for the unweighted national file
``tmd/storage/output/tmd.csv.gz``.

Paralleling ``tests/test_fingerprint.py`` (the area-weights fingerprint),
but for per-column statistics of the unweighted national file.

Run:

    pytest tests/test_tmd_file_fingerprint.py -v --update-fingerprint
    pytest tests/test_tmd_file_fingerprint.py -v

``--update-fingerprint`` writes or overwrites the reference
``tests/fingerprints/tmd_file_fingerprint.json`` and skips assertions.
Running without the flag compares the current file against the
committed reference.

Fingerprint method: for each column of ``tmd.csv.gz``, compute six
summary statistics — ``count`` (integer, exact match) and the floating-
point statistics ``sum``, ``weighted_sum`` (= ``Σ column × s006``),
``std``, ``min``, and ``max`` (compared with relative tolerance
``rtol=1e-3``). ``weighted_sum`` locks each column's relationship with
the record weight ``s006``; without it, a regeneration that preserved
each column's distribution but shuffled which records received which
weights could pass the fingerprint while every weighted 2022 total
changed.

Tax-Calculator version metadata: the fingerprint JSON records the
``taxcalc.__version__`` that was installed when it was generated. When
the test fails AND the installed version differs from the recorded one,
the failure message flags the version drift as a likely cause and
points to the regeneration command.

Replaces the previously-skipped ``test_tmd_stats`` pattern (exact-text
diff of ``df.describe()`` output, which failed on bit-different
floating-point results across machines).
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import taxcalc

from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR

TMD_FILE = STORAGE_FOLDER / "output" / "tmd.csv.gz"
FINGERPRINT_DIR = Path(__file__).parent / "fingerprints"
FINGERPRINT_PATH = FINGERPRINT_DIR / "tmd_file_fingerprint.json"

# Cross-machine floating-point noise in aggregate statistics is O(1e-6)
# relative; real data changes are typically >1%. 1e-3 sits cleanly between.
RELATIVE_TOLERANCE = 1e-3

EXACT_STATS = ("count",)
FLOAT_STATS = ("sum", "weighted_sum", "std", "min", "max")

WEIGHT_COL = "s006"


def _compute_column_stats(col_values, weights):
    """Compute the six fingerprint statistics for one column."""
    return {
        "count": int(col_values.shape[0]),
        "sum": float(col_values.sum()),
        "weighted_sum": float((col_values * weights).sum()),
        "std": float(col_values.std()),
        "min": float(col_values.min()),
        "max": float(col_values.max()),
    }


def _compute_fingerprint(df):
    """Build the fingerprint dict for tmd.csv.gz.

    The returned dict has two top-level keys:

        "metadata" — scope, taxyear, column count, record count,
                     generated date, taxcalc version
        "columns"  — dict keyed by column name

    Column keys are stored via json.dump(..., sort_keys=True) for a
    stable alphabetical on-disk order.
    """
    if WEIGHT_COL not in df.columns:
        raise KeyError(
            f"tmd.csv.gz has no {WEIGHT_COL!r} column "
            f"(columns: {list(df.columns)[:10]}...)"
        )
    weights = df[WEIGHT_COL].to_numpy()
    per_column = {}
    for col in df.columns:
        per_column[col] = _compute_column_stats(df[col].to_numpy(), weights)
    return {
        "metadata": {
            "scope": "tmd_file",
            "taxyear": TAXYEAR,
            "n_columns": len(per_column),
            "n_records": int(len(df)),
            "generated": date.today().isoformat(),
            "taxcalc_version": taxcalc.__version__,
        },
        "columns": per_column,
    }


def _save_fingerprint(fp):
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, sort_keys=True)
    return FINGERPRINT_PATH


def _load_fingerprint():
    if not FINGERPRINT_PATH.exists():
        return None
    with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _compare_fingerprints(reference, current):
    """Return a list of human-readable mismatch strings.

    Empty list means the two fingerprints match within tolerance.
    """
    ref_cols = reference.get("columns", {})
    cur_cols = current.get("columns", {})

    mismatches = []

    missing = sorted(set(ref_cols) - set(cur_cols))
    extra = sorted(set(cur_cols) - set(ref_cols))
    for c in missing:
        mismatches.append(f"{c}: missing from current run")
    for c in extra:
        mismatches.append(f"{c}: unexpected column (not in reference)")

    for col in sorted(set(ref_cols) & set(cur_cols)):
        ref = ref_cols[col]
        cur = cur_cols[col]
        for stat in EXACT_STATS:
            if ref.get(stat) != cur.get(stat):
                mismatches.append(
                    f"{col}.{stat}: {ref.get(stat)} -> {cur.get(stat)}"
                )
        for stat in FLOAT_STATS:
            ref_v = ref.get(stat)
            cur_v = cur.get(stat)
            if ref_v is None or cur_v is None:
                mismatches.append(
                    f"{col}.{stat}: {ref_v} -> {cur_v} (missing value)"
                )
                continue
            if not np.isclose(cur_v, ref_v, rtol=RELATIVE_TOLERANCE):
                denom = abs(ref_v) if ref_v != 0 else 1.0
                rel = abs(cur_v - ref_v) / denom
                mismatches.append(
                    f"{col}.{stat}: {ref_v:.6g} -> {cur_v:.6g} "
                    f"(rel diff {rel:.2e}, rtol={RELATIVE_TOLERANCE:.0e})"
                )

    return mismatches


@pytest.fixture
def update_mode(request):
    return request.config.getoption("--update-fingerprint")


@pytest.mark.skipif(
    not TMD_FILE.exists(),
    reason=(
        f"{TMD_FILE} not found — run `make data` first to build the "
        "unweighted national file"
    ),
)
# pylint: disable=redefined-outer-name
def test_tmd_file_fingerprint(update_mode):
    """Check per-column summary statistics of tmd.csv.gz against reference."""
    df = pd.read_csv(TMD_FILE)
    current = _compute_fingerprint(df)

    if update_mode:
        path = _save_fingerprint(current)
        pytest.skip(
            f"Saved to {path} — re-run without --update-fingerprint to test"
        )

    reference = _load_fingerprint()
    if reference is None:
        path = _save_fingerprint(current)
        pytest.skip(
            f"No reference found. Saved to {path} — re-run without "
            "--update-fingerprint to test"
        )

    ref_n_cols = reference.get("metadata", {}).get("n_columns")
    cur_n_cols = current["metadata"]["n_columns"]
    assert (
        cur_n_cols == ref_n_cols
    ), f"Column count: reference={ref_n_cols}, current={cur_n_cols}"

    ref_n_rec = reference.get("metadata", {}).get("n_records")
    cur_n_rec = current["metadata"]["n_records"]
    assert (
        cur_n_rec == ref_n_rec
    ), f"Record count: reference={ref_n_rec}, current={cur_n_rec}"

    mismatches = _compare_fingerprints(reference, current)
    if mismatches:
        msg_parts = [
            f"{len(mismatches)} statistic mismatch(es) in tmd.csv.gz "
            "fingerprint:",
            *mismatches,
        ]
        ref_version = reference.get("metadata", {}).get("taxcalc_version")
        cur_version = taxcalc.__version__
        if ref_version and ref_version != cur_version:
            msg_parts.append("")
            msg_parts.append(
                f"Tax-Calculator version drift: fingerprint generated "
                f"under {ref_version}, currently installed "
                f"{cur_version}. Some TMD outputs are downstream of "
                f"Tax-Calculator behavior, so a version change can "
                f"legitimately shift these statistics. If the new "
                f"version is intentional, regenerate the reference "
                f"with: pytest tests/test_tmd_file_fingerprint.py "
                f"--update-fingerprint"
            )
        pytest.fail("\n".join(msg_parts))
