"""
Test of tmd/storage/output/tmd.csv.gz weights.

Checks that the full weight distribution fingerprint matches
the expected Clarabel QP output.  Clarabel is cross-machine
deterministic, so np.allclose defaults (rtol=1e-5, atol=1e-8)
should hold.
"""

import numpy as np
import pytest


@pytest.mark.weight_distribution
def test_weights(tmd_variables):
    """Check weight distribution matches Clarabel QP fingerprint."""
    wght = tmd_variables["s006"].to_numpy()
    actual = {
        "n": len(wght),
        "total": wght.sum(),
        "mean": wght.mean(),
        "sdev": wght.std(),
        "min": wght.min(),
        "p25": np.percentile(wght, 25),
        "p50": np.median(wght),
        "p75": np.percentile(wght, 75),
        "max": wght.max(),
        "sum_sq": np.sum(wght**2),
    }
    expect = {
        "n": 215291,
        "total": 187475138.4,
        "mean": 870.80,
        "sdev": 1209.99,
        "min": 0.10769,
        "p25": 21.542,
        "p50": 392.158,
        "p75": 1328.248,
        "max": 16527.65,
        "sum_sq": 478456942738.8,
    }
    # n must be exact
    assert (
        actual["n"] == expect["n"]
    ), f"n mismatch: actual={actual["n"]}, expected={expect["n"]}"
    # all float stats checked with np.allclose defaults
    float_stats = [
        "total",
        "mean",
        "sdev",
        "min",
        "p25",
        "p50",
        "p75",
        "max",
        "sum_sq",
    ]
    diffs = []
    for stat in float_stats:
        if not np.allclose([actual[stat]], [expect[stat]]):
            diffs.append(
                f"  {stat}: actual={actual[stat]}, " f"expected={expect[stat]}"
            )
    if diffs:
        msg = "Weight fingerprint mismatch:\n" + "\n".join(diffs)
        raise ValueError(msg)
