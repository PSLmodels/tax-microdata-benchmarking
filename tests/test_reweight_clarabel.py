"""
Unit tests for reweight_clarabel helper functions.

These tests run without make data (no data dependency).
"""

import numpy as np
from tmd.utils.reweight_clarabel import _compute_constraint_bounds


def test_positive_target_symmetric_band():
    """Positive target should give symmetric +/-tol band."""
    targets = np.array([1e6])
    cl, cu = _compute_constraint_bounds(targets, 0.005, 0.0)
    assert np.isclose(cl[0], 1e6 * 0.995)
    assert np.isclose(cu[0], 1e6 * 1.005)


def test_negative_target_tolerance_on_abs():
    """Negative target: tolerance is on |target|."""
    targets = np.array([-5e5])
    cl, cu = _compute_constraint_bounds(targets, 0.005, 0.0)
    assert cl[0] < targets[0]
    assert cu[0] > targets[0]
    band = 5e5 * 0.005
    assert np.isclose(cl[0], -5e5 - band)
    assert np.isclose(cu[0], -5e5 + band)


def test_zero_target_exact_bounds():
    """Zero target with zero floor should give exact zero bounds."""
    targets = np.array([0.0])
    cl, cu = _compute_constraint_bounds(targets, 0.005, 0.0)
    assert cl[0] == 0.0
    assert cu[0] == 0.0


def test_tolerance_override_widens_constraint():
    """Tolerance override should widen specific constraints."""
    targets = np.array([1e6, 2e6])
    labels = ["employment_income/total/AGI in 1-5k", "uc/total/AGI in 1-5k"]
    _, cu = _compute_constraint_bounds(
        targets,
        0.005,
        0.0,
        target_labels=labels,
        tolerance_overrides={"uc": 0.05},
    )
    # First target: default 0.5%
    assert np.isclose(cu[0] - targets[0], targets[0] * 0.005)
    # Second target: overridden to 5%
    assert np.isclose(cu[1] - targets[1], targets[1] * 0.05)
