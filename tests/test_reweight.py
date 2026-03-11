"""
Unit tests for tmd/utils/reweight.py helper functions.
"""

import warnings
import numpy as np
import pandas as pd
import pytest
from tmd.storage import STORAGE_FOLDER
from tmd.utils.reweight import (
    _compute_constraint_bounds,
    _drop_impossible_targets,
    build_loss_matrix,
)


def test_drop_impossible_targets_removes_all_zero_column():
    """All-zero columns are impossible targets: dropped with a UserWarning."""
    loss_matrix = pd.DataFrame(
        {
            "good_a": [1.0, 2.0, 0.0],
            "bad_zero": [0.0, 0.0, 0.0],
            "good_b": [0.0, 3.0, 1.0],
        }
    )
    targets_arr = np.array([100.0, 50.0, 200.0])
    with pytest.warns(UserWarning, match="bad_zero"):
        result_matrix, result_targets = _drop_impossible_targets(
            loss_matrix, targets_arr
        )
    assert "bad_zero" not in result_matrix.columns
    assert list(result_matrix.columns) == ["good_a", "good_b"]
    np.testing.assert_array_equal(result_targets, [100.0, 200.0])


def test_drop_impossible_targets_keeps_all_when_none_zero():
    """No columns are dropped when none are all-zero."""
    loss_matrix = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        }
    )
    targets_arr = np.array([10.0, 20.0])
    result_matrix, result_targets = _drop_impossible_targets(
        loss_matrix, targets_arr
    )
    assert list(result_matrix.columns) == ["a", "b"]
    np.testing.assert_array_equal(result_targets, [10.0, 20.0])


def test_drop_impossible_targets_column_with_single_nonzero_is_kept():
    """A column with at least one nonzero value is not impossible."""
    loss_matrix = pd.DataFrame(
        {
            "almost_zero": [0.0, 0.0, 1e-10],
        }
    )
    targets_arr = np.array([5.0])
    result_matrix, result_targets = _drop_impossible_targets(
        loss_matrix, targets_arr
    )
    assert "almost_zero" in result_matrix.columns
    assert len(result_targets) == 1


def test_no_all_zero_columns_in_real_loss_matrix(tmd_variables):
    """The real loss matrix must have no all-zero columns.

    All-zero columns mean no reweighting can hit the target.
    This is a data problem that must be fixed upstream, not
    silently filtered out at optimization time.
    """
    targets = pd.read_csv(STORAGE_FOLDER / "input" / "soi.csv")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_loss_matrix(tmd_variables, targets, 2021)
    impossible = [w for w in caught if "impossible targets" in str(w.message)]
    if impossible:
        raise AssertionError(str(impossible[0].message))


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
