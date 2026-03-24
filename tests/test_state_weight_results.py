"""
Post-solve validation of state weight files.

These tests verify that the actual state weight outputs are valid.
They are skipped if weight files have not been generated yet
(i.e., solve_weights --scope states has not been run).

Run after:
    python -m tmd.areas.prepare_targets --scope states
    python -m tmd.areas.solve_weights --scope states --workers 8
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    STATE_TARGET_DIR,
    STATE_WEIGHT_DIR,
    _build_constraint_matrix,
    _drop_impossible_targets,
    _load_taxcalc_data,
)
from tmd.areas.prepare.constants import ALL_STATES
from tmd.imputation_assumptions import TAXYEAR

# Skip entire module if weight files haven't been generated
_WEIGHT_FILES = list(STATE_WEIGHT_DIR.glob("*_tmd_weights.csv.gz"))
pytestmark = pytest.mark.skipif(
    len(_WEIGHT_FILES) < 51,
    reason="State weight files not generated yet",
)

# Also need cached data files for target accuracy checks
_CACHED = Path(__file__).parent.parent / "tmd" / "storage" / "output"
_HAS_CACHED = (_CACHED / "tmd.csv.gz").exists() and (
    _CACHED / "cached_c00100.npy"
).exists()


class TestStateWeightFiles:
    """Basic validity checks on all 51 state weight files."""

    def test_all_states_have_weight_files(self):
        """Every state has a weight file."""
        for st in ALL_STATES:
            wpath = STATE_WEIGHT_DIR / f"{st.lower()}_tmd_weights.csv.gz"
            assert wpath.exists(), f"Missing weight file for {st}"

    def test_all_states_have_log_files(self):
        """Every state has a solver log."""
        for st in ALL_STATES:
            logpath = STATE_WEIGHT_DIR / f"{st.lower()}.log"
            assert logpath.exists(), f"Missing log file for {st}"

    def test_weight_columns(self):
        """Weight files have expected year columns."""
        wpath = STATE_WEIGHT_DIR / "mn_tmd_weights.csv.gz"
        wdf = pd.read_csv(wpath)
        expected = [f"WT{yr}" for yr in range(TAXYEAR, 2035)]
        assert list(wdf.columns) == expected

    def test_weight_row_count(self):
        """Weight files have one row per TMD record."""
        wpath = STATE_WEIGHT_DIR / "mn_tmd_weights.csv.gz"
        wdf = pd.read_csv(wpath)
        # Should match TMD record count (215,494 for 2022)
        assert len(wdf) > 200_000

    @pytest.mark.parametrize(
        "state",
        [s.lower() for s in ALL_STATES],
    )
    def test_weights_nonnegative(self, state):
        """All weights are non-negative."""
        wpath = STATE_WEIGHT_DIR / f"{state}_tmd_weights.csv.gz"
        wdf = pd.read_csv(wpath)
        assert (wdf >= 0).all().all(), f"{state}: negative weights found"

    @pytest.mark.parametrize(
        "state",
        [s.lower() for s in ALL_STATES],
    )
    def test_weights_no_nan(self, state):
        """No NaN or inf values in weights."""
        wpath = STATE_WEIGHT_DIR / f"{state}_tmd_weights.csv.gz"
        wdf = pd.read_csv(wpath)
        assert not wdf.isna().any().any(), f"{state}: NaN values found"
        assert np.isfinite(wdf.values).all(), f"{state}: inf values found"

    @pytest.mark.parametrize(
        "state",
        [s.lower() for s in ALL_STATES],
    )
    def test_solver_status_solved(self, state):
        """Solver log reports Solved status."""
        logpath = STATE_WEIGHT_DIR / f"{state}.log"
        log_text = logpath.read_text()
        assert (
            "Solver status: Solved" in log_text
        ), f"{state}: solver did not report Solved"


@pytest.mark.skipif(
    not _HAS_CACHED,
    reason="Cached TMD data files not available",
)
class TestStateTargetAccuracy:
    """Verify weighted sums hit targets within tolerance."""

    @pytest.fixture(scope="class")
    def vdf(self):
        """Load TMD data once for all accuracy tests."""
        return _load_taxcalc_data()

    @pytest.mark.parametrize(
        "state",
        ["al", "ca", "mn", "ny", "tx"],
    )
    def test_targets_hit(self, vdf, state):
        """Weighted sums match targets within constraint tolerance."""
        out = io.StringIO()
        B_csc, targets, labels, pop_share = _build_constraint_matrix(
            state,
            vdf,
            out,
            target_dir=STATE_TARGET_DIR,
        )
        B_csc, targets, labels = _drop_impossible_targets(
            B_csc,
            targets,
            labels,
            out,
        )

        # Load weights and compute multipliers
        wpath = STATE_WEIGHT_DIR / f"{state}_tmd_weights.csv.gz"
        wdf = pd.read_csv(wpath)
        area_weights = wdf[f"WT{TAXYEAR}"].values
        w0 = pop_share * vdf["s006"].values
        # Avoid division by zero for zero-weight records
        safe_w0 = np.where(w0 > 0, w0, 1.0)
        x = area_weights / safe_w0
        x = np.where(w0 > 0, x, 0.0)

        # Check target accuracy
        achieved = np.asarray(B_csc @ x).ravel()
        rel_errors = np.abs(achieved - targets) / np.maximum(
            np.abs(targets), 1.0
        )
        # Allow small margin above solver tolerance for floating-point
        # differences between solver internals and weight-file roundtrip
        eps = 1e-4
        n_violated = int((rel_errors > AREA_CONSTRAINT_TOL + eps).sum())
        max_err = rel_errors.max()
        assert n_violated == 0, (
            f"{state}: {n_violated} targets violated, "
            f"max error = {max_err * 100:.3f}%"
        )
