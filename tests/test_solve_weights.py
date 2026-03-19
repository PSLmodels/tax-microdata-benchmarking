"""
Tests for the Clarabel-based state weight solver pipeline.

Tests:
  - Clarabel solver on faux xx area
  - Quality report log parser
  - CLI scope parsing
  - Batch area filtering
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmd.areas import AREAS_FOLDER
from tmd.areas.batch_weights import _filter_areas
from tmd.areas.create_area_weights import (
    AREA_CONSTRAINT_TOL,
    _build_constraint_matrix,
    _drop_impossible_targets,
    _load_taxcalc_data,
    _solve_area_qp,
    create_area_weights_file,
)
from tmd.areas.quality_report import (
    _humanize_desc,
    parse_log,
)
from tmd.areas.solve_weights import (
    _parse_scope,
)
from tmd.imputation_assumptions import TAXYEAR

# --- Solver test on faux xx area ---


def test_clarabel_solver_xx():
    """
    Solve faux xx area with Clarabel and verify targets are hit.
    """
    # xx targets are in the flat targets/ directory
    target_dir = AREAS_FOLDER / "targets"

    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        rc = create_area_weights_file(
            "xx",
            write_log=True,
            write_file=True,
            target_dir=target_dir,
            weight_dir=weight_dir,
        )
        assert rc == 0, "create_area_weights_file returned non-zero"

        # Verify weights file was created
        wpath = weight_dir / "xx_tmd_weights.csv.gz"
        assert wpath.exists(), "Weight file not created"

        # Verify weights file has expected columns
        wdf = pd.read_csv(wpath)
        assert f"WT{TAXYEAR}" in wdf.columns
        assert f"WT{TAXYEAR + 1}" in wdf.columns

        # Verify all weights are non-negative
        assert (wdf[f"WT{TAXYEAR}"] >= 0).all(), "Negative weights found"

        # Verify log file was created
        logpath = weight_dir / "xx.log"
        assert logpath.exists(), "Log file not created"

        # Verify log contains solver status
        log_text = logpath.read_text()
        assert "Solver status:" in log_text
        assert "TARGET ACCURACY" in log_text


def test_clarabel_solver_xx_targets_hit():
    """
    Verify the Clarabel solver actually hits the xx area targets
    within the constraint tolerance.
    """
    target_dir = AREAS_FOLDER / "targets"
    out = io.StringIO()

    vdf = _load_taxcalc_data()
    B_csc, targets, labels, _pop_share = _build_constraint_matrix(
        "xx", vdf, out, target_dir=target_dir
    )
    B_csc, targets, labels = _drop_impossible_targets(
        B_csc, targets, labels, out
    )

    n_records = B_csc.shape[1]
    x_opt, _s_lo, _s_hi, info = _solve_area_qp(
        B_csc,
        targets,
        labels,
        n_records,
        out=out,
    )

    # Check that solver succeeded
    status = info["status"]
    assert "Solved" in status, f"Solver status: {status}"

    # Check all targets hit within tolerance
    achieved = np.asarray(B_csc @ x_opt).ravel()
    rel_errors = np.abs(achieved - targets) / np.maximum(np.abs(targets), 1.0)
    eps = 1e-9
    n_violated = int((rel_errors > AREA_CONSTRAINT_TOL + eps).sum())
    assert n_violated == 0, (
        f"{n_violated} targets violated; "
        f"max error = {rel_errors.max() * 100:.3f}%"
    )


# --- Quality report log parser ---


_V = "c00100/cnt=1/scope=0"
_V1 = (  # noqa: E501
    f"    0.500% | target= 12345 | achieved= 12407"
    f" | {_V}/agi=[-9e+99,1.0)/fs=0"
)
_V2 = (
    f"    0.489% | target=   567 | achieved=   570"
    f" | {_V}/agi=[1e+06,9e+99)/fs=1"
)
_V3 = (
    f"    0.478% | target=  1234 | achieved=  1240"
    f" | {_V}/agi=[1e+06,9e+99)/fs=2"
)


def test_parse_log_solved():
    """Test log parser with a synthetic solved log."""
    _hit = "  targets hit: 175/178 (tolerance: +/-0.5% + eps)"
    _mult = (
        "  min=0.000000, p5=0.450000,"
        + " median=0.980000,"
        + " p95=1.650000, max=45.000000"
    )
    _d1 = "    [    0.0000,     0.0000):    2662 (  1.24%)"
    _d2 = "    [    0.0000,     0.1000):    5000 (  2.33%)"
    log_content = "\n".join(
        [
            "Solver status: Solved",
            "Iterations: 42",
            "Solve time: 3.14s",
            "TARGET ACCURACY (178 targets):",
            "  mean |relative error|: 0.001234",
            "  max  |relative error|: 0.004567",
            _hit,
            "  VIOLATED: 3 targets",
            _V1,
            _V2,
            _V3,
            "MULTIPLIER DISTRIBUTION:",
            _mult,
            "  RMSE from 1.0: 0.550000",
            "  distribution (n=214377):",
            _d1,
            _d2,
            "ALL CONSTRAINTS SATISFIED WITHOUT SLACK",
            "",
        ]
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False
    ) as f:
        f.write(log_content)
        f.flush()
        result = parse_log(Path(f.name))

    assert result["status"] == "Solved"
    assert result["solve_time"] == pytest.approx(3.14)
    assert result["mean_err"] == pytest.approx(0.001234)
    assert result["max_err"] == pytest.approx(0.004567)
    assert result["targets_hit"] == 175
    assert result["targets_total"] == 178
    assert result["n_violated"] == 3
    assert result["w_min"] == pytest.approx(0.0)
    assert result["w_median"] == pytest.approx(0.98)
    assert result["w_rmse"] == pytest.approx(0.55)
    assert result["n_records"] == 214377
    assert len(result["violated_details"]) == 3


def test_parse_log_missing():
    """Test log parser with non-existent file."""
    result = parse_log(Path("/nonexistent/path.log"))
    assert result["status"] == "NO LOG"


# --- Scope parsing ---


def test_solve_weights_parse_scope_states():
    """Test scope parsing for 'states'."""
    assert _parse_scope("states") is None
    assert _parse_scope("all") is None


def test_solve_weights_parse_scope_specific():
    """Test scope parsing for specific states."""
    result = _parse_scope("MN,CA,TX")
    assert result == ["MN", "CA", "TX"]


def test_solve_weights_parse_scope_excludes():
    """Test scope parsing excludes PR, US, OA."""
    result = _parse_scope("MN,PR,US,OA,CA")
    assert result == ["MN", "CA"]


# --- Batch area filtering ---


def test_filter_areas_states():
    """Test batch filter for states."""
    areas = ["al", "ca", "mn", "mn01", "xx"]
    assert _filter_areas(areas, "states") == [
        "al",
        "ca",
        "mn",
    ]


def test_filter_areas_cds():
    """Test batch filter for CDs."""
    areas = ["al", "ca", "mn01", "mn02"]
    assert _filter_areas(areas, "cds") == ["mn01", "mn02"]


def test_filter_areas_specific():
    """Test batch filter for specific areas."""
    areas = ["al", "ca", "mn", "tx"]
    assert _filter_areas(areas, "mn,tx") == ["mn", "tx"]


# --- Humanize description ---


def test_humanize_desc():
    """Test human-readable label generation."""
    desc = "c00100/cnt=1/scope=1" + "/agi=[500000.0,1000000.0)/fs=4"
    result = _humanize_desc(desc)
    assert "c00100" in result
    assert "returns" in result
    assert "HoH" in result
    assert "$500K" in result
