"""
Tests for the state target preparation pipeline.

Three test levels:
  1. Unit: SOI share computation and rescaling
  2. Integration: recipe expansion and target file structure
  3. End-to-end: full pipeline on a single state
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmd.areas.prepare.constants import (
    ALL_SHARING_MAPPINGS,
    AreaType,
)
from tmd.areas.prepare.census_population import get_state_population
from tmd.areas.prepare.extended_targets import build_extended_targets
from tmd.areas.prepare.soi_state_data import (
    create_soilong,
    create_state_base_targets,
)
from tmd.areas.prepare.target_file_writer import write_area_target_files
from tmd.areas.prepare.target_sharing import (
    compute_soi_shares,
    prepare_area_targets,
)

# --- Paths ---

REPO_ROOT = Path(__file__).parent.parent
_PREPARE = REPO_ROOT / "tmd" / "areas" / "prepare"
SOI_RAW_DIR = _PREPARE / "data" / "soi_states"
CACHED_ALLVARS = (
    REPO_ROOT / "tmd" / "storage" / "output" / "cached_allvars.csv"
)
RECIPE_PATH = _PREPARE / "recipes" / "states.json"
VARMAP_PATH = _PREPARE / "recipes" / "state_variable_mapping.csv"

_EXCLUDE = {"US", "OA", "PR"}


# ---- Unit tests: SOI share computation ----


class TestSOIShares:
    """Test that SOI shares are computed correctly."""

    @pytest.fixture(scope="class")
    def shares_data(self):
        """Load SOI data and compute shares for 2022."""
        soilong = create_soilong(SOI_RAW_DIR, years=[2022])
        pop_df = get_state_population(2022)
        base_targets = create_state_base_targets(soilong, pop_df, 2022)
        return compute_soi_shares(base_targets, ALL_SHARING_MAPPINGS)

    def test_shares_sum_to_one(self, shares_data):
        """Non-zero 51-state shares sum to 1.0."""
        state_shares = shares_data[~shares_data["stabbr"].isin(_EXCLUDE)]
        group_cols = [
            "basesoivname",
            "count",
            "scope",
            "fstatus",
            "agistub",
        ]
        group_sums = state_shares.groupby(group_cols)["soi_share"].sum()
        nonzero = group_sums[group_sums > 0]
        np.testing.assert_allclose(
            nonzero.values,
            1.0,
            atol=1e-10,
            err_msg="51-state shares should sum to 1.0",
        )

    def test_negative_shares_only_for_loss_variables(self, shares_data):
        """Negative shares only for variables with losses."""
        neg = shares_data[shares_data["soi_share"] < 0]
        loss_vars = {"26270", "00900"}
        neg_vars = set(neg["basesoivname"].unique())
        assert neg_vars.issubset(
            loss_vars
        ), f"Unexpected negatives: {neg_vars - loss_vars}"

    def test_mn_agi_share_reasonable(self, shares_data):
        """MN AGI share is roughly 1-3%."""
        mn_agi = shares_data[
            (shares_data["stabbr"] == "MN")
            & (shares_data["basesoivname"] == "00100")
            & (shares_data["count"] == 0)
            & (shares_data["agistub"] == 0)
        ]
        assert len(mn_agi) == 1
        share = mn_agi["soi_share"].values[0]
        assert 0.01 < share < 0.03

    def test_xtot_equals_us_population(self):
        """XTOT 51-state sum equals US Census population."""
        pop_df = get_state_population(2022)
        us_pop = pop_df.loc[pop_df["stabbr"] == "US", "population"].values[0]
        soilong = create_soilong(SOI_RAW_DIR, years=[2022])
        base = create_state_base_targets(soilong, pop_df, 2022)
        xtot = base[base["basesoivname"] == "XTOT"]
        state_sum = xtot[~xtot["stabbr"].isin(_EXCLUDE)]["target"].sum()
        assert state_sum == us_pop


# ---- Integration tests: target file structure ----


class TestTargetFileWriter:
    """Test recipe expansion produces correct target files."""

    @pytest.fixture(scope="class")
    def mn_targets(self, tmp_path_factory):
        """Run the full pipeline for MN."""
        enhanced = prepare_area_targets(
            area_type=AreaType.STATE,
            area_data_year=2022,
        )
        enhanced = enhanced[
            (enhanced["area"] == "MN") & ~enhanced["area"].isin(_EXCLUDE)
        ]
        extra = build_extended_targets(
            cached_allvars_path=CACHED_ALLVARS,
            soi_year=2022,
            areas=["MN"],
        )
        out_dir = tmp_path_factory.mktemp("targets")
        result = write_area_target_files(
            recipe_path=RECIPE_PATH,
            enhanced_targets=enhanced,
            variable_mapping_path=VARMAP_PATH,
            output_dir=out_dir,
            extra_targets=extra,
        )
        rows = pd.read_csv(out_dir / "mn_targets.csv")
        return rows, result

    def test_mn_target_count(self, mn_targets):
        """MN has ~178 targets (base + extended)."""
        rows, _ = mn_targets
        assert 170 <= len(rows) <= 185

    def test_required_columns(self, mn_targets):
        """Target file has expected columns."""
        rows, _ = mn_targets
        expected = {
            "varname",
            "count",
            "scope",
            "agilo",
            "agihi",
            "fstatus",
            "target",
        }
        assert set(rows.columns) == expected

    def test_no_nan_targets(self, mn_targets):
        """No target value is NaN."""
        rows, _ = mn_targets
        assert not rows["target"].isna().any()

    def test_xtot_is_first(self, mn_targets):
        """XTOT (population) is the first row."""
        rows, _ = mn_targets
        assert rows.iloc[0]["varname"] == "XTOT"
        assert rows.iloc[0]["scope"] == 0

    def test_extended_variables_present(self, mn_targets):
        """Extended target variables are present."""
        rows, _ = mn_targets
        varnames = set(rows["varname"])
        for var in [
            "capgains_net",
            "e00600",
            "c19200",
            "e18400",
            "e18500",
            "eitc",
            "ctc_total",
        ]:
            assert var in varnames, f"{var} missing"


# ---- End-to-end test ----


def test_prepare_mn_end_to_end(tmp_path):
    """Full pipeline for MN produces valid output."""
    enhanced = prepare_area_targets(
        area_type=AreaType.STATE,
        area_data_year=2022,
    )
    enhanced = enhanced[
        (enhanced["area"] == "MN") & ~enhanced["area"].isin(_EXCLUDE)
    ]
    extra = build_extended_targets(
        cached_allvars_path=CACHED_ALLVARS,
        soi_year=2022,
        areas=["MN"],
    )
    result = write_area_target_files(
        recipe_path=RECIPE_PATH,
        enhanced_targets=enhanced,
        variable_mapping_path=VARMAP_PATH,
        output_dir=tmp_path,
        extra_targets=extra,
    )
    assert "MN" in result
    assert result["MN"] > 100
    fpath = tmp_path / "mn_targets.csv"
    assert fpath.exists()
    with open(fpath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == result["MN"]
    assert rows[0]["varname"] == "XTOT"
