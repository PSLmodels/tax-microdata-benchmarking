"""
Tests for 117th->118th / 117th->119th Congress crosswalk loading.

Runs the same data-quality checks as
``tmd.areas.prepare.validate_crosswalk`` so that a regression in
either crosswalk file (or in the parameterized loader) is caught by
the test suite.  Tests are automatically skipped if a crosswalk file
is not present.
"""

import pytest

from tmd.areas.prepare.soi_cd_data import (
    _CROSSWALK_PATHS,
    SUPPORTED_CONGRESSES,
    load_crosswalk,
)
from tmd.areas.prepare.validate_crosswalk import (
    _CHANGED_STATES_119,
    _check_afact_sums,
    _check_at_large_recoding,
    _check_changed_vs_unchanged_states,
    _check_pop_conservation,
    _check_target_cd_count,
)


def _requires_crosswalk(congress: int):
    """Skip a test if the crosswalk file for ``congress`` is missing."""
    path = _CROSSWALK_PATHS[congress]
    if not path.exists():
        pytest.skip(f"Crosswalk file not found: {path}")


@pytest.mark.parametrize("congress", SUPPORTED_CONGRESSES)
class TestCrosswalkStructure:
    """Data-quality checks on a single crosswalk file."""

    def test_loads(self, congress):
        """Crosswalk loads and has the expected neutral columns."""
        _requires_crosswalk(congress)
        cw = load_crosswalk(congress=congress)
        for col in ("stabbr", "cd117", "cd_target", "afact2"):
            assert col in cw.columns, f"missing {col}"

    def test_afact_sums(self, congress):
        """afact2 sums to 1.0 per (stabbr, cd117) within CSV rounding."""
        _requires_crosswalk(congress)
        cw = load_crosswalk(congress=congress)
        ok, msg = _check_afact_sums(cw)
        assert ok, msg

    def test_at_large_recoding(self, congress):
        """Single-district states should have cd117 re-coded to '01'."""
        _requires_crosswalk(congress)
        cw = load_crosswalk(congress=congress)
        ok, msg = _check_at_large_recoding(cw)
        assert ok, msg

    def test_pop_conservation(self, congress):
        """Population totals are conserved between source and target."""
        _requires_crosswalk(congress)
        ok, msg = _check_pop_conservation(congress)
        assert ok, msg

    def test_target_cd_count(self, congress):
        """436 distinct target CDs (excluding PR)."""
        _requires_crosswalk(congress)
        cw = load_crosswalk(congress=congress)
        ok, msg = _check_target_cd_count(cw, congress)
        assert ok, msg


class TestCrosswalk118Vs119:
    """Cross-check between the two crosswalks."""

    def test_expected_states_change(self):
        """Only AL, GA, LA, NY, NC should differ between 118 and 119."""
        if not _CROSSWALK_PATHS[118].exists():
            pytest.skip("118 crosswalk not found")
        if not _CROSSWALK_PATHS[119].exists():
            pytest.skip("119 crosswalk not found")
        ok, msg = _check_changed_vs_unchanged_states()
        assert ok, msg

    def test_unchanged_states_have_identical_factors(self):
        """For non-changed states, 117->118 and 117->119 should match."""
        if not _CROSSWALK_PATHS[118].exists():
            pytest.skip("118 crosswalk not found")
        if not _CROSSWALK_PATHS[119].exists():
            pytest.skip("119 crosswalk not found")
        cw118 = load_crosswalk(congress=118)
        cw119 = load_crosswalk(congress=119)
        common = set(cw118["stabbr"]) & set(cw119["stabbr"])
        unchanged = common - _CHANGED_STATES_119
        # Pick a few representative unchanged states to spot-check
        for state in ("CA", "TX", "FL", "OH", "WY"):
            if state not in unchanged:
                continue
            a = (
                cw118[cw118["stabbr"] == state]
                .sort_values(["cd117", "cd_target"])
                .reset_index(drop=True)
            )
            b = (
                cw119[cw119["stabbr"] == state]
                .sort_values(["cd117", "cd_target"])
                .reset_index(drop=True)
            )
            assert len(a) == len(b), f"{state}: row count differs"
            assert (
                a["cd_target"].values == b["cd_target"].values
            ).all(), f"{state}: target CDs differ"
            assert (
                a["afact2"].values == b["afact2"].values
            ).all(), f"{state}: afact2 differs"
