"""
Test the cumulative weight growth factors used to extrapolate the TAXYEAR
sampling weights in tmd/create_taxcalc_sampling_weights.py and in
tmd/areas/create_area_weights.py.
"""

import yaml
import pytest
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR, POPULATION_FILE, RETURNS_FILE
from tmd.utils.weight_growth import cumulative_growth

FIRST_YEAR = TAXYEAR
LAST_YEAR = TAXYEAR + 53


def read_input_yaml(filename):
    with open(
        STORAGE_FOLDER / "input" / filename, "r", encoding="utf-8"
    ) as ifile:
        return yaml.safe_load(ifile.read())


def test_growth_covers_all_weight_years():
    growth = cumulative_growth(FIRST_YEAR, LAST_YEAR)
    assert set(growth) == set(range(FIRST_YEAR, LAST_YEAR + 1))
    assert growth[FIRST_YEAR] == 1.0
    # weights never shrink over the projection period
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        assert growth[year] > 0.0


@pytest.mark.skipif(
    RETURNS_FILE is None,
    reason="TAXYEAR has no tax return projection file",
)
def test_growth_tracks_return_projection():
    """
    Through the last projected return year, cumulative growth must equal
    the growth in projected returns, because a sampling weight counts
    returns.  This is what keeps aggregate AGI and income tax revenue in
    line with the CBO projections used in tests/test_tmd_revenues.py.
    """
    rets = read_input_yaml(RETURNS_FILE)
    growth = cumulative_growth(FIRST_YEAR, LAST_YEAR)
    for year in sorted(rets):
        expect = rets[year] / rets[FIRST_YEAR]
        assert growth[year] == pytest.approx(expect, rel=1e-9)


@pytest.mark.skipif(
    RETURNS_FILE is None,
    reason="TAXYEAR has no tax return projection file",
)
def test_growth_reverts_to_population_after_projection():
    """
    After the last projected return year, returns per capita are held
    fixed, so cumulative growth changes at the rate of population growth.
    """
    pop = read_input_yaml(POPULATION_FILE)
    rets = read_input_yaml(RETURNS_FILE)
    last_rets_year = max(rets)
    assert last_rets_year < LAST_YEAR, "no years left to test"
    growth = cumulative_growth(FIRST_YEAR, LAST_YEAR)
    for year in range(last_rets_year + 1, LAST_YEAR + 1):
        expect = growth[year - 1] * pop[year] / pop[year - 1]
        assert growth[year] == pytest.approx(expect, rel=1e-9)


def test_population_only_growth():
    """
    A None returns_file must reproduce pure population extrapolation,
    which is the TAXYEAR=2021 behavior.
    """
    pop = read_input_yaml(POPULATION_FILE)
    growth = cumulative_growth(FIRST_YEAR, LAST_YEAR, returns_file=None)
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        expect = pop[year] / pop[FIRST_YEAR]
        assert growth[year] == pytest.approx(expect, rel=1e-9)


@pytest.mark.skipif(
    RETURNS_FILE is None,
    reason="TAXYEAR has no tax return projection file",
)
def test_returns_grow_faster_than_population():
    """
    Guard on the premise of using returns rather than population: CBO
    projects the filing share of the population to keep rising.
    """
    pop = read_input_yaml(POPULATION_FILE)
    rets = read_input_yaml(RETURNS_FILE)
    last_rets_year = max(rets)
    rets_growth = rets[last_rets_year] / rets[FIRST_YEAR]
    pop_growth = pop[last_rets_year] / pop[FIRST_YEAR]
    assert rets_growth > pop_growth
