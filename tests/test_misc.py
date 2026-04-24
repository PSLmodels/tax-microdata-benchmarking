"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""

import pytest
from tmd.imputation_assumptions import TAXYEAR


def test_no_negative_weights(tmd_variables):
    assert tmd_variables.s006.min() >= 0, "Negative weights found"


@pytest.mark.pop
def test_population(tmd_variables):
    weight = tmd_variables.s006
    people = tmd_variables.XTOT
    pop = (weight * people).sum() * 1e-6
    exp_pop = {2021: 331.9, 2022: 334.0}
    # target 2021 (July 1) population is 331.894 million from this URL:
    # https://www.census.gov/newsroom/press-releases/2021/
    #       2021-population-estimates.html
    # target 2022 (July 1) population is 333.996 million from the
    # "Monthly Population Estimates for the United States: April 1, 2020
    # to December 1, 2026 (NA-EST2025-POP)" table which is at this URL:
    # https://www.census.gov/data/tables/time-series/demo/popest/
    #         2020s-national-total.html
    r_tol = 0.001
    assert abs(pop / exp_pop[TAXYEAR] - 1) < r_tol, (
        f"{TAXYEAR} population ({pop:.2f}) not within {(r_tol * 100):.1f}% "
        f"of expected {exp_pop[TAXYEAR]:.2f} million"
    )
