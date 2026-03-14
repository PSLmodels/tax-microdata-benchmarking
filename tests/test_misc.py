"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""

import pytest
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR, CREDIT_CLAIMING
from tests.conftest import create_tmd_records


def test_no_negative_weights(tmd_variables):
    assert tmd_variables.s006.min() >= 0, "Negative weights found"


def test_partnership_s_corp_income(tmd_variables):
    weight = tmd_variables.s006
    e26270 = tmd_variables.e26270
    assert (
        abs((weight * e26270).sum() / 1e9 / 975 - 1) < 0.1
    ), "Partnership/S-Corp income not within 10% of 975 billion dollars"


@pytest.mark.pop
def test_population(tmd_variables):
    weight = tmd_variables.s006
    people = tmd_variables.XTOT
    pop = (weight * people).sum() * 1e-6
    exp_pop = {2021: 331.9, 2022: 334.0}
    r_tol = 0.001
    assert abs(pop / exp_pop[TAXYEAR] - 1) < r_tol, (
        f"{TAXYEAR} population ({pop:.2f}) not within {(r_tol * 100):.1f}% "
        f"of expected {exp_pop[TAXYEAR]:.2f} million"
    )
    # target 2021 (July 1) population is 331.894 million from this URL:
    # https://www.census.gov/newsroom/press-releases/2021/
    #       2021-population-estimates.html
    # target 2022 (July 1) population is 333.996 million from the
    # "Monthly Population Estimates for the United States: April 1, 2020
    # to December 1, 2026 (NA-EST2025-POP)" table which is at this URL:
    # https://www.census.gov/data/tables/time-series/demo/popest/
    #         2020s-national-total.html


@pytest.mark.skip
@pytest.mark.itax
def test_income_tax():

    def compare(name, act, exp, tol):
        assert (
            abs(act / exp - 1) < tol
        ), f"{name}:act,exp,tol= {act} {exp} {tol}"

    # use national tmd files to compute various TAXYEAR income tax statistics
    pol = tc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    rec = create_tmd_records(
        data_path=STORAGE_FOLDER / "output" / "tmd.csv.gz",
        weights_path=STORAGE_FOLDER / "output" / "tmd_weights.csv.gz",
        growfactors_path=STORAGE_FOLDER / "output" / "tmd_growfactors.csv",
    )
    sim = tc.Calculator(policy=pol, records=rec)
    sim.advance_to_year(TAXYEAR)
    sim.calc_all()
    wght = sim.array("s006")
    agi = sim.array("c00100")
    itax = sim.array("iitax")
    mars = sim.array("MARS")
    # check various income tax statistics
    compare("wght_sum", wght.sum(), 194e6, 0.01)
    hiagi = agi >= 1e6
    compare("wght_sum_hiagi", (wght * hiagi).sum(), 0.875e6, 0.01)
    compare("wght_itax_sum", (wght * itax).sum(), 1690e9, 0.01)
    compare("wght_itax_sum_hiagi", ((wght * itax) * hiagi).sum(), 911e9, 0.01)
    # count weighted number of tax units with zero agi by filing status
    agi0 = agi == 0
    compare("wght_sum_agi0_fs0", (wght * agi0).sum(), 28.32e6, 0.01)
    mars1 = mars == 1
    compare("wght_sum_agi0_fs1", (wght * mars1 * agi0).sum(), 27.17e6, 0.01)
    mars2 = mars == 2
    compare("wght_sum_agi0_fs2", (wght * mars2 * agi0).sum(), 1.05e6, 0.01)
    mars4 = mars == 4
    compare("wght_sum_agi0_fs4", (wght * mars4 * agi0).sum(), 0.063e6, 0.01)
    # count weighted number of PUF tax units with zero agi by filing status
    puf = sim.array("data_source") == 1
    pwght = puf * wght
    compare("Pwght_sum_agi0_fs0", (pwght * agi0).sum(), 0.846e6, 0.01)
    mars1 = mars == 1
    compare("Pwght_sum_agi0_fs1", (pwght * mars1 * agi0).sum(), 0.616e6, 0.01)
    mars2 = mars == 2
    compare("Pwght_sum_agi0_fs2", (pwght * mars2 * agi0).sum(), 0.136e6, 0.01)
    mars4 = mars == 4
    compare("Pwght_sum_agi0_fs4", (pwght * mars4 * agi0).sum(), 0.0628e6, 0.01)
