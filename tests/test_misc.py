"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""

import pytest
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER


def test_no_negative_weights(tmd_variables):
    assert tmd_variables.s006.min() >= 0, "Negative weights found"


def test_partnership_s_corp_income(tmd_variables):
    weight = tmd_variables.s006
    e26270 = tmd_variables.e26270
    assert (
        abs((weight * e26270).sum() / 1e9 / 975 - 1) < 0.1
    ), "Partnership/S-Corp income not within 10% of 975 billion dollars"


def test_population(tmd_variables):
    weight = tmd_variables.s006
    people = tmd_variables.XTOT
    population = (weight * people).sum()
    assert (
        abs(population / 1e6 / 334.18 - 1) < 0.01
    ), "Population not within 1% of 334.18 million"


@pytest.mark.itax
def test_income_tax():

    def compare(name, act, exp, tol):
        assert (
            abs(act / exp - 1) < tol
        ), f"{name}:act,exp,tol= {act} {exp} {tol}"

    # use national tmd files to compute various 2021 income tax statistics
    rec = tc.Records.tmd_constructor(
        data_path=(STORAGE_FOLDER / "output" / "tmd.csv.gz"),
        weights_path=(STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"),
        growfactors_path=(STORAGE_FOLDER / "output" / "tmd_growfactors.csv"),
        exact_calculations=True,
    )
    sim = tc.Calculator(records=rec, policy=tc.Policy())
    sim.advance_to_year(2021)
    sim.calc_all()
    wght = sim.array("s006")
    agi = sim.array("c00100")
    itax = sim.array("iitax")
    mars = sim.array("MARS")
    # check various income tax statistics
    compare("wght_sum", wght.sum(), 184e6, 0.01)
    hiagi = agi >= 1e6
    compare("wght_sum_hiagi", (wght * hiagi).sum(), 0.850e6, 0.01)
    compare("wght_itax_sum", (wght * itax).sum(), 1523e9, 0.01)
    compare("wght_itax_sum_hiagi", ((wght * itax) * hiagi).sum(), 832e9, 0.01)
    # count weighted number of tax units with zero agi by filing status
    agi0 = agi == 0
    compare("wght_sum_agi0_fs0", (wght * agi0).sum(), 16.22e6, 0.01)
    mars1 = mars == 1
    compare("wght_sum_agi0_fs1", (wght * mars1 * agi0).sum(), 12.01e6, 0.01)
    mars2 = mars == 2
    compare("wght_sum_agi0_fs2", (wght * mars2 * agi0).sum(), 2.00e6, 0.01)
    mars4 = mars == 4
    compare("wght_sum_agi0_fs4", (wght * mars4 * agi0).sum(), 1.53e6, 0.01)
