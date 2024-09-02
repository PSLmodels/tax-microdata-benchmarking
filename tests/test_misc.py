"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""

import numpy as np


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


def test_agi_bin(tmd_variables):
    bin = tmd_variables.agi_bin
    assert np.min(bin) == 0, "Minimum value in agi_bin is not zero"
    assert np.max(bin) == 6, "Maximum value in agi_bin is not six"
