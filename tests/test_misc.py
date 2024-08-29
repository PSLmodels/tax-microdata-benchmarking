"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""


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
        abs(population / 1e6 / 334 - 1) < 0.1
    ), "Population not within 10% of 334 million"
