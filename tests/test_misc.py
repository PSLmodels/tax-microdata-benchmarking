"""
Miscellaneous tests of tmd.csv variable weighted totals.
"""


def test_no_negative_weights(tmd_variables):
    assert tmd_variables.s006.min() >= 0, "Negative weights found"


def test_qbided_close_to_soi(tmd_variables):
    weight = tmd_variables.s006
    qbid = tmd_variables.qbided
    assert (
        abs((weight * qbid).sum() / 1e9 / 205.8 - 1) < 0.25
    ), "QBIDED not within 25% of 205.8 billion dollars"


def test_partnership_s_corp_income_close_to_soi(tmd_variables):
    weight = tmd_variables.s006
    e26270 = tmd_variables.e26270
    assert (
        abs((weight * e26270).sum() / 1e9 / 975 - 1) < 0.1
    ), "Partnership/S-Corp income not within 10% of 975 billion dollars"
