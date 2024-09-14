"""
Tests of tmd/areas/create_area_weights.py script.
"""

from tmd.areas.create_area_weights import create_area_weights_file


def test_area_bb():
    """
    Optimize national weights using the faux bb area targets.
    """
    rmse = create_area_weights_file("bb", write_file=False)
    assert rmse < 1e-4
    assert 1 == 2, "TO SHOW STDOUT RESULTS"
