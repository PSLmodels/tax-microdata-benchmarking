"""
Tests of tmd/areas/create_area_weights.py script.
"""

from tmd.areas.create_area_weights import create_area_weights_file


def test_area_xx():
    """
    Optimize national weights for faux xx area using the faux xx area targets.
    """
    rmse = create_area_weights_file("xx", write_log=False, write_file=False)
    assert rmse < 1e-4
