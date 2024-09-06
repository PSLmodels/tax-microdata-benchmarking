"""
Tests of tmd/areas/create_area_weights.py script.
"""

from tmd.areas.create_area_weights import create_area_weights_file


def test_area_zz():
    """
    Optimize national weights for the faux zz area targets.
    """
    loss = create_area_weights_file("zz", write_file=False)
    assert loss < 1e-02
