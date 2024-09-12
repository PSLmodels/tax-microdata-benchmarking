"""
Tests of tmd/areas/create_area_weights.py script.
"""

from tmd.areas.create_area_weights import create_area_weights_file


def test_area_bb():
    """
    Optimize national weights for the faux bb area targets.
    """
    loss = create_area_weights_file("bb", write_file=False)
    assert loss < 1e-7
