"""
Tests of tmd/areas/create_area_weights.py script.
"""

import sys
from difflib import context_diff
import pytest
from tmd.areas import AREAS_FOLDER
from tmd.areas.make_all import make_all_areas


@pytest.skip
def test_area_make():
    """
    Optimize national weights for faux bb area using the faux bb area targets.
    """
    # write area/weights/bb_tmd_weights.csv.gz and area/weights/bb.log files
    create_area_weights_file("bb", write_file=True)
    # compare area/weights/bb.log file with area/weights/bb.log-expect file
    wpath = AREAS_FOLDER / "weights"
    with open(wpath / "bb.log", "r", encoding="utf-8") as afile:
        act = afile.readlines()
    with open(wpath / "bb.log-expect", "r", encoding="utf-8") as efile:
        exp = efile.readlines()
    diffs = list(
        context_diff(act, exp, fromfile="ACTUAL", tofile="EXPECT", n=0)
    )
    if len(diffs) > 0:
        sys.stdout.writelines(diffs)
        raise ValueError("ACT vs EXP differences for area/weights/bb.log")
