"""
Tests of tmd/areas/create_area_weights.py script.
"""

import sys
from difflib import context_diff
import pytest
from tmd.areas import AREAS_FOLDER
from tmd.areas.make_all import make_all_areas


# @pytest.mark.skip
def test_area_make():
    """
    Make area weights for faux bb area using the faux bb area targets.
    """
    make_all_areas(only_list=["bb"])
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
        sys.stdout.write(">>>>> FULL FILE:\n")
        sys.stdout.write("------------------------------------------------\n")
        sys.stdout.writelines(act)
        sys.stdout.write("------------------------------------------------\n")
        sys.stdout.write(">>>>> DIFFS FILE:\n")
        sys.stdout.writelines(diffs)
        sys.stdout.write("------------------------------------------------\n")
        raise ValueError("ACT vs EXP differences for area/weights/bb.log")
