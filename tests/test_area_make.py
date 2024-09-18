"""
Tests of tmd/areas/make_all.py script.
"""

import sys
from difflib import context_diff
import pytest
from tmd.areas import AREAS_FOLDER
from tmd.areas.make_all import make_all_areas


@pytest.mark.skip
def test_area_make():
    """
    Compare areas/weights/bb.log file with areas/weights/bb.log-expect file.
    """
    make_all_areas(1, make_only_list=["bb"])
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
        sys.stdout.write("------------------------------------------------\n")
        sys.stdout.writelines(diffs)
        sys.stdout.write("------------------------------------------------\n")
        raise ValueError("ACT vs EXP differences for areas/weights/bb.log")
