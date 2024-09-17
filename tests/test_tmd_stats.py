"""
Test of tmd/storage/output/tmd.csv.gz content.
"""

import sys
from difflib import context_diff
import pytest


@pytest.mark.skip
def test_tmd_stats(tests_folder, tmd_variables):
    """
    Create tests/tmd.stats-actual and compare with tests/tmd.stats-expect.
    """
    # create tmd.stats-actual file
    actual_path = tests_folder / "tmd.stats-actual"
    with open(actual_path, "w", encoding="utf-8") as actual_file:
        for vname in sorted(tmd_variables.columns):
            stats = tmd_variables[vname].describe().to_dict()
            for key, val in stats.items():
                actual_file.write(f"{vname}  {key}  {val}\n")
    # compare actual file with expect file
    with open(actual_path, "r", encoding="utf-8") as afile:
        act = afile.readlines()
    expect_path = tests_folder / "tmd.stats-expect"
    with open(expect_path, "r", encoding="utf-8") as efile:
        exp = efile.readlines()
    diffs = list(
        context_diff(act, exp, fromfile="ACTUAL", tofile="EXPECT", n=0)
    )
    if len(diffs) > 0:
        sys.stdout.write(">>>>> TMD.STATS DIFF FILE:\n")
        sys.stdout.write("------------------------------------------------\n")
        sys.stdout.writelines(diffs)
        sys.stdout.write("------------------------------------------------\n")
        actual_path.unlink()
        raise ValueError("ACTUAL vs EXPECT differences")
    actual_path.unlink()
