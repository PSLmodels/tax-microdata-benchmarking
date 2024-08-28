"""
Test 2023 and 2026 tax expenditures calculated using tmd files
against expected tax expenditure values in the tests folder.
"""

import difflib
import pytest
import numpy as np
import pandas as pd
from tmd.utils.taxcalc_utils import get_tax_expenditure_results
from tmd.storage import STORAGE_FOLDER


@pytest.mark.taxexpdiffs
def test_tax_expenditures_differences(
    tests_folder,
    tmd_variables,
    tmd_weights_path,
    tmd_growfactors_path,
):
    reltol = 0.003  # relative np.allclose tolerance
    _ = get_tax_expenditure_results(
        tmd_variables,
        2021,  # input variables data year
        2023,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )
    _ = get_tax_expenditure_results(
        tmd_variables,
        2021,  # input variables data year
        2026,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )
    act_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    exp_path = tests_folder / "expected_tax_expenditures"
    actdf = pd.read_csv(act_path, sep=" ", header=None)
    expdf = pd.read_csv(exp_path, sep=" ", header=None)
    actval = actdf.iloc[:, 3].to_numpy(dtype=np.float64)
    expval = expdf.iloc[:, 3].to_numpy(dtype=np.float64)
    same = np.allclose(actval, expval, atol=0.0, rtol=reltol)
    if same:
        return
    # if same is False
    with open(act_path, "r") as actfile:
        act = actfile.readlines()
    with open(exp_path, "r") as expfile:
        exp = expfile.readlines()
    diffs = list(
        difflib.context_diff(act, exp, fromfile="actual", tofile="expect", n=0)
    )
    if len(diffs) > 0:
        emsg = "\nThere are actual vs expect tax expenditure differences:\n"
        for line in diffs:
            emsg += line
        raise ValueError(emsg)
