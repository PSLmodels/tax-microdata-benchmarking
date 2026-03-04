"""
Test 2023 tax expenditures calculated using tmd files
against expected tax expenditure values in the tests folder.
"""

import pytest
import numpy as np
from tmd.storage import STORAGE_FOLDER
from tmd.create_taxcalc_input_variables import TAXYEAR
from tmd.utils.taxcalc_utils import get_tax_expenditure_results


@pytest.mark.taxexpdiffs
def test_tax_exp_diffs(
    tests_folder,
    tmd_variables,
    tmd_weights_path,
    tmd_growfactors_path,
):
    _ = get_tax_expenditure_results(
        tmd_variables,
        TAXYEAR,  # input variables data year
        2023,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )
    act_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    with open(act_path, "r", encoding="utf-8") as actfile:
        act = actfile.readlines()
    exp_path = tests_folder / "expected_tax_expenditures"
    with open(exp_path, "r", encoding="utf-8") as expfile:
        exp = expfile.readlines()
    assert len(act) == len(exp), "number of act and exp rows differ"
    a_tol = 0.1  # handles :.1f rounding of tax expenditures
    r_tol = 5e-5  # larger than the np.allclose default value of 1e-5
    diffs = []
    for rowidx, act_row in enumerate(act):
        atok = act_row.split()
        etok = exp[rowidx].split()
        for tokidx in range(3):
            assert atok[tokidx] == etok[tokidx], "act vs exp tokens differ"
        act_val = float(atok[3])
        exp_val = float(etok[3])
        if not np.allclose([act_val], [exp_val], atol=a_tol, rtol=r_tol):
            msg = (
                f"{atok[2]},act,exp,atol,rtol= "
                f"{act_val} {exp_val} {a_tol} {r_tol}\n"
            )
            diffs.append(msg)
    if len(diffs) > 0:
        emsg = "\nACT-vs-EXP TAX EXPENDITURE DIFFERENCES:\n" + "".join(diffs)
        raise ValueError(emsg)
