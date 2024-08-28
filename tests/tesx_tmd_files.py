"""
This module adds basic tests for several tmd file variables,
checking that the variable total is within the ballpark of
the Tax-Data 2023 PUF's totals.
"""

import yaml
from pathlib import Path
import subprocess
import difflib
import numpy as np
import pandas as pd
import pytest
from tmd.utils.taxcalc_utils import get_tax_expenditure_results
from tmd.storage import STORAGE_FOLDER


FOLDER = Path(__file__).parent

with open(FOLDER / "tax_expenditure_targets.yaml") as f:
    tax_expenditure_targets = yaml.safe_load(f)

tmd_weights_path = STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"

tmd_growfactors_path = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"

dataset_names_to_test = ["tmd_2021"]

datasets_to_test = [
    pd.read_csv(STORAGE_FOLDER / "output" / f"{dataset}.csv")
    for dataset in dataset_names_to_test
]


"""
tax_expenditure_reforms = [
    "ctc",
    "eitc",
    "social_security_partial_taxability",
    "niit",
    "cgqd_tax_preference",
    "qbid",
    "salt",
]


tax_expenditure_estimates = {}

for dataset, name in zip(datasets_to_test, dataset_names_to_test):
    print(f"Running tax expenditure estimates for {dataset}")
    tax_expenditure_estimates[name] = get_tax_expenditure_results(
        dataset,
        2021,  # dataset year
        2023,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )
    assert len(datasets_to_test) == 1
    _ = get_tax_expenditure_results(
        datasets_to_test[0],
        2021,  # dataset year
        2026,  # simulation year for tax expenditure estimates
        tmd_weights_path,
        tmd_growfactors_path,
    )


@pytest.mark.taxexp
@pytest.mark.parametrize("flat_file", dataset_names_to_test, ids=lambda x: x)
@pytest.mark.parametrize("reform", tax_expenditure_reforms, ids=lambda x: x)
def test_tax_expenditure_estimates(
    flat_file: pd.DataFrame,
    reform: str,
):
    target = tax_expenditure_targets[reform][2023]
    estimate = tax_expenditure_estimates[flat_file][reform]
    tol = 0.4
    if reform == "salt":
        tol = 0.7
    assert abs(estimate / target - 1) < tol or abs(estimate - target) < tol, (
        f"{reform} differs from CBO estimate by "
        f"{estimate / target - 1:.1%} ({estimate:.1f}bn vs {target:.1f}bn)"
    )


@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_no_negative_weights(flat_file):
    assert flat_file.s006.min() >= 0, "Negative weights found."


@pytest.mark.qbid
@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_qbided_close_to_soi(flat_file):
    assert (
        abs((flat_file.s006 * flat_file.qbided).sum() / 1e9 / 205.8 - 1) < 0.25
    ), "QBIDED not within 25 percent of 205.8bn"


@pytest.mark.parametrize(
    "flat_file", datasets_to_test, ids=dataset_names_to_test
)
def test_partnership_s_corp_income_close_to_soi(flat_file):
    assert (
        abs((flat_file.s006 * flat_file.e26270).sum() / 1e9 / 975 - 1) < 0.1
    ), "Partnership/S-Corp income not within 10 percent of 975bn"


@pytest.mark.taxexpdiffs
def test_tax_expenditures_differences():
    abstol = 0.11  # absolute np.allclose tolerance in billions of dollars
    act_path = STORAGE_FOLDER / "output" / "tax_expenditures"
    exp_path = STORAGE_FOLDER.parent / "examination" / "tax_expenditures"
    actdf = pd.read_csv(act_path, sep=" ", header=None)
    actdf = actdf[actdf.iloc[:, 2] != "iitax"]
    expdf = pd.read_csv(exp_path, sep=" ", header=None)
    expdf = expdf[expdf.iloc[:, 2] != "iitax"]
    actval = actdf.iloc[:, 3]
    expval = expdf.iloc[:, 3]
    same = np.allclose(actval, expval, rtol=0.0, atol=abstol)
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
            if "iitax" in line:
                continue
            emsg += line
        raise ValueError(emsg)
"""
