"""
Configure pytest unit tests.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR

# convert all numpy floating-point execeptions into errors
np.seterr(all="raise")


def create_tmd_records(
    data_path, weights_path, growfactors_path, exact_calculations=True
):
    """
    Create tc.Records with start_year=TAXYEAR.
    Bypasses tmd_constructor() which hardcodes start_year=2021.
    """
    return tc.Records(
        data=pd.read_csv(data_path),
        start_year=TAXYEAR,
        gfactors=tc.GrowFactors(growfactors_filename=str(growfactors_path)),
        weights=pd.read_csv(weights_path),
        adjust_ratios=None,
        exact_calculations=exact_calculations,
        weights_scale=1.0,
    )


@pytest.fixture(scope="session")
def tests_folder():
    return Path(__file__).parent


@pytest.fixture(scope="session")
def tmd_variables():
    return pd.read_csv(STORAGE_FOLDER / "output" / "tmd.csv.gz")


@pytest.fixture(scope="session")
def tmd_variables_path():
    return STORAGE_FOLDER / "output" / "tmd.csv.gz"


@pytest.fixture(scope="session")
def tmd_weights_path():
    return STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"


@pytest.fixture(scope="session")
def tmd_growfactors_path():
    return STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
