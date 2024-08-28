from pathlib import Path
import yaml
import pytest
import numpy as np
import pandas as pd
from tmd.storage import STORAGE_FOLDER

# convert all numpy floating-point execeptions into errors
np.seterr(all="raise")


@pytest.fixture(scope="session")
def tests_folder():
    return Path(__file__).parent


@pytest.fixture(scope="session")
def tmd_variables():
    return pd.read_csv(STORAGE_FOLDER / "output" / "tmd.csv.gz")


@pytest.fixture(scope="session")
def tmd_weights_path():
    return STORAGE_FOLDER / "output" / "tmd_weights.csv.gz"


@pytest.fixture(scope="session")
def tmd_gfactors_path():
    return STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
