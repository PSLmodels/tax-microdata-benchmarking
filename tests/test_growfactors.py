"""
Test contents of the raw growth factors file that is read by the
tmd/create_raw_growth_factors.py script during `make data`.

Ported from the TaxData repository (tests/test_growfactors.py).
"""

import pandas as pd
import pytest

from tmd.create_raw_growth_factors import INPUT_FILE
from tmd.imputation_assumptions import TAXYEAR
from tmd.storage import STORAGE_FOLDER

FIRST_TAXCALC_POLICY_YEAR = 2013
MIN_VALUE = 0.15
MAX_VALUE = 8.70


@pytest.fixture(scope="module", name="growfactors")
def fixture_growfactors():
    """
    Return DataFrame containing the raw growth factors used for TAXYEAR.
    """
    path = STORAGE_FOLDER / "input" / INPUT_FILE[TAXYEAR]
    return pd.read_csv(path, index_col="YEAR")


def test_growfactor_start_year(growfactors):
    """
    Check that the growth factors file can support Tax-Calculator Policy
    needs, which begin in FIRST_TAXCALC_POLICY_YEAR.
    """
    assert growfactors.index.min() <= FIRST_TAXCALC_POLICY_YEAR


def test_growfactor_values(growfactors):
    """
    Check that each growth factor value is in a plausible min,max range,
    that every first-year value is one, and that no values are missing.
    """
    first_year = growfactors.index.min()
    for fname in growfactors:
        assert growfactors[fname][first_year] == 1.0
        assert growfactors[fname].min() >= MIN_VALUE
        assert growfactors[fname].max() <= MAX_VALUE
    assert growfactors.isnull().sum().sum() == 0
