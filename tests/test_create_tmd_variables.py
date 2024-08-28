"""
Optional test of create_taxcalc_input_variables.py function in order to
generate pytest warnings.
"""

import pytest
from tmd.create_taxcalc_input_variables import create_variable_file


@pytest.mark.skip
def test_create_taxcalc_tmd_variables_file():
    create_variable_file(write_file=False)
