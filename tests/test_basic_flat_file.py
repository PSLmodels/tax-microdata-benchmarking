import os
import pytest
import yaml
from pathlib import Path

test_mode = os.environ.get("TEST_MODE", "lite")

FOLDER = Path(__file__).parent
with open(FOLDER / "tc_variable_totals.yaml") as f:
    tc_variable_totals = yaml.safe_load(f)


def pytest_namespace():
    return {"flat_file": None}


@pytest.mark.dependency()
def test_flat_file_builds():
    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
    )

    flat_file = create_stacked_flat_file(2021, reweight=test_mode == "full")

    pytest.flat_file = flat_file


@pytest.mark.dependency(depends=["test_flat_file_builds"])
def test_taxable_pension_income():
    flat_file = pytest.flat_file
    weight = flat_file.s006
    pension_income = flat_file.e01700
    total_pension_income = (pension_income * weight).sum()
    assert abs(total_pension_income / 750e9 - 1) < 0.05


@pytest.mark.dependency(depends=["test_flat_file_builds"])
@pytest.mark.parametrize("variable", tc_variable_totals.keys())
def test_tc_variable_totals(variable):
    flat_file = pytest.flat_file
    weight = flat_file.s006
    total = (flat_file[variable] * weight).sum()
    assert abs(total / tc_variable_totals[variable] - 1) < 0.5
