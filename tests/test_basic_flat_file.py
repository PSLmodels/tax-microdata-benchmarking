import os
import pytest

test_mode = os.environ.get("TEST_MODE", "lite")


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
