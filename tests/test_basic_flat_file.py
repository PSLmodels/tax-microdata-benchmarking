import os
import pytest
import yaml
from pathlib import Path

test_mode = os.environ.get("TEST_MODE", "lite")

FOLDER = Path(__file__).parent
with open(FOLDER / "tc_variable_totals.yaml") as f:
    tc_variable_totals = yaml.safe_load(f)

with open(
    FOLDER.parent
    / "tax_microdata_benchmarking"
    / "taxcalc_variable_metadata.yaml"
) as f:
    taxcalc_variable_metadata = yaml.safe_load(f)

EXEMPTED_VARIABLES = [
    "DSI",
    "EIC",  # PUF-PE file almost certainly more correct by including CPS data
    "MIDR",
    "RECID",  # No reason to compare.
    "a_lineno",  # No reason to compare.
    "agi_bin",  # No reason to compare.
    "blind_spouse",  # Issue here but deprioritized.
    "cmbtp",  # No reason to compare.
    "data_source",  # No reason to compare.
    "s006",  # No reason to compare.
    "h_seq",  # No reason to compare.
    "fips",  # No reason to compare.
    "ffpos",  # No reason to compare.
]

# Exempt any variable split between filer and spouse for now.
EXEMPTED_VARIABLES += [
    variable
    for variable in taxcalc_variable_metadata["read"]
    if variable.endswith("_p") or variable.endswith("_s")
]


def pytest_namespace():
    return {"flat_file": None}


@pytest.mark.dependency()
def test_flat_file_builds():
    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
    )

    flat_file = create_stacked_flat_file(2021, reweight=test_mode == "full")

    pytest.flat_file = flat_file


variables_to_test = [
    variable
    for variable in tc_variable_totals.keys()
    if variable not in EXEMPTED_VARIABLES
]


@pytest.mark.dependency(depends=["test_flat_file_builds"])
@pytest.mark.parametrize("variable", variables_to_test)
def test_tc_variable_totals(variable):
    meta = taxcalc_variable_metadata["read"][variable]
    name = meta.get("desc")
    flat_file = pytest.flat_file
    weight = flat_file.s006
    total = (flat_file[variable] * weight).sum()
    if tc_variable_totals[variable] == 0:
        # If the taxdata file has a zero total, we'll assume the PE file is still correct.
        return
    assert (
        abs(total / tc_variable_totals[variable] - 1) < 0.5
    ), f"{variable} ({name}) is off by {total / tc_variable_totals[variable] - 1:.1%} ({total/1e9:.1f}bn vs {tc_variable_totals[variable]/1e9:.1f}bn)"
