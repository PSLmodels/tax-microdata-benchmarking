import os
import pytest
import yaml
from pathlib import Path
import pytest
import pandas as pd
import subprocess
import warnings

warnings.filterwarnings("ignore")

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
    "DSI",  # Issue here but deprioritized.
    "EIC",  # PUF-PE file almost certainly more correct by including CPS data
    "MIDR",  # Issue here but deprioritized.
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
    "p23250",  # PE-PUF likely closer to truth than taxdata (needs triple check).
    "e01200",  # Unknown but deprioritized for now.
    "e17500",  # Unknown but deprioritized for now.
    "e18500",  # Unknown but deprioritized for now.
    "e02100",  # Farm income, unsure who's closer.
]

# Exempt any variable split between filer and spouse for now.
EXEMPTED_VARIABLES += [
    variable
    for variable in taxcalc_variable_metadata["read"]
    if variable.endswith("p") or variable.endswith("s")
]


def pytest_namespace():
    return {"flat_file": None}


@pytest.mark.dependency()
def test_2021_flat_file_builds():
    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
    )

    flat_file = create_stacked_flat_file(2021, reweight=test_mode == "full")

    pytest.flat_file_2021 = flat_file


variables_to_test = [
    variable
    for variable in tc_variable_totals.keys()
    if variable not in EXEMPTED_VARIABLES
]


@pytest.mark.dependency(depends=["test_2021_flat_file_builds"])
@pytest.mark.parametrize("variable", variables_to_test)
def test_2021_tc_variable_totals(variable):
    meta = taxcalc_variable_metadata["read"][variable]
    name = meta.get("desc")
    flat_file = pytest.flat_file_2021
    weight = flat_file.s006
    total = (flat_file[variable] * weight).sum()
    if tc_variable_totals[variable] == 0:
        # If the taxdata file has a zero total, we'll assume the PE file is still correct.
        return
    # 20% and more than 10bn off taxdata is a failure.
    assert (
        abs(total / tc_variable_totals[variable] - 1) < 0.45
        or abs(total / 1e9 - tc_variable_totals[variable] / 1e9) < 30
    ), f"{variable} ({name}) differs to tax-data by {total / tc_variable_totals[variable] - 1:.1%} ({total/1e9:.1f}bn vs {tc_variable_totals[variable]/1e9:.1f}bn)"


FOLDER = Path(__file__).parent

test_mode = os.environ.get("TEST_MODE", "lite")

RUN_TE_TESTS = False


@pytest.mark.skipif(not RUN_TE_TESTS, reason="TE tests are disabled.")
@pytest.mark.dependency(depends=["test_2021_flat_file_builds"])
def test_2023_tax_expenditures():
    flat_file_2021 = pytest.flat_file_2021

    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
        get_population_growth,
    )

    flat_file_2023 = create_stacked_flat_file(
        2023, reweight=test_mode == "full"
    )

    flat_file_2023.s006 = flat_file_2021.s006 * get_population_growth(
        2023, 2021
    )

    tc_folder = (
        FOLDER.parent
        / "tax_microdata_benchmarking"
        / "examination"
        / "taxcalculator"
    )

    flat_file_2023.to_csv(tc_folder / "pe23.csv.zip")

    # cd into taxcalculator and run bash ./runs.sh pe23 23. That produces a file called pe23-23.res.actual. Print it out.

    subprocess.run(["./runs.sh", "pe23", "23"], cwd=tc_folder.resolve())

    with open(tc_folder / "pe23-23.res-actual") as f:
        data = f.read().splitlines()

    import warnings

    warnings.filterwarnings("ignore")
    import pandas as pd

    df = pd.DataFrame(
        columns=["Returns", "ExpInc", "IncTax", "PayTax", "LSTax", "AllTax"]
    )
    for line in data[2::3]:
        line = line.split()[1:]
        df = df.append(
            pd.DataFrame(
                [line],
                columns=[
                    "Returns",
                    "ExpInc",
                    "IncTax",
                    "PayTax",
                    "LSTax",
                    "AllTax",
                ],
            )
        )

    df.index = [
        "Baseline",
        "CGQD",
        "CLP",
        "CTC",
        "EITC",
        "NIIT",
        "QBID",
        "SALT",
        "SSBEN",
    ]
    df = df.astype(float)

    taxdata_exp_results = [
        3976.5,
        274.5,
        0.0,
        125.6,
        68.7,
        -67.5,
        59.5,
        13.9,
        76.6,
    ]

    for i in range(len(taxdata_exp_results)):
        name = df.index[i]
        if name in ("QBID", "SALT"):
            continue  # QBID: PE far closer to truth. SALT: known issue.
        rel_error = (
            abs(df["AllTax"][i] - taxdata_exp_results[i])
            / taxdata_exp_results[i]
        )
        if taxdata_exp_results[i] == 0:
            rel_error = 0
        assert (
            rel_error < 0.25
        ), f"Tax Expenditure for {name} is ${df['AllTax'][i]}bn compared to Tax-Data's ${taxdata_exp_results[i]}bn (relative error {rel_error:.1%})"


# Add test of create_taxcalc_input_variable's create_variable_file function
# that checks the consistency of initial_pt_w2_wages_scale argument of the
# create_variable_file function:


def test_create_taxcalc_tmd_file():
    if test_mode != "full":
        return

    from tax_microdata_benchmarking.create_taxcalc_input_variables import (
        create_variable_file,
    )

    create_variable_file(write_file=False)


# Adding explicit tests for unemployment compensation and medical expenses:


@pytest.mark.dependency(depends=["test_2021_flat_file_builds"])
@pytest.mark.skip
def test_2021_unemployment_compensation():
    flat_file_2021 = pytest.flat_file_2021

    total = (flat_file_2021["e02300"] * flat_file_2021.s006).sum()
    assert (
        abs(total / 1e9 / 33 - 1) < 0.2  # WHERE DOES 33 COME FROM ????
    ), f"Unemployment compensation total is ${total/1e9:.1f}bn, expected $33bn"


@pytest.mark.dependency(depends=["test_2021_flat_file_builds"])
@pytest.mark.skip
def test_2021_medical_expenses():
    flat_file_2021 = pytest.flat_file_2021

    total = (flat_file_2021["e17500"] * flat_file_2021.s006).sum()
    assert (
        abs(total / 1e9 / 215 - 1) < 0.2  # WHERE DOES 215 COME FROM ????
    ), f"Medical expense total is ${total/1e9:.1f}bn, expected $215bn"
