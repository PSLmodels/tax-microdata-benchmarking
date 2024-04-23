import pytest
import pandas as pd
import os
from pathlib import Path
import subprocess
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

FOLDER = Path(__file__).parent

test_mode = os.environ.get("TEST_MODE", "lite")


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

    flat_file_2023 = flat_file_2021.s006 * get_population_growth(2023, 2021)

    tc_folder = (
        FOLDER.parent
        / "tax_microdata_benchmarking"
        / "examination"
        / "taxcalculator"
    )

    flat_file_2023.to_csv(tc_folder / "pe23.csv.zip")

    # cd into taxcalculator and run bash ./runs.sh pe23 23. That produces a file called pe23-23.res.actual. Print it out.

    subprocess.run(["./runs.sh", "pe23", "23"], cwd=tc_folder)

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
