"""
This module enables the creation of a general, hands-off PUF data input file for Tax-Calculator.
"""

from tax_microdata_benchmarking.utils.cloud import download_gh_release_asset
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm


def load_puf(
    skip_if_exists: bool = True,
) -> pd.DataFrame:
    """
    Download the IRS Public Use File (PUF) from a private GitHub repo.

    Args:
        skip_if_exists (bool): Whether to skip the download if the file already exists.

    Returns:
        pd.DataFrame: The PUF as a DataFrame.
    """
    path = STORAGE_FOLDER / "input" / "puf_2015.csv.gz"
    if skip_if_exists and path.exists():
        return pd.read_csv(path)
    puf = download_gh_release_asset(
        repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
        release_name="puf-2015",
        asset_name="puf_2015.csv.gz",
    )
    return puf


with open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml") as f:
    taxcalc_variable_metadata = yaml.safe_load(f)


def create_puf(time_period: int = 2021) -> pd.DataFrame:
    puf = load_puf().dropna()

    microdata = pd.DataFrame()

    # Columns to lowercase
    puf.columns = puf.columns.str.lower()

    # Scan the PUF for columns in taxcalc's variable metadata, then pass as-is.
    variables = list(taxcalc_variable_metadata.get("read", []))
    for variable in variables:
        if variable in puf.columns:
            microdata[variable] = puf[variable]

    microdata["RECID"] = list(range(1, len(microdata) + 1))
    microdata["MARS"] = puf.mars.clip(1, 5)
    microdata["FLPDYR"] = time_period
    microdata.s006 /= 1e2

    FILER_SUM_COLUMNS = [
        "e00200",
        "e00900",
        "e02100",
    ]
    for column in FILER_SUM_COLUMNS:
        microdata[column + "p"] = microdata[column]
        microdata[column + "s"] = 0

    return microdata
