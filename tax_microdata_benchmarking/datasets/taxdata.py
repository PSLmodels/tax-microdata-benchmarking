"""
Any code relating to the taxdata dataset will go here if needed.
"""

from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.taxcalc import add_taxcalc_outputs
from tax_microdata_benchmarking.utils.cloud import download_gh_release_asset


def load_taxdata_puf(time_period: int = 2023):
    """
    Load the Tax-Calculator-format PUF.

    Args:
        time_period (int): The year to load the PUF for. Defaults to 2023.

    Returns:
        pd.DataFrame: The PUF as a DataFrame.
    """
    import pandas as pd

    tc_path = STORAGE_FOLDER / "input" / "tc23.csv"
    if not tc_path.exists():
        td = download_gh_release_asset(
            "nikhilwoodruff/tax-microdata-benchmarking-releases",
            "taxdata-puf-2023",
            "tc23.csv",
        )
        td.to_csv(tc_path, index=False)
    else:
        td = pd.read_csv(STORAGE_FOLDER / "input" / "tc23.csv")
    return add_taxcalc_outputs(td, time_period)
