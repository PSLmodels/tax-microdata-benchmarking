"""
Any code relating to the taxdata dataset will go here if needed.
"""

from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.taxcalc import add_taxcalc_outputs


def load_taxdata_puf(time_period: int = 2023):
    """
    Load the Tax-Calculator-format PUF.

    Args:
        time_period (int): The year to load the PUF for. Defaults to 2023.

    Returns:
        pd.DataFrame: The PUF as a DataFrame.
    """
    import pandas as pd

    td = pd.read_csv(STORAGE_FOLDER / "input" / "tc23.csv")
    return add_taxcalc_outputs(td, time_period)
