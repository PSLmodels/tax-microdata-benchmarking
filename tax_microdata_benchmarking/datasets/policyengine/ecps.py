"""
This module enables generation of the Tax-Calculator-format ECPS (or just downloading it from the public GitHub release).
"""

from .flat_file import create_flat_file
from policyengine_us.data import EnhancedCPS_2022
import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.taxcalc import add_taxcalc_outputs


def create_ecps(
    from_scratch: bool = False, time_period: int = 2021
) -> pd.DataFrame:
    """
    Create the Tax-Calculator-format ECPS.

    Args:
        from_scratch (bool): Whether to start from scratch or download the ECPS from the public GitHub release.
        time_period (int): The year to create the ECPS for.

    Returns:
        pd.DataFrame: The ECPS as a DataFrame.
    """
    if from_scratch:
        EnhancedCPS_2022().generate()

    flat_file = create_flat_file(
        source_dataset="enhanced_cps_2022", target_year=time_period
    )

    flat_file_through_tc = add_taxcalc_outputs(flat_file, time_period)

    return flat_file_through_tc
