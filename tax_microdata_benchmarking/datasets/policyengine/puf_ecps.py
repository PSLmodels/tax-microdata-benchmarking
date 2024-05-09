"""
This module enables generation of the Tax-Calculator-format PUF-ECPS.
"""

from .flat_file import create_flat_file
from policyengine_us.data import EnhancedCPS_2022
import pandas as pd
import numpy as np
from tax_microdata_benchmarking.utils.taxcalc import add_taxcalc_outputs
from tax_microdata_benchmarking.storage import STORAGE_FOLDER


def create_puf_ecps_flat_file(
    target_year: int = 2021,
    reweight: bool = True,
):
    cps_based_flat_file = create_flat_file(
        source_dataset="enhanced_cps_2022", target_year=target_year
    )
    puf_based_flat_file = create_flat_file(
        source_dataset="puf_2022", target_year=target_year
    )
    nonfilers_file = cps_based_flat_file[cps_based_flat_file.is_tax_filer == 0]
    stacked_file = pd.concat(
        [puf_based_flat_file, nonfilers_file]
    ).reset_index(drop=True)

    qbi = np.maximum(
        0,
        stacked_file.e00900
        + stacked_file.e26270
        + stacked_file.e02100
        + stacked_file.e27200,
    )
    stacked_file["PT_binc_w2_wages"] = (
        qbi * 0.314  # Solved in 2021 using adjust_qbi.py
    )
    stacked_file = add_taxcalc_outputs(stacked_file, target_year)
    if reweight:
        from tax_microdata_benchmarking.utils.reweight import (
            reweight,
        )  # Only import if needed- PyTorch can be kept as an optional dependency this way.

        stacked_file["s006_original"] = stacked_file.s006
        stacked_file = reweight(stacked_file, time_period=target_year)
    return stacked_file
