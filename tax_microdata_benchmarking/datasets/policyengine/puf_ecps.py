"""
This module enables generation of the Tax-Calculator-format PUF-ECPS.
"""

from .flat_file import create_flat_file, get_population_growth
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
        )

        stacked_file["s006_original"] = stacked_file.s006

        if target_year > 2021:
            path_to_21 = STORAGE_FOLDER / "output" / "puf_ecps_2021.csv.gz"
            if path_to_21.exists():
                stacked_file_21 = pd.read_csv(path_to_21)
            else:
                stacked_file_21 = create_puf_ecps_flat_file(
                    2021, reweight=True
                )
            weights_21 = stacked_file_21.s006
            population_growth = get_population_growth(target_year, 2021)
            stacked_file["s006"] = weights_21 * population_growth
        else:
            stacked_file = reweight(stacked_file, time_period=target_year)
    return stacked_file
