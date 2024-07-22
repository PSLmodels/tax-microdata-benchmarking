import pandas as pd
from policyengine_us import Microsimulation
from tax_microdata_benchmarking.datasets.puf import (
    PUF_2021,
    create_pe_puf_2021,
)
from tax_microdata_benchmarking.datasets.cps import CPS_2021, create_cps_2021
from tax_microdata_benchmarking.datasets.taxcalc_dataset import (
    create_tc_dataset,
)
from tax_microdata_benchmarking.utils.trace import trace1
from tax_microdata_benchmarking.utils.taxcalc_utils import add_taxcalc_outputs
from tax_microdata_benchmarking.utils.reweight import reweight
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import numpy as np


def create_policyengine_ecps():
    # always create CPS_2021 and PUF_2021
    # (because imputation assumptions may have changed)
    from policyengine_us.data import EnhancedCPS_2022

    tc_ecps_21 = create_tc_dataset(EnhancedCPS_2022, 2021)
    
    sim = Microsimulation(dataset=EnhancedCPS_2022)
    filer = sim.calculate("tax_unit_is_filer", period=2022).values > 0
    tc_ecps_21["data_source"] = np.where(filer, 1, 0)

    print("Adding Tax-Calculator outputs for 2021...")
    tc_ecps_21 = add_taxcalc_outputs(tc_ecps_21, 2021, 2021)
    tc_ecps_21["s006_original"] = tc_ecps_21.s006.values

    trace1("B", tc_ecps_21)

    print("Reweighting...")
    tc_ecps_21 = reweight(tc_ecps_21, 2021)

    trace1("C", tc_ecps_21)

    return tc_ecps_21


if __name__ == "__main__":
    pe_ecps = create_policyengine_ecps()
    pe_ecps.to_csv(STORAGE_FOLDER / "output" / "pe_ecps_2021.csv", index=False)
