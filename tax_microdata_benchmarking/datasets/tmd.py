import taxcalc

taxcalc.Records

from tax_microdata_benchmarking.datasets.puf import PUF_2021, create_pe_puf_2021
from tax_microdata_benchmarking.datasets.cps import CPS_2021, create_cps_2021
from tax_microdata_benchmarking.datasets.taxcalc import create_tc_dataset, add_taxcalc_outputs
from tax_microdata_benchmarking.utils.reweight import reweight
import pandas as pd

def create_tmd_2021():
    # create_cps_2021()
    # create_pe_puf_2021()
    tc_puf_21 = create_tc_dataset(PUF_2021)
    tc_cps_21 = create_tc_dataset(CPS_2021)

    # Add nonfiler flag to tc_cps_21 with 2022 filing rules (2021 had large changes)
    from policyengine_us import Microsimulation
    sim = Microsimulation(dataset=CPS_2021)
    nonfiler = ~(sim.calculate("tax_unit_is_filer", period=2022).values > 0)
    tc_cps_21 = tc_cps_21[nonfiler]

    combined = pd.concat([tc_puf_21, tc_cps_21], ignore_index=True)

    # Add Tax-Calculator outputs

    combined = add_taxcalc_outputs(combined, 2021)
    combined["s006_original"] = combined.s006.values
    combined = reweight(combined, 2021, weight_deviation_penalty=0)

    return combined

if __name__ == "__main__":
    create_tmd_2021().to_csv("tmd_2021.csv", index=False)
