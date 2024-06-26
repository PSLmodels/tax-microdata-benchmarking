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
import pandas as pd


def create_tmd_2021():
    # always create CPS_2021 and PUF_2021
    # (because imputation assumptions may have changed)
    create_cps_2021()
    create_pe_puf_2021()

    tc_puf_21 = create_tc_dataset(PUF_2021, 2015)
    tc_cps_21 = create_tc_dataset(CPS_2021, 2015)

    print("Combining PUF and CPS nonfilers...")

    # Add nonfiler flag to tc_cps_21 with 2022 filing rules
    # (2021 had large changes)
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2021)
    nonfiler = ~(sim.calculate("tax_unit_is_filer", period=2022).values > 0)
    tc_cps_21 = tc_cps_21[nonfiler]

    combined = pd.concat([tc_puf_21, tc_cps_21], ignore_index=True)

    trace1("A", combined)

    # Add Tax-Calculator outputs
    print("Adding Tax-Calculator outputs...")
    combined = add_taxcalc_outputs(combined, 2021)
    combined["s006_original"] = combined.s006.values

    trace1("B", combined)

    print("Reweighting...")
    combined = reweight(combined, 2021, weight_deviation_penalty=0)

    trace1("C", combined)

    return combined


if __name__ == "__main__":
    tmd = create_tmd_2021()
    tmd.to_csv(STORAGE_FOLDER / "output" / "tmd_2021.csv", index=False)
