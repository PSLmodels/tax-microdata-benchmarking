import numpy as np
import pandas as pd
from tmd.imputation_assumptions import TAXYEAR, CPS_WEIGHTS_SCALE
from tmd.datasets.puf import create_tc_puf
from tmd.datasets.cps import create_tc_cps
from tmd.utils.taxcalc_utils import add_taxcalc_outputs
from tmd.utils.reweight import reweight


def create_tmd_dataframe(taxyear: int) -> pd.DataFrame:
    """
    Create DataFrame for given taxyear containing PUF filers and CPS nonfilers.
    """
    # always call create_tc_puf and create_tc_cps
    # (because imputation assumptions may have changed)
    tc_puf = create_tc_puf(taxyear)
    tc_cps, nonfiler = create_tc_cps(taxyear)
    tc_cps = tc_cps[nonfiler].reset_index(drop=True)

    # scale CPS weights to get sensible combined population count
    tc_cps["s006"] *= CPS_WEIGHTS_SCALE[TAXYEAR]

    print("Combining PUF filers and CPS nonfilers...")
    combined = pd.concat([tc_puf, tc_cps], ignore_index=True)

    # ensure RECID values are unique
    combined["RECID"] = np.arange(1, len(combined) + 1, dtype=int)

    print(f"Adding Tax-Calculator outputs for {taxyear}...")
    combined = add_taxcalc_outputs(combined, taxyear, taxyear)
    # ... drop CPS records with positive income tax amount
    idx = combined[((combined.data_source == 0) & (combined.iitax > 0))].index
    combined.drop(idx, inplace=True)

    print("Reweighting...")
    combined["s006_original"] = combined["s006"].values
    combined = reweight(combined, taxyear)

    combined = combined.reindex(sorted(combined.columns), axis=1)

    return combined
