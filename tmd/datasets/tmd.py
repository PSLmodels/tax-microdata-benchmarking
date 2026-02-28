import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from tmd.imputation_assumptions import CPS_WEIGHTS_SCALE
from tmd.datasets.puf import PUF_2021, create_pe_puf_2021
from tmd.datasets.cps import CPS_2021, create_cps_2021
from tmd.datasets.taxcalc_dataset import create_tc_dataset
from tmd.utils.trace import trace1
from tmd.utils.taxcalc_utils import add_taxcalc_outputs


def create_tmd_2021():
    # always create CPS_2021 and PUF_2021
    # (because imputation assumptions may have changed)
    create_cps_2021()
    create_pe_puf_2021()

    tc_puf_21 = create_tc_dataset(PUF_2021, 2021)
    tc_cps_21 = create_tc_dataset(CPS_2021, 2021)

    # identify CPS nonfilers using 2022 filing rules
    # (because 2021 had large COVID-related anomalies)
    sim = Microsimulation(dataset=CPS_2021)
    nonfiler = ~(sim.calculate("tax_unit_is_filer", period=2022).values > 0)
    tc_cps_21 = tc_cps_21[nonfiler]

    print("Combining PUF filers and CPS nonfilers...")
    combined = pd.concat([tc_puf_21, tc_cps_21], ignore_index=True)

    # ensure RECID values are unique
    combined["RECID"] = np.arange(1, len(combined) + 1, dtype=int)

    trace1("A", combined)

    print("Adding Tax-Calculator outputs for 2021...")
    combined = add_taxcalc_outputs(combined, 2021, 2021)
    # ... drop CPS records with positive 2021 income tax amount
    idx = combined[((combined.data_source == 0) & (combined.iitax > 0))].index
    combined.drop(idx, inplace=True)
    # ... scale CPS records weight to get correct population count
    scale = np.where(combined.data_source == 0, CPS_WEIGHTS_SCALE, 1.0)
    combined["s006"] *= scale

    trace1("B", combined)

    print("Reweighting...")
    combined["s006_original"] = combined["s006"].values
    # Solver selection via environment variables:
    #   default:            Clarabel QP (constrained, reproducible)
    #   PYTORCH_REWEIGHT=1: PyTorch L-BFGS (original penalty-based)
    #   SCIPY_REWEIGHT=1:   scipy L-BFGS-B (penalty-based backup)
    use_pytorch = os.environ.get("PYTORCH_REWEIGHT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    use_scipy = os.environ.get("SCIPY_REWEIGHT", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_pytorch:
        reweight_import = "from tmd.utils.reweight import reweight"
        reweight_call = "reweight(df, 2021)"
        print("...using penalty-based solver (PyTorch L-BFGS)")
    elif use_scipy:
        reweight_import = "from tmd.utils.reweight import reweight_lbfgsb"
        reweight_call = "reweight_lbfgsb(df, 2021)"
        print("...using penalty-based solver (scipy L-BFGS-B)")
    else:
        reweight_import = (
            "from tmd.utils.reweight_clarabel import reweight_clarabel"
        )
        reweight_call = "reweight_clarabel(df, 2021)"
        print("...using constrained QP solver (Clarabel)")
    # Run reweighting in a subprocess so that prior PyTorch
    # operations (PolicyEngine Microsimulation) don't affect
    # the optimizer state.
    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_path = f"{tmpdir}/snapshot.csv.gz"
        result_path = f"{tmpdir}/result.csv.gz"
        combined.to_csv(snapshot_path, index=False)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import pandas as pd; "
                "import sys; sys.path.insert(0, '.'); "
                f"{reweight_import}; "
                f"df = pd.read_csv('{snapshot_path}'); "
                f"df = {reweight_call}; "
                f"df[['RECID','s006']].to_csv("
                f"'{result_path}', index=False)",
            ],
            check=True,
        )
        reweighted = pd.read_csv(result_path)
    combined["s006"] = combined.merge(
        reweighted, on="RECID", suffixes=("_old", "")
    )["s006"].values

    trace1("C", combined)

    combined = combined.reindex(sorted(combined.columns), axis=1)

    return combined


if __name__ == "__main__":
    tmd = create_tmd_2021()
