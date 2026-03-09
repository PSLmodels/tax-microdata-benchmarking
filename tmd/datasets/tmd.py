import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from tmd.imputation_assumptions import CPS_WEIGHTS_SCALE
from tmd.datasets.puf import create_tc_puf
from tmd.datasets.cps import create_tc_cps
from tmd.utils.trace import trace1
from tmd.utils.taxcalc_utils import add_taxcalc_outputs


def create_tmd_dataframe(taxyear: int) -> pd.DataFrame:
    """
    Create DataFrame for given taxyear containing PUF filers and CPS nonfilers.
    """
    # always create_tc_puf and create_tc_cps
    # (because imputation assumptions may have changed)
    tc_puf = create_tc_puf(taxyear)
    tc_cps, nonfiler = create_tc_cps(taxyear)
    tc_cps = tc_cps[nonfiler].reset_index(drop=True)

    print("Combining PUF filers and CPS nonfilers...")
    combined = pd.concat([tc_puf, tc_cps], ignore_index=True)

    # ensure RECID values are unique
    combined["RECID"] = np.arange(1, len(combined) + 1, dtype=int)

    trace1("A", combined)

    print(f"Adding Tax-Calculator outputs for {taxyear}...")
    combined = add_taxcalc_outputs(combined, taxyear, taxyear)
    # ... drop CPS records with positive income tax amount
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
        reweight_call = f"reweight(df, {taxyear})"
        print("...using penalty-based solver (PyTorch L-BFGS)")
    elif use_scipy:
        reweight_import = "from tmd.utils.reweight import reweight_lbfgsb"
        reweight_call = f"reweight_lbfgsb(df, {taxyear})"
        print("...using penalty-based solver (scipy L-BFGS-B)")
    else:
        reweight_import = (
            "from tmd.utils.reweight_clarabel import reweight_clarabel"
        )
        reweight_call = f"reweight_clarabel(df, {taxyear})"
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
