from pathlib import Path

STORAGE_FOLDER = Path(__file__).parent

# Growth factors used to index Policy parameters (as distinct from
# tmd_growfactors.csv, which extrapolates Records variables).  This file
# is created in the first stage of the `make tmd_files` pipeline.
POLICY_GROWFACTORS_PATH = STORAGE_FOLDER / "output" / "growfactors.csv"

CACHED_TAXCALC_VARIABLES = [
    "c00100",  # AGI
    "iitax",  # individual income tax liability (including refundable credits)
]
