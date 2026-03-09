import numpy as np
import pandas as pd
from tmd.datasets.cps import _load_raw_person
from tmd.utils.imputation import Imputation
from tmd.imputation_assumptions import (
    IMPUTATION_RF_RNG_SEED,
    IMPUTATION_BETA_RNG_SEED,
)

# IRS 401(k) elective deferral limits and catch-up limits by year
_401K_LIMITS = {
    2020: (19_500, 6_500),
    2021: (19_500, 6_500),
    2022: (20_500, 6_500),
    2023: (22_500, 7_500),
    2024: (23_000, 7_500),
    2025: (23_500, 7_500),
}
_CATCH_UP_AGE = 50


def impute_pension_contributions(
    taxyear: int, puf_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute pension contributions onto PUF records using
    CPS person-level data from the cached raw CPS HDF5 file.
    """
    person = _load_raw_person(taxyear)

    # get year-specific 401(k) limits
    if taxyear not in _401K_LIMITS:
        raise ValueError(
            f"No 401(k) limits defined for taxyear {taxyear}. "
            f"Add limits to _401K_LIMITS in pension_contributions.py."
        )
    limit_401k_base, limit_401k_catchup = _401K_LIMITS[taxyear]

    # person-level variables from raw CPS
    employment_income = person["WSAL_VAL"].values.astype(float)
    self_employment_income = person["SEMP_VAL"].values.astype(float)
    retcb_val = person["RETCB_VAL"].values.astype(float)
    age = person["A_AGE"].values
    weight = person["A_FNLWGT"].values / 1e2

    # allocate RETCB_VAL to traditional 401(k) contributions
    # (same logic as in cps.py: SE pension first, then 401k up to limit)
    se_pension = np.where(self_employment_income > 0, retcb_val, 0)
    remaining = np.maximum(retcb_val - se_pension, 0)
    catch_up_eligible = age >= _CATCH_UP_AGE
    limit_401k = limit_401k_base + catch_up_eligible * limit_401k_catchup
    trad_401k = np.where(
        employment_income > 0,
        np.minimum(remaining, limit_401k),
        0,
    )

    # build CPS DataFrame for imputation training
    cps_df = pd.DataFrame(
        {
            "employment_income": employment_income,
            "pension_contributions": trad_401k,
            "weight": weight,
        }
    )

    # train and predict
    pension_contributions = Imputation()
    pension_contributions.rf_rng_seed = IMPUTATION_RF_RNG_SEED
    pension_contributions.beta_rng_seed = IMPUTATION_BETA_RNG_SEED
    pension_contributions.train(
        X=cps_df[["employment_income"]],
        Y=cps_df[["pension_contributions"]],
        sample_weight=cps_df["weight"],
    )
    return pension_contributions.predict(
        X=puf_df[["employment_income"]],
    )
