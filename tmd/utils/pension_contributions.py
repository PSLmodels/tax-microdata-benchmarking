import numpy as np
from policyengine_us import Microsimulation
from tmd.datasets.cps import CPS_2021
from tmd.utils.imputation import Imputation
from tmd.imputation_assumptions import (
    IMPUTATION_RF_RNG_SEED,
    IMPUTATION_BETA_RNG_SEED,
)


def impute_pretax_pension_contributions(puf_df):

    cps = Microsimulation(dataset=CPS_2021)
    cps_df = cps.calculate_dataframe(
        [
            "employment_income",
            "household_weight",
            "traditional_401k_contributions",
            "traditional_403b_contributions",
        ]
    )
    cps_df["pretax_pension_contributions"] = (
        cps_df["traditional_401k_contributions"]
        + cps_df["traditional_403b_contributions"]
    )

    pension_contributions = Imputation()
    pension_contributions.rf_rng_seed = IMPUTATION_RF_RNG_SEED
    pension_contributions.beta_rng_seed = IMPUTATION_BETA_RNG_SEED

    pension_contributions.train(
        X=cps_df[["employment_income"]],
        Y=cps_df[["pretax_pension_contributions"]],
        sample_weight=cps_df["household_weight"],
    )
    return pension_contributions.predict(
        X=puf_df[["employment_income"]],
    )
