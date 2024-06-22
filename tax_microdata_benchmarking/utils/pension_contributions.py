import numpy as np
from survey_enhance import Imputation
from policyengine_us import Microsimulation
from tax_microdata_benchmarking.datasets.cps import CPS_2021
from tax_microdata_benchmarking.imputation_assumptions import (
    TRAIN_RNG_SEED,
    PREDICT_RNG_SEED,
)

def impute_pension_contributions_to_puf(puf_df):

    cps = Microsimulation(dataset=CPS_2021)
    cps_df = cps.calculate_dataframe(
        ["employment_income", "household_weight", "pre_tax_contributions"]
    )

    pension_contributions = Imputation()
    pension_contributions.random_generator=np.random.default_rng(
        PREDICT_RNG_SEED,
    )
    pension_contributions.train(
        X=cps_df[["employment_income"]],
        Y=cps_df[["pre_tax_contributions"]],
        sample_weight=cps_df["household_weight"],
    )
    return pension_contributions.predict(
        X=puf_df[["employment_income"]],
    )
