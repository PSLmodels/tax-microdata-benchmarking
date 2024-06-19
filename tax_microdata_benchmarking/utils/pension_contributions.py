from survey_enhance import Imputation
from policyengine_us import Microsimulation
from tax_microdata_benchmarking.datasets.cps import CPS_2021


def impute_pension_contributions_to_puf(puf_df):

    cps = Microsimulation(dataset=CPS_2021)
    cps_df = cps.calculate_dataframe(
        ["employment_income", "household_weight", "pre_tax_contributions"]
    )

    pension_contributions = Imputation()
    pension_contributions.train(
        X=cps_df[["employment_income"]],
        Y=cps_df[["pre_tax_contributions"]],
        sample_weight=cps_df["household_weight"],
    )
    return pension_contributions.predict(X=puf_df[["employment_income"]])
