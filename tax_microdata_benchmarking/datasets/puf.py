import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from microdf import MicroDataFrame
from policyengine_us.system import system
from policyengine_core.data import Dataset
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.pension_contributions import (
    impute_pension_contributions_to_puf,
)
from tax_microdata_benchmarking.datasets.uprate_puf import uprate_puf
from tax_microdata_benchmarking.utils.imputation import Imputation
from tax_microdata_benchmarking.imputation_assumptions import (
    IMPUTATION_RF_RNG_SEED,
    IMPUTATION_BETA_RNG_SEED,
    W2_WAGES_SCALE,
)


FILER_AGE_RNG = np.random.default_rng(seed=64963751)
SPOUSE_GENDER_RNG = np.random.default_rng(seed=83746519)
DEP_AGE_RNG = np.random.default_rng(seed=24354657)
DEP_GENDER_RNG = np.random.default_rng(seed=74382916)
EARN_SPLIT_RNG = np.random.default_rng(seed=18374659)


def impute_missing_demographics(
    puf: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    puf_with_demographics = (
        puf[puf.RECID.isin(demographics.RECID)]
        .merge(demographics, on="RECID")
        .fillna(0)
    )

    DEMOGRAPHIC_VARIABLES = [
        "AGEDP1",
        "AGEDP2",
        "AGEDP3",
        "AGERANGE",
        "EARNSPLIT",
        "GENDER",
    ]
    NON_DEMOGRAPHIC_VARIABLES = [
        "E00200",
        "MARS",
        "DSI",
        "EIC",
        "XTOT",
    ]

    demographics_from_puf = Imputation()
    demographics_from_puf.rf_rng_seed = IMPUTATION_RF_RNG_SEED
    demographics_from_puf.beta_rng_seed = IMPUTATION_BETA_RNG_SEED

    demographics_from_puf.train(
        X=puf_with_demographics[NON_DEMOGRAPHIC_VARIABLES],
        Y=puf_with_demographics[DEMOGRAPHIC_VARIABLES],
    )

    puf_without_demographics = puf[
        ~puf.RECID.isin(puf_with_demographics.RECID)
    ].reset_index()
    predicted_demographics = demographics_from_puf.predict(
        X=puf_without_demographics,
    )
    puf_with_imputed_demographics = pd.concat(
        [puf_without_demographics, predicted_demographics], axis=1
    )

    weighted_puf_with_demographics = MicroDataFrame(
        puf_with_demographics, weights="S006"
    )
    weighted_puf_with_imputed_demographics = MicroDataFrame(
        puf_with_imputed_demographics, weights="S006"
    )

    puf_combined = pd.concat(
        [
            weighted_puf_with_demographics,
            weighted_puf_with_imputed_demographics,
        ]
    )

    return puf_combined


def decode_age_filer(age_range: int) -> int:
    if age_range == 0:
        return 40
    AGERANGE_FILER_DECODE = {
        1: 18,
        2: 26,
        3: 35,
        4: 45,
        5: 55,
        6: 65,
        7: 80,
    }
    lower = AGERANGE_FILER_DECODE[age_range]
    upper = AGERANGE_FILER_DECODE[age_range + 1]
    return FILER_AGE_RNG.integers(low=lower, high=upper, endpoint=False)


def decode_age_dependent(age_range: int) -> int:
    if age_range == 0:
        return 0
    AGERANGE_DEPENDENT_DECODE = {
        0: 0,
        1: 0,
        2: 5,
        3: 13,
        4: 17,
        5: 19,
        6: 25,
        7: 30,
    }
    lower = AGERANGE_DEPENDENT_DECODE[age_range]
    upper = AGERANGE_DEPENDENT_DECODE[age_range + 1]
    return DEP_AGE_RNG.integers(low=lower, high=upper, endpoint=False)


def preprocess_puf(puf: pd.DataFrame) -> pd.DataFrame:
    # rescale weights
    puf.S006 = puf.S006 / 100
    # remove aggregate records    ????????? ALREADY DONE ????????????
    # puf = puf[puf.MARS != 0]  # ????????? ALREADY DONE ????????????
    filing_status = puf.MARS.map(
        {
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    newvars = {
        "household_id": puf.RECID,
        "household_weight": puf.S006,
        "filing_status": filing_status,
        "exemptions_count": puf.XTOT,
    }
    newdf = pd.DataFrame(newvars)

    # add new renamed variables
    puf = pd.concat([puf, newdf], axis=1)

    # puf["adjusted_gross_income"] = puf.E00100
    puf["alimony_expense"] = puf.E03500
    puf["alimony_income"] = puf.E00800
    puf["casualty_loss"] = puf.E20500
    puf["cdcc_relevant_expenses"] = puf.E32800
    puf["charitable_cash_donations"] = puf.E19800
    puf["charitable_non_cash_donations"] = puf.E20100
    puf["domestic_production_ald"] = puf.E03240
    puf["early_withdrawal_penalty"] = puf.E03400
    puf["educator_expense"] = puf.E03220
    puf["employment_income"] = puf.E00200
    puf["estate_income"] = puf.E26390 - puf.E26400
    puf["farm_income"] = puf.T27800
    puf["farm_rent_income"] = puf.E27200
    puf["health_savings_account_ald"] = puf.E03290
    puf["interest_deduction"] = puf.E19200
    puf["long_term_capital_gains"] = puf.P23250
    puf["long_term_capital_gains_on_collectibles"] = puf.E24518
    puf["medical_expense"] = puf.E17500
    puf["misc_deduction"] = puf.E20400
    puf["non_qualified_dividend_income"] = puf.E00600 - puf.E00650
    puf["partnership_s_corp_income"] = puf.E26270
    puf["qualified_dividend_income"] = puf.E00650
    puf["qualified_tuition_expenses"] = puf.E03230
    puf["real_estate_taxes"] = puf.E18500
    puf["rental_income"] = puf.E25850 - puf.E25860
    puf["self_employment_income"] = puf.E00900
    puf["self_employed_health_insurance_ald"] = puf.E03270
    puf["self_employed_pension_contribution_ald"] = puf.E03300
    puf["short_term_capital_gains"] = puf.P22250
    puf["social_security"] = puf.E02400
    puf["state_and_local_sales_or_income_tax"] = puf.E18400
    puf["student_loan_interest"] = puf.E03210
    puf["taxable_interest_income"] = puf.E00300
    puf["taxable_pension_income"] = puf.E01700
    puf["taxable_unemployment_compensation"] = puf.E02300
    puf["taxable_ira_distributions"] = puf.E01400
    puf["tax_exempt_interest_income"] = puf.E00400
    puf["tax_exempt_pension_income"] = puf.E01500 - puf.E01700
    puf["traditional_ira_contributions"] = puf.E03150
    puf["unrecaptured_section_1250_gain"] = puf.E24515
    puf["foreign_tax_credit"] = puf.E07300
    puf["amt_foreign_tax_credit"] = puf.E62900
    puf["miscellaneous_income"] = puf.E01200
    puf["salt_refund_income"] = puf.E00700
    puf["investment_income_elected_form_4952"] = puf.E58990
    puf["general_business_credit"] = puf.E07400
    puf["prior_year_minimum_tax_credit"] = puf.E07600
    puf["excess_withheld_payroll_tax"] = puf.E11200
    puf["non_sch_d_capital_gains"] = puf.E01100
    puf["american_opportunity_credit"] = puf.E87521
    puf["energy_efficient_home_improvement_credit"] = puf.E07260
    puf["early_withdrawal_penalty"] = puf.E09900
    # puf["qualified_tuition_expenses"] = puf.E87530 # PE uses the same variable for qualified tuition (general) and qualified tuition (Lifetime Learning Credit). Revisit here.
    puf["other_credits"] = puf.P08000
    puf["savers_credit"] = puf.E07240
    puf["recapture_of_investment_credit"] = puf.E09700
    puf["unreported_payroll_tax"] = puf.E09800
    # Ignore f2441 (AMT form attached)
    # Ignore cmbtp (estimate of AMT income not in AGI)
    # Ignore k1bx14s and k1bx14p (partner self-employment income included in partnership and S-corp income)
    qbi = np.maximum(0, puf.E00900 + puf.E26270 + puf.E02100 + puf.E27200)
    puf["w2_wages_from_qualified_business"] = qbi * W2_WAGES_SCALE

    return puf


FINANCIAL_SUBSET = [
    # "adjusted_gross_income",
    "alimony_expense",
    "alimony_income",
    "casualty_loss",
    "cdcc_relevant_expenses",
    "charitable_cash_donations",
    "charitable_non_cash_donations",
    "domestic_production_ald",
    "early_withdrawal_penalty",
    "educator_expense",
    "employment_income",
    "estate_income",
    "farm_income",
    "farm_rent_income",
    "health_savings_account_ald",
    "interest_deduction",
    "long_term_capital_gains",
    "long_term_capital_gains_on_collectibles",
    "medical_expense",
    "misc_deduction",
    "non_qualified_dividend_income",
    "non_sch_d_capital_gains",
    "partnership_s_corp_income",
    "qualified_dividend_income",
    "qualified_tuition_expenses",
    "real_estate_taxes",
    "rental_income",
    "self_employment_income",
    "self_employed_health_insurance_ald",
    "self_employed_pension_contribution_ald",
    "short_term_capital_gains",
    "social_security",
    "state_and_local_sales_or_income_tax",
    "student_loan_interest",
    "taxable_interest_income",
    "taxable_pension_income",
    "taxable_unemployment_compensation",
    "taxable_ira_distributions",
    "tax_exempt_interest_income",
    "tax_exempt_pension_income",
    "traditional_ira_contributions",
    "unrecaptured_section_1250_gain",
    "foreign_tax_credit",
    "amt_foreign_tax_credit",
    "miscellaneous_income",
    "salt_refund_income",
    "investment_income_elected_form_4952",
    "general_business_credit",
    "prior_year_minimum_tax_credit",
    "excess_withheld_payroll_tax",
    "american_opportunity_credit",
    "energy_efficient_home_improvement_credit",
    "other_credits",
    "savers_credit",
    "recapture_of_investment_credit",
    "unreported_payroll_tax",
    "pre_tax_contributions",
    "w2_wages_from_qualified_business",
]


class PUF(Dataset):
    time_period = None
    data_format = Dataset.ARRAYS

    def generate(self, puf: pd.DataFrame, demographics: pd.DataFrame):
        print("Importing PolicyEngine US variable metadata...")

        itmded_dump = False
        if itmded_dump:
            IDVARS = ["E17500", "E18400", "E18500", "E19200", "E19800"]
            wght = puf.S006 / 100.0
            for var in IDVARS:
                print(f"%%15:{var}= {(puf[var]*wght).sum()*1e-9:.3f}")

        if self.time_period > 2015:
            puf = uprate_puf(puf, 2015, self.time_period)

        if itmded_dump:
            wght = puf.S006 / 100.0
            for var in IDVARS:
                print(f"%%21:{var}= {(puf[var]*wght).sum()*1e-9:.3f}")

        puf = puf[puf.MARS != 0]

        print("Pre-processing PUF...")
        original_recid = puf.RECID.values.copy()
        puf = preprocess_puf(puf)
        print("Imputing missing PUF demographics...")
        puf = impute_missing_demographics(puf, demographics)
        print("Imputing PUF pension contributions...")
        puf["pre_tax_contributions"] = impute_pension_contributions_to_puf(
            puf[["employment_income"]]
        )

        # Sort in original PUF order
        puf = puf.set_index("RECID").loc[original_recid].reset_index()
        puf = puf.fillna(0)
        self.variable_to_entity = {
            variable: system.variables[variable].entity.key
            for variable in system.variables
        }

        VARIABLES = [
            "person_id",
            "tax_unit_id",
            "marital_unit_id",
            "spm_unit_id",
            "family_id",
            "household_id",
            "person_tax_unit_id",
            "person_marital_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_household_id",
            "age",
            "household_weight",
            "is_male",
            "filing_status",
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ] + FINANCIAL_SUBSET

        self.holder = {variable: [] for variable in VARIABLES}

        i = 0
        self.earn_splits = []
        for _, row in tqdm(
            puf.iterrows(),
            total=len(puf),
            desc="Constructing hierarchical PUF",
        ):
            i += 1
            exemptions = row["exemptions_count"]
            tax_unit_id = row["household_id"]
            self.add_tax_unit(row, tax_unit_id)
            self.add_filer(row, tax_unit_id)
            exemptions -= 1
            if row["filing_status"] == "JOINT":
                self.add_spouse(row, tax_unit_id)
                exemptions -= 1

            for j in range(min(3, exemptions)):
                self.add_dependent(row, tax_unit_id, j)

        groups_assumed_to_be_tax_unit_like = [
            "family",
            "spm_unit",
            "household",
        ]

        for group in groups_assumed_to_be_tax_unit_like:
            self.holder[f"{group}_id"] = self.holder["tax_unit_id"]
            self.holder[f"person_{group}_id"] = self.holder[
                "person_tax_unit_id"
            ]

        for key in self.holder:
            if key == "filing_status":
                self.holder[key] = np.array(self.holder[key]).astype("S")
            else:
                self.holder[key] = np.array(self.holder[key]).astype(float)
                assert not np.isnan(self.holder[key]).any(), f"{key} has NaNs."

        self.save_dataset(self.holder)

    def add_tax_unit(self, row, tax_unit_id):
        self.holder["tax_unit_id"].append(tax_unit_id)

        for key in FINANCIAL_SUBSET:
            if self.variable_to_entity[key] == "tax_unit":
                self.holder[key].append(row[key])

        earnings_split = round(row["EARNSPLIT"])
        if earnings_split > 0:
            SPLIT_DECODES = {
                1: 0.0,
                2: 0.25,
                3: 0.75,
                4: 1.0,
            }
            lower = SPLIT_DECODES[earnings_split]
            upper = SPLIT_DECODES[earnings_split + 1]
            frac = (upper - lower) * EARN_SPLIT_RNG.random() + lower
            self.earn_splits.append(1.0 - frac)
        else:
            self.earn_splits.append(1.0)

        self.holder["filing_status"].append(row["filing_status"])

    def add_filer(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 1)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(True)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(decode_age_filer(round(row["AGERANGE"])))

        self.holder["household_weight"].append(row["household_weight"])
        self.holder["is_male"].append(row["GENDER"] == 1)

        for key in FINANCIAL_SUBSET:
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * self.earn_splits[-1])

    def add_spouse(self, row, tax_unit_id):
        person_id = int(tax_unit_id * 1e2 + 2)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id - 1)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(True)
        self.holder["is_tax_unit_dependent"].append(False)

        self.holder["age"].append(
            decode_age_filer(round(row["AGERANGE"]))
        )  # Assume same age as filer for now

        # 96% of joint filers are opposite-gender

        is_opposite_gender = SPOUSE_GENDER_RNG.random() < 0.96
        opposite_gender_code = 0 if row["GENDER"] == 1 else 1
        same_gender_code = 1 - opposite_gender_code
        self.holder["is_male"].append(
            opposite_gender_code if is_opposite_gender else same_gender_code
        )

        for key in FINANCIAL_SUBSET:
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(row[key] * (1 - self.earn_splits[-1]))

    def add_dependent(self, row, tax_unit_id, dependent_id):
        person_id = int(tax_unit_id * 1e2 + 3 + dependent_id)
        self.holder["person_id"].append(person_id)
        self.holder["person_tax_unit_id"].append(tax_unit_id)
        self.holder["person_marital_unit_id"].append(person_id)
        self.holder["marital_unit_id"].append(person_id)
        self.holder["is_tax_unit_head"].append(False)
        self.holder["is_tax_unit_spouse"].append(False)
        self.holder["is_tax_unit_dependent"].append(True)

        age = decode_age_dependent(round(row[f"AGEDP{dependent_id + 1}"]))
        self.holder["age"].append(age)

        for key in FINANCIAL_SUBSET:
            if self.variable_to_entity[key] == "person":
                self.holder[key].append(0)

        self.holder["is_male"].append(DEP_GENDER_RNG.choice([0, 1]))


class PUF_2015(PUF):
    label = "PUF 2015"
    name = "puf_2015"
    time_period = 2015
    file_path = STORAGE_FOLDER / "output" / "pe_puf_2015.h5"


class PUF_2021(PUF):
    label = "PUF 2021"
    name = "puf_2021"
    time_period = 2021
    file_path = STORAGE_FOLDER / "output" / "pe_puf_2021.h5"


def create_pe_puf_2015():
    puf = pd.read_csv(STORAGE_FOLDER / "input" / "puf_2015.csv")
    demographics = pd.read_csv(
        STORAGE_FOLDER / "input" / "demographics_2015.csv"
    )

    pe_puf = PUF_2015()
    pe_puf.generate(puf, demographics)


def create_pe_puf_2021():
    puf = pd.read_csv(STORAGE_FOLDER / "input" / "puf_2015.csv")
    demographics = pd.read_csv(
        STORAGE_FOLDER / "input" / "demographics_2015.csv"
    )

    pe_puf = PUF_2021()
    pe_puf.generate(puf, demographics)


if __name__ == "__main__":
    create_pe_puf_2015()
    create_pe_puf_2021()
