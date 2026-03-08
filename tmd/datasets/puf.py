import yaml
import numpy as np
import pandas as pd
from microdf import MicroDataFrame
from tmd.storage import STORAGE_FOLDER
from tmd.datasets.uprate_puf import uprate_puf
from tmd.utils.imputation import Imputation
from tmd.utils.pension_contributions import impute_pretax_pension_contributions
from tmd.imputation_assumptions import (
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
    filing_status = puf.MARS.map(
        {
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    qbi = np.maximum(0, puf.E00900 + puf.E26270 + puf.E02100 + puf.E27200)
    newvars = {
        "household_id": puf.RECID,
        "household_weight": puf.S006,
        "filing_status": filing_status,
        "exemptions_count": puf.XTOT,
        "alimony_expense": puf.E03500,
        "alimony_income": puf.E00800,
        "casualty_loss": puf.E20500,
        "cdcc_relevant_expenses": puf.E32800,
        "charitable_cash_donations": puf.E19800,
        "charitable_non_cash_donations": puf.E20100,
        "domestic_production_ald": puf.E03240,
        "early_withdrawal_penalty": puf.E03400,
        "educator_expense": puf.E03220,
        "employment_income": puf.E00200,
        "estate_income": (puf.E26390 - puf.E26400),
        "farm_income": puf.T27800,
        "farm_rent_income": puf.E27200,
        "health_savings_account_ald": puf.E03290,
        "interest_deduction": puf.E19200,
        "long_term_capital_gains": puf.P23250,
        "long_term_capital_gains_on_collectibles": puf.E24518,
        "medical_expense": puf.E17500,
        "misc_deduction": puf.E20400,
        "non_qualified_dividend_income": (puf.E00600 - puf.E00650),
        "partnership_s_corp_income": puf.E26270,
        "qualified_dividend_income": puf.E00650,
        "qualified_tuition_expenses": puf.E03230,
        "real_estate_taxes": puf.E18500,
        "rental_income": (puf.E25850 - puf.E25860),
        "self_employment_income": puf.E00900,
        "self_employed_health_insurance_ald": puf.E03270,
        "self_employed_pension_contribution_ald": puf.E03300,
        "short_term_capital_gains": puf.P22250,
        "social_security": puf.E02400,
        "state_and_local_sales_or_income_tax": puf.E18400,
        "student_loan_interest": puf.E03210,
        "taxable_interest_income": puf.E00300,
        "taxable_pension_income": puf.E01700,
        "taxable_unemployment_compensation": puf.E02300,
        "taxable_ira_distributions": puf.E01400,
        "tax_exempt_interest_income": puf.E00400,
        "tax_exempt_pension_income": (puf.E01500 - puf.E01700),
        "traditional_ira_contributions": puf.E03150,
        "unrecaptured_section_1250_gain": puf.E24515,
        "foreign_tax_credit": puf.E07300,
        "amt_foreign_tax_credit": puf.E62900,
        "miscellaneous_income": puf.E01200,
        "salt_refund_income": puf.E00700,
        "investment_income_elected_form_4952": puf.E58990,
        "general_business_credit": puf.E07400,
        "prior_year_minimum_tax_credit": puf.E07600,
        "excess_withheld_payroll_tax": puf.E11200,
        "non_sch_d_capital_gains": puf.E01100,
        "american_opportunity_credit": puf.E87521,
        "energy_efficient_home_improvement_credit": puf.E07260,
        "qualified_retirement_penalty": puf.E09900,
        # "qualified_tuition_expenses": puf.E87530,
        # PE uses the same variable for qualified tuition (general)
        # and qualified tuition (Lifetime Learning Credit). Revisit this.
        "other_credits": puf.P08000,
        "savers_credit": puf.E07240,
        "recapture_of_investment_credit": puf.E09700,
        "unreported_payroll_tax": puf.E09800,
        # Ignore f2441 (CDCC form attached)
        # Ignore cmbtp (estimate of AMT income not in AGI)
        # Ignore k1bx14s and k1bx14p (partner self-employment income included
        #                             in partnership and S-corp income)
        # "adjusted_gross_income": puf.E00100,
        "w2_wages_from_qualified_business": (qbi * W2_WAGES_SCALE),
    }
    newdf = pd.DataFrame(newvars)
    # add new renamed variables to original puf dataframe
    puf = pd.concat([puf, newdf], axis=1)
    return puf


def create_tc_puf(taxyear: int) -> pd.DataFrame:
    """
    Create a Tax-Calculator-compatible PUF DataFrame for the given taxyear
    directly from raw PUF data, without using PolicyEngine (PE) Dataset or
    hierarchical data files.
    """
    # fresh RNG objects (same seeds as module-level RNGs used in PUF class)
    filer_age_rng_head = np.random.default_rng(seed=64963751)
    filer_age_rng_spouse = np.random.default_rng(seed=64963753)
    dep_age_rng = np.random.default_rng(seed=24354657)
    earn_split_rng = np.random.default_rng(seed=18374659)

    # read and prepare raw PUF data
    print("Reading raw PUF 2015 data...")
    puf = pd.read_csv(STORAGE_FOLDER / "input" / "puf_2015.csv")
    demographics = pd.read_csv(
        STORAGE_FOLDER / "input" / "demographics_2015.csv"
    )
    if taxyear > 2015:
        puf = uprate_puf(puf, 2015, taxyear)

    # remove aggregate records
    puf = puf[puf.MARS != 0].copy()

    # save raw PUF variables before preprocessing renames columns
    eic_raw = np.minimum(puf["EIC"].values.copy(), 3)
    f2441_raw = puf["F2441"].values.copy()
    mars_raw = puf["MARS"].values.copy()

    print("Pre-processing PUF...")
    original_recid = puf.RECID.values.copy()
    puf = preprocess_puf(puf)

    print("Imputing missing PUF demographics...")
    puf = impute_missing_demographics(puf, demographics)

    # sort in original PUF order and fill NaN
    puf = puf.set_index("RECID").loc[original_recid].reset_index()
    puf = puf.fillna(0)

    n = len(puf)
    is_joint = mars_raw == 2

    # compute earnings splits (vectorized)
    print("Computing earnings splits...")
    earnsplit = np.round(puf["EARNSPLIT"].values).astype(int)
    head_frac = np.ones(n)
    split_mask = (earnsplit >= 1) & (earnsplit <= 3)
    if split_mask.any():
        SPLIT_BOUNDS = {1: (0.0, 0.25), 2: (0.25, 0.75), 3: (0.75, 1.0)}
        rand_vals = earn_split_rng.random(split_mask.sum())
        lowers = np.array([SPLIT_BOUNDS[v][0] for v in earnsplit[split_mask]])
        uppers = np.array([SPLIT_BOUNDS[v][1] for v in earnsplit[split_mask]])
        fracs = (uppers - lowers) * rand_vals + lowers
        head_frac[split_mask] = 1.0 - fracs

    # for person-level variables, the tax-unit total equals
    # head + spouse = value * head_frac + value * (1 - head_frac) * is_joint
    person_scale = head_frac + (1.0 - head_frac) * is_joint

    # decode demographic ages (vectorized)
    def _decode_filer_ages(agerange_vals, rng):
        DECODE = {1: 18, 2: 26, 3: 35, 4: 45, 5: 55, 6: 65, 7: 80}
        ages = np.full(len(agerange_vals), 40, dtype=int)
        for ar in range(1, 7):
            mask = agerange_vals == ar
            cnt = mask.sum()
            if cnt > 0:
                ages[mask] = rng.integers(DECODE[ar], DECODE[ar + 1], size=cnt)
        return ages

    def _decode_dep_ages(agerange_vals, rng):
        DECODE = {0: 0, 1: 0, 2: 5, 3: 13, 4: 17, 5: 19, 6: 25, 7: 30}
        ages = np.zeros(len(agerange_vals), dtype=int)
        for ar in range(1, 7):
            mask = agerange_vals == ar
            cnt = mask.sum()
            if cnt > 0:
                ages[mask] = rng.integers(DECODE[ar], DECODE[ar + 1], size=cnt)
        return ages

    agerange = np.round(puf["AGERANGE"].values).astype(int)
    age_head = _decode_filer_ages(agerange, filer_age_rng_head)
    age_spouse_all = _decode_filer_ages(agerange, filer_age_rng_spouse)
    age_spouse = np.where(is_joint, age_spouse_all, 0)

    # dependent ages and existence flags
    exemptions = np.round(puf["exemptions_count"].values).astype(int)
    n_deps = np.clip(exemptions - 1 - is_joint.astype(int), 0, 3)
    dep_ages = []
    dep_exists = []
    for j in range(3):
        exists_j = j < n_deps
        dep_exists.append(exists_j)
        dep_agerange = np.round(puf[f"AGEDP{j + 1}"].values).astype(int)
        dep_ages.append(_decode_dep_ages(dep_agerange, dep_age_rng))

    # demographic counts from dependent ages
    nu18 = sum((dep_ages[j] < 18) * dep_exists[j] for j in range(3))
    nu13 = sum((dep_ages[j] < 13) * dep_exists[j] for j in range(3))
    nu06 = sum((dep_ages[j] < 6) * dep_exists[j] for j in range(3))
    n1820 = sum(
        ((dep_ages[j] >= 18) & (dep_ages[j] < 21)) * dep_exists[j]
        for j in range(3)
    )
    n21 = sum((dep_ages[j] >= 21) * dep_exists[j] for j in range(3))
    n24 = sum((dep_ages[j] < 17) * dep_exists[j] for j in range(3))
    elderly_dependents = sum(
        (dep_ages[j] >= 65) * dep_exists[j] for j in range(3)
    )

    # head/spouse income splits
    emp_income = puf["employment_income"].values
    se_income = puf["self_employment_income"].values
    farm_inc = puf["farm_income"].values
    e00200p = emp_income * head_frac
    e00200s = emp_income * (1.0 - head_frac) * is_joint
    e00900p = se_income * head_frac
    e00900s = se_income * (1.0 - head_frac) * is_joint
    e02100p = farm_inc * head_frac
    e02100s = farm_inc * (1.0 - head_frac) * is_joint

    # pension contributions
    print("Imputing pretax pension contributions...")
    head_emp = emp_income * head_frac
    spouse_emp = emp_income * (1.0 - head_frac) * is_joint
    all_emp = np.concatenate([head_emp, spouse_emp])
    ei_df = pd.DataFrame({"employment_income": all_emp})
    pc_df = impute_pretax_pension_contributions(ei_df)
    pencon_all = np.minimum(all_emp, pc_df.pretax_pension_contributions.values)
    pencon_p = pencon_all[:n]
    pencon_s = pencon_all[n:]

    # build Tax-Calculator (TC) variable dictionary
    print(f"Building Tax-Calculator dataset for {taxyear}...")

    # mapping from TC variable name to PE-named column in preprocessed PUF:
    #  for person-level variables, the tax-unit total is scaled by
    #  person_scale (= head_frac + (1-head_frac)*is_joint) to match
    #  policyengine_us pipeline's sum-over-nondependents aggregation.
    tc_to_pe = {
        "RECID": "household_id",
        "S006": "household_weight",
        "E03500": "alimony_expense",
        "E00800": "alimony_income",
        "G20500": "casualty_loss",
        "E32800": "cdcc_relevant_expenses",
        "E19800": "charitable_cash_donations",
        "E20100": "charitable_non_cash_donations",
        "XTOT": "exemptions_count",
        "E03240": "domestic_production_ald",
        "E03400": "early_withdrawal_penalty",
        "E03220": "educator_expense",
        "E00200": "employment_income",
        "E02100": "farm_income",
        "E03290": "health_savings_account_ald",
        "E19200": "interest_deduction",
        "P23250": "long_term_capital_gains",
        "E24518": "long_term_capital_gains_on_collectibles",
        "E17500": "medical_expense",
        "E00650": "qualified_dividend_income",
        "E26270": "partnership_s_corp_income",
        "E03230": "qualified_tuition_expenses",
        "e87530": "qualified_tuition_expenses",
        "E18500": "real_estate_taxes",
        "E00900": "self_employment_income",
        "E03270": "self_employed_health_insurance_ald",
        "E03300": "self_employed_pension_contribution_ald",
        "P22250": "short_term_capital_gains",
        "E02400": "social_security",
        "E18400": "state_and_local_sales_or_income_tax",
        "E03210": "student_loan_interest",
        "E00300": "taxable_interest_income",
        "E02300": "taxable_unemployment_compensation",
        "E01400": "taxable_ira_distributions",
        "E00400": "tax_exempt_interest_income",
        "E01700": "taxable_pension_income",
        "E03150": "traditional_ira_contributions",
        "E24515": "unrecaptured_section_1250_gain",
        "E27200": "farm_rent_income",
        "PT_binc_w2_wages": "w2_wages_from_qualified_business",
        "e20400": "misc_deduction",
        "e07300": "foreign_tax_credit",
        "e62900": "amt_foreign_tax_credit",
        "e01200": "miscellaneous_income",
        "e00700": "salt_refund_income",
        "e58990": "investment_income_elected_form_4952",
        "e07400": "general_business_credit",
        "e07600": "prior_year_minimum_tax_credit",
        "e11200": "excess_withheld_payroll_tax",
        "e01100": "non_sch_d_capital_gains",
        "e87521": "american_opportunity_credit",
        "e07260": "energy_efficient_home_improvement_credit",
        "e09900": "qualified_retirement_penalty",
        "p08000": "other_credits",
        "e07240": "savers_credit",
        "e09700": "recapture_of_investment_credit",
        "e09800": "unreported_payroll_tax",
    }
    PERSON_LEVEL_VARS = {
        "alimony_expense",
        "alimony_income",
        "casualty_loss",
        "charitable_cash_donations",
        "charitable_non_cash_donations",
        "early_withdrawal_penalty",
        "educator_expense",
        "employment_income",
        "estate_income",
        "farm_income",
        "farm_rent_income",
        "long_term_capital_gains",
        "long_term_capital_gains_on_collectibles",
        "medical_expense",
        "non_qualified_dividend_income",
        "non_sch_d_capital_gains",
        "partnership_s_corp_income",
        "qualified_dividend_income",
        "qualified_tuition_expenses",
        "real_estate_taxes",
        "rental_income",
        "self_employment_income",
        "short_term_capital_gains",
        "social_security",
        "student_loan_interest",
        "taxable_interest_income",
        "taxable_pension_income",
        "taxable_unemployment_compensation",
        "taxable_ira_distributions",
        "tax_exempt_interest_income",
        "tax_exempt_pension_income",
        "traditional_ira_contributions",
        "amt_foreign_tax_credit",
        "miscellaneous_income",
        "salt_refund_income",
        "investment_income_elected_form_4952",
        "general_business_credit",
        "prior_year_minimum_tax_credit",
        "excess_withheld_payroll_tax",
        "other_credits",
        "w2_wages_from_qualified_business",
    }
    # Tax-Calculator names that are not financial amounts (no person_scale)
    NO_SCALE = {"RECID", "S006", "XTOT"}

    var = {}
    for tcname, pename in tc_to_pe.items():
        if tcname in NO_SCALE:
            var[tcname] = puf[pename].values
        elif pename in PERSON_LEVEL_VARS:
            var[tcname] = puf[pename].values * person_scale
        else:
            var[tcname] = puf[pename].values

    # raw PUF variables
    var["f2441"] = f2441_raw
    var["EIC"] = eic_raw
    var["MARS"] = mars_raw

    # zero-valued variables
    zeros = np.zeros(n, dtype=int)
    for tcname in [
        "a_lineno",
        "agi_bin",
        "h_seq",
        "ffpos",
        "fips",
        "DSI",
        "MIDR",
        "PT_SSTB_income",
        "PT_ubia_property",
        "cmbtp",
        "f6251",
        "k1bx14p",
        "k1bx14s",
        "tanf_ben",
        "vet_ben",
        "wic_ben",
        "snap_ben",
        "housing_ben",
        "ssi_ben",
        "mcare_ben",
        "mcaid_ben",
        "other_ben",
    ]:
        var[tcname] = zeros

    # computed variables
    var["E00600"] = (
        puf["non_qualified_dividend_income"].values
        + puf["qualified_dividend_income"].values
    ) * person_scale
    var["E01500"] = (
        puf["tax_exempt_pension_income"].values
        + puf["taxable_pension_income"].values
    ) * person_scale
    ones = np.ones(n, dtype=int)
    var["FLPDYR"] = ones * taxyear
    var["data_source"] = ones  # PUF data
    var["e02000"] = (
        puf["rental_income"].values
        + puf["partnership_s_corp_income"].values
        + puf["estate_income"].values
        + puf["farm_rent_income"].values
    ) * person_scale

    # head/spouse splits
    var["e00200p"] = e00200p
    var["e00200s"] = e00200s
    var["e00900p"] = e00900p
    var["e00900s"] = e00900s
    var["e02100p"] = e02100p
    var["e02100s"] = e02100s
    var["pencon_p"] = pencon_p
    var["pencon_s"] = pencon_s

    # demographics
    var["age_head"] = age_head
    var["age_spouse"] = age_spouse
    var["blind_head"] = zeros
    var["blind_spouse"] = zeros
    var["nu18"] = nu18
    var["nu13"] = nu13
    var["nu06"] = nu06
    var["n1820"] = n1820
    var["n21"] = n21
    var["n24"] = n24
    var["elderly_dependents"] = elderly_dependents

    df = pd.DataFrame(var)

    # correct variable name casing for Tax-Calculator
    with open(
        STORAGE_FOLDER / "input" / "tc_variable_metadata.yaml",
        "r",
        encoding="utf-8",
    ) as yfile:
        tc_variable_metadata = yaml.safe_load(yfile)
    renames = {}
    for variable in df.columns:
        if variable.upper() in tc_variable_metadata["read"]:
            renames[variable] = variable.upper()
        elif variable.lower() in tc_variable_metadata["read"]:
            renames[variable] = variable.lower()
    df.rename(columns=renames, inplace=True)

    return df
