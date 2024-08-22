# Create a Tax-Calculator-compatible dataset from any PolicyEngine hierarchical dataset.
import yaml
from typing import Type
import numpy as np
import pandas as pd
from tmd.storage import STORAGE_FOLDER
from tmd.datasets.puf import PUF_2015, PUF_2021
from policyengine_us import Microsimulation
from policyengine_us.system import system


def create_tc_dataset(pe_dataset: Type, year: int) -> pd.DataFrame:
    pe_sim = Microsimulation(dataset=pe_dataset)

    print(f"Creating tc dataset from '{pe_dataset.label}' for year {year}...")

    is_non_dep = ~pe_sim.calculate("is_tax_unit_dependent").values
    tax_unit = pe_sim.populations["tax_unit"]

    def pe(variable):
        if system.variables[variable].entity.key == "person":
            # sum over nondependents
            values = pe_sim.calculate(variable).values
            return np.array(tax_unit.sum(values * is_non_dep))
        else:
            return np.array(pe_sim.calculate(variable, map_to="tax_unit"))

    # specify tcname-to-pename dictionary for simple one-to-one variables
    vnames = {
        "RECID": "household_id",
        "S006": "tax_unit_weight",
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
        "E27200": "farm_rent_income",
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
        "E01700": "taxable_pension_income",
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
        "e09900": "early_withdrawal_penalty",
        "p08000": "other_credits",
        "e07240": "savers_credit",
        "e09700": "recapture_of_investment_credit",
        "e09800": "unreported_payroll_tax",
        "f2441": "count_cdcc_eligible",
    }
    # specify Tax-Calculator names of variables that have zero values
    zero_names = [
        "a_lineno",  # taxdata-specific (CPS matched person ID)
        "agi_bin",  # taxdata-specific (AGI bin)
        "h_seq",  # taxdata-specific (CPS matched household ID)
        "ffpos",  # taxdata-specific (CPS matched family ID)
        "fips",  # no FIPS data
        "DSI",  # claimed as dependent on another return, assume not
        "MIDR",  # separately filing spouse itemizes, assume not
        "PT_SSTB_income",  # PT SSTB business income, assume none
        "PT_ubia_property",  # PT business capital, assume none
        "cmbtp",
        "f6251",
        "k1bx14p",
        "k1bx14s",
        "tanf_ben",  # TANF benefits, assume none
        "vet_ben",  # veteran's benefits, assume none
        "wic_ben",  # WIC benefits, assume none
        "snap_ben",  # SNAP benefits, assume none
        "housing_ben",  # housing benefits, assume none
        "ssi_ben",  # SSI benefits, assume none
        "mcare_ben",  # Medicare benefits, assume none
        "mcaid_ben",  # Medicaid benefits, assume none
        "other_ben",  # Other benefits, assume none
    ]
    # specify Tax-Calculator array variable dictionary and use it to create df
    var = {}
    for tcname, pename in vnames.items():
        var[tcname] = pe(pename)
    zeros = np.zeros_like(var["RECID"], dtype=int)
    for tcname in zero_names:
        var[tcname] = zeros
    var["E00600"] = pe("non_qualified_dividend_income") + pe(
        "qualified_dividend_income"
    )
    var["E01500"] = pe("tax_exempt_pension_income") + pe(
        "taxable_pension_income"
    )
    var["MARS"] = (
        pd.Series(pe("filing_status"))
        .map(
            {
                "SINGLE": 1,
                "JOINT": 2,
                "SEPARATE": 3,
                "HEAD_OF_HOUSEHOLD": 4,
                "SURVIVING_SPOUSE": 5,
            }
        )
        .values
    )
    var["EIC"] = np.minimum(pe("eitc_child_count"), 3)
    ones = np.ones_like(zeros, dtype=int)
    var["FLPDYR"] = ones * year
    if "puf" in pe_dataset.__name__.lower():
        var["data_source"] = ones
    else:
        var["data_source"] = zeros
    var["e02000"] = (
        pe("rental_income")
        + pe("partnership_s_corp_income")
        + pe("estate_income")
        + pe("farm_rent_income")
    )
    df = pd.DataFrame(var)

    # specify person-to-tax_unit mapping function
    map_to_tax_unit = lambda arr: pe_sim.map_result(arr, "person", "tax_unit")

    # specify df head/spouse variables
    head = pe_sim.calculate("is_tax_unit_head").values
    spouse = pe_sim.calculate("is_tax_unit_spouse").values

    employment_income = pe_sim.calculate("employment_income").values
    self_employment_income = pe_sim.calculate("self_employment_income").values
    farm_income = pe_sim.calculate("farm_income").values
    pre_tax_contributions = pe_sim.calculate("pre_tax_contributions").values

    df["e00200p"] = map_to_tax_unit(employment_income * head)
    df["e00200s"] = map_to_tax_unit(employment_income * spouse)
    df["e00900p"] = map_to_tax_unit(self_employment_income * head)
    df["e00900s"] = map_to_tax_unit(self_employment_income * spouse)
    df["e02100p"] = map_to_tax_unit(farm_income * head)
    df["e02100s"] = map_to_tax_unit(farm_income * spouse)
    df["pencon_p"] = map_to_tax_unit(pre_tax_contributions * head)
    df["pencon_s"] = map_to_tax_unit(pre_tax_contributions * spouse)

    # specify df demographics
    age = pe_sim.calculate("age").values
    dependent = pe_sim.calculate("is_tax_unit_dependent").values
    blind = pe_sim.calculate("is_blind").values

    df["age_head"] = map_to_tax_unit(age * head)
    df["age_spouse"] = map_to_tax_unit(age * spouse)
    df["blind_head"] = map_to_tax_unit(blind * head)
    df["blind_spouse"] = map_to_tax_unit(blind * spouse)
    df["nu18"] = map_to_tax_unit((age < 18) * dependent)
    df["nu13"] = map_to_tax_unit((age < 13) * dependent)
    df["nu06"] = map_to_tax_unit((age < 6) * dependent)
    df["n1820"] = map_to_tax_unit(((age >= 18) & (age < 21)) * dependent)
    df["n21"] = map_to_tax_unit((age >= 21) * dependent)
    df["n24"] = map_to_tax_unit((age < 17) * dependent)  # usinng taxdata logic
    df["elderly_dependents"] = map_to_tax_unit((age >= 65) * dependent)

    # correct case of df variable names for Tax-Calculator
    tc_variable_metadata = yaml.safe_load(
        open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml", "r")
    )
    renames = {}
    for variable in df.columns:
        if variable.upper() in tc_variable_metadata["read"]:
            renames[variable] = variable.upper()
        elif variable.lower() in tc_variable_metadata["read"]:
            renames[variable] = variable.lower()
    df.rename(columns=renames, inplace=True)

    return df


def create_tc_puf_2015():
    return create_tc_dataset(PUF_2015, 2015)


def create_tc_puf_2021():
    return create_tc_dataset(PUF_2021, 2021)


if __name__ == "__main__":
    create_tc_dataset(PUF_2015).to_csv(
        STORAGE_FOLDER / "output" / "tc_puf_2015.csv.gz", index=False
    )
    create_tc_dataset(PUF_2021).to_csv(
        STORAGE_FOLDER / "output" / "tc_puf_2021.csv.gz", index=False
    )
