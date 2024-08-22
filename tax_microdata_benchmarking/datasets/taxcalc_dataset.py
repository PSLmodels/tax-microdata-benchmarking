# Create a Tax-Calculator-compatible dataset from any PolicyEngine hierarchical dataset.
import yaml
from typing import Type
import numpy as np
import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.datasets.puf import PUF_2015, PUF_2021
from policyengine_us import Microsimulation
from policyengine_us.system import system


def create_tc_dataset(pe_dataset: Type, year: int) -> pd.DataFrame:
    pe_sim = Microsimulation(dataset=pe_dataset)

    print(f"Creating tc dataset from '{pe_dataset.label}' for year {year}...")

    is_non_dep = ~pe_sim.calculate("is_tax_unit_dependent").values
    tax_unit = pe_sim.populations["tax_unit"]

    def pe(variable):
        if system.variables[variable].entity.key == "person":
            # sum over non-dependents
            values = pe_sim.calculate(variable).values
            return np.array(tax_unit.sum(values * is_non_dep))
        else:
            return np.array(pe_sim.calculate(variable, map_to="tax_unit"))

    # specify tcname-to-pename dictionary
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
    # specify Tax-Calculator array variable dictionary
    var = {}
    for tcname, pename in vnames.items():
        var[tcname] = pe(pename)
    zeros = np.zeros_like(var["RECID"])
    for tcname in zero_names:
        var[tcname] = zeros

    var["E00600"] = pe("non_qualified_dividend_income") + pe(
        "qualified_dividend_income"
    )
    var["E01500"] = pe("tax_exempt_pension_income") + pe(
        "taxable_pension_income"
    )

    df = pd.DataFrame(var)

    """
    df = pd.DataFrame()

    df["E03500"] = pe("alimony_expense")
    df["E00800"] = pe("alimony_income")
    df["G20500"] = pe(
        "casualty_loss"
    )  # Amend with taxdata treatment from e20500
    df["E32800"] = pe("cdcc_relevant_expenses")
    df["E19800"] = pe("charitable_cash_donations")
    df["E20100"] = pe("charitable_non_cash_donations")
    df["XTOT"] = pe("exemptions_count")
    df["E03240"] = pe("domestic_production_ald")
    df["E03400"] = pe("early_withdrawal_penalty")
    df["E03220"] = pe("educator_expense")
    df["E00200"] = pe("employment_income")
    df["E02100"] = pe("farm_income")
    df["E27200"] = pe("farm_rent_income")
    df["E03290"] = pe("health_savings_account_ald")
    df["E19200"] = pe("interest_deduction")
    df["P23250"] = pe("long_term_capital_gains")
    df["E24518"] = pe("long_term_capital_gains_on_collectibles")
    df["E17500"] = pe("medical_expense")
    df["E00600"] = pe("non_qualified_dividend_income") + pe(
        "qualified_dividend_income"
    )
    df["E00650"] = pe("qualified_dividend_income")
    df["E26270"] = pe("partnership_s_corp_income")
    df["E03230"] = pe("qualified_tuition_expenses")
    df["E18500"] = pe("real_estate_taxes")
    df["E00900"] = pe("self_employment_income")
    df["E03270"] = pe("self_employed_health_insurance_ald")
    df["E03300"] = pe("self_employed_pension_contribution_ald")
    df["P22250"] = pe("short_term_capital_gains")
    df["E02400"] = pe("social_security")
    df["E18400"] = pe("state_and_local_sales_or_income_tax")
    df["E03210"] = pe("student_loan_interest")
    df["E00300"] = pe("taxable_interest_income")
    df["E01700"] = pe("taxable_pension_income")
    df["E02300"] = pe("taxable_unemployment_compensation")
    df["E01400"] = pe("taxable_ira_distributions")
    df["E00400"] = pe("tax_exempt_interest_income")
    df["E01500"] = pe("tax_exempt_pension_income") + pe(
        "taxable_pension_income"
    )
    df["E01700"] = pe("taxable_pension_income")
    df["E03150"] = pe("traditional_ira_contributions")
    df["E24515"] = pe("unrecaptured_section_1250_gain")
    df["E27200"] = pe("farm_rent_income")
    """

    df["MARS"] = (
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

    """
    df["RECID"] = pe("household_id")
    df["S006"] = pe("tax_unit_weight")
    """

    """
    df["a_lineno"] = 0  # TD-specific (CPS matched person ID)
    df["agi_bin"] = 0  # TD-specific (AGI bin)
    df["h_seq"] = 0  # TD-specific (CPS matched household ID)
    df["ffpos"] = 0  # TD-specific (CPS matched family ID)
    df["fips"] = 0  # No FIPS data
    df["DSI"] = 0  # Claimed as dependent on another return, assume not
    """

    df["EIC"] = np.minimum(pe("eitc_child_count"), 3)
    df["FLPDYR"] = year

    """
    df["MIDR"] = 0  # Separately filing spouse itemizes, assume not
    df["PT_SSTB_income"] = (
        0  # Business income is from specified service trade assume not
    )
    df["tanf_ben"] = 0  # TANF benefits, assume none
    df["vet_ben"] = 0  # Veteran's benefits, assume none
    df["wic_ben"] = 0  # WIC benefits, assume none
    df["snap_ben"] = 0  # SNAP benefits, assume none
    df["housing_ben"] = 0  # Housing benefits, assume none
    df["ssi_ben"] = 0  # SSI benefits, assume none
    df["mcare_ben"] = 0  # Medicare benefits, assume none
    df["mcaid_ben"] = 0  # Medicaid benefits, assume none
    df["other_ben"] = 0  # Other benefits, assume none
    """

    """
    df["PT_binc_w2_wages"] = pe("w2_wages_from_qualified_business")
    df["PT_ubia_property"] = 0
    """

    df["data_source"] = 1 if "puf" in pe_dataset.__name__.lower() else 0
    df["e02000"] = (
        pe("rental_income")
        + pe("partnership_s_corp_income")
        + pe("estate_income")
        + pe("farm_rent_income")
    )

    """
    df["e20400"] = pe("misc_deduction")
    df["e07300"] = pe("foreign_tax_credit")
    df["e62900"] = pe("amt_foreign_tax_credit")
    df["e01200"] = pe("miscellaneous_income")
    df["e00700"] = pe("salt_refund_income")
    df["e58990"] = pe("investment_income_elected_form_4952")
    df["e07400"] = pe("general_business_credit")
    df["e07600"] = pe("prior_year_minimum_tax_credit")
    df["e11200"] = pe("excess_withheld_payroll_tax")
    df["e01100"] = pe("non_sch_d_capital_gains")
    df["e87521"] = pe("american_opportunity_credit")
    df["e07260"] = pe("energy_efficient_home_improvement_credit")
    df["e09900"] = pe("early_withdrawal_penalty")
    df["p08000"] = pe("other_credits")
    df["e07240"] = pe("savers_credit")
    df["e09700"] = pe("recapture_of_investment_credit")
    df["e09800"] = pe("unreported_payroll_tax")
    df["f2441"] = pe("count_cdcc_eligible")
    df["e87530"] = df[
        "E03230"
    ]  # Assume same definition for tuition expenses (for now).

    df["cmbtp"] = 0
    df["f6251"] = 0
    df["k1bx14p"] = 0
    df["k1bx14s"] = 0
    """

    # head and spouse pairs

    map_to_tax_unit = lambda arr: pe_sim.map_result(arr, "person", "tax_unit")

    filer = pe_sim.calculate("is_tax_unit_head").values
    spouse = pe_sim.calculate("is_tax_unit_spouse").values

    employment_income = pe_sim.calculate("employment_income").values
    self_employment_income = pe_sim.calculate("self_employment_income").values
    farm_income = pe_sim.calculate("farm_income").values
    pre_tax_contributions = pe_sim.calculate("pre_tax_contributions").values

    df["e00200p"] = map_to_tax_unit(employment_income * filer)
    df["e00200s"] = map_to_tax_unit(employment_income * spouse)
    df["e00900p"] = map_to_tax_unit(self_employment_income * filer)
    df["e00900s"] = map_to_tax_unit(self_employment_income * spouse)
    df["e02100p"] = map_to_tax_unit(farm_income * filer)
    df["e02100s"] = map_to_tax_unit(farm_income * spouse)
    df["pencon_p"] = map_to_tax_unit(pre_tax_contributions * filer)
    df["pencon_s"] = map_to_tax_unit(pre_tax_contributions * spouse)

    # Demographics

    age = pe_sim.calculate("age").values
    head = pe_sim.calculate("is_tax_unit_head").values
    spouse = pe_sim.calculate("is_tax_unit_spouse").values
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
    df["n24"] = map_to_tax_unit(
        (age < 17) * dependent
    )  # following taxdata code.
    df["elderly_dependents"] = map_to_tax_unit((age >= 65) * dependent)

    # correct case of variable names for Tax-Calculator
    tc_variable_metadata = yaml.safe_load(
        open(STORAGE_FOLDER / "input" / "taxcalc_variable_metadata.yaml", "r")
    )
    renames = {}
    for variable in df.columns:
        if variable.upper() in tc_variable_metadata["read"]:
            renames[variable] = variable.upper()
        elif variable.lower() in tc_variable_metadata["read"]:
            renames[variable] = variable.lower()
    df = df.rename(columns=renames)

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
