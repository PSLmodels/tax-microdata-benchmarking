# This file should create tax_microdata.csv.gz in the root of the repo.

import taxcalc as tc
from policyengine_us import Microsimulation
from policyengine_us.model_api import *
import numpy as np
import pandas as pd


class TaxCalcVariableAlias(Variable):
    label = "TaxCalc Variable Alias"
    definition_period = YEAR
    entity = TaxUnit
    value_type = float


class tc_RECID(TaxCalcVariableAlias):
    label = "record ID"

    def formula(tax_unit, period, parameters):
        return tax_unit("tax_unit_id", period)


class tc_MARS(TaxCalcVariableAlias):
    label = "filing status"

    def formula(tax_unit, period, parameters):
        filing_status = tax_unit("filing_status", period).decode_to_str()
        CODE_MAP = {
            "SINGLE": 1,
            "JOINT": 2,
            "SEPARATE": 3,
            "HEAD_OF_HOUSEHOLD": 4,
            "WIDOW": 5,
        }
        return pd.Series(filing_status).map(CODE_MAP)


class tc_e00200p(TaxCalcVariableAlias):
    label = "wages less pension contributions (filer)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        employment_income = person("employment_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(employment_income * is_tax_unit_head)


class tc_e00200s(TaxCalcVariableAlias):
    label = "wages less pension contributions (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        employment_income = person("employment_income", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(employment_income * is_tax_unit_spouse)


class tc_e00200(TaxCalcVariableAlias):
    label = "wages less pension contributions"

    adds = [
        "tc_e00200p",
        "tc_e00200s",
    ]


class tc_age_head(TaxCalcVariableAlias):
    label = "age of head of tax unit"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.max(age * is_tax_unit_head)


class tc_age_spouse(TaxCalcVariableAlias):
    label = "age of spouse of head of tax unit"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.max(age * is_tax_unit_spouse)


class tc_blind_head(TaxCalcVariableAlias):
    label = "blindness of head of tax unit"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        is_blind = person("is_blind", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.max(is_blind * is_tax_unit_head)


class tc_blind_spouse(TaxCalcVariableAlias):
    label = "blindness of spouse of head of tax unit"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        is_blind = person("is_blind", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.max(is_blind * is_tax_unit_spouse)


class tc_fips(TaxCalcVariableAlias):
    label = "FIPS state code"

    def formula(tax_unit, period, parameters):
        return tax_unit.household("state_fips", period)


# h_seq, a_lineno, ffpos skipped, just ID variables (household, person, family)


class tc_s006(TaxCalcVariableAlias):
    label = "tax unit weight"

    def formula(tax_unit, period, parameters):
        return tax_unit("tax_unit_weight", period)


class tc_FLPDYR(TaxCalcVariableAlias):
    label = "tax year to calculate for"  # QUESTION: how does this work? Just going to put 2023 for now.

    def formula(tax_unit, period, parameters):
        return period.start.year


class tc_EIC(TaxCalcVariableAlias):
    label = "EITC-qualifying children"

    def formula(tax_unit, period, parameters):
        return min_(
            add(tax_unit, period, ["is_eitc_qualifying_child"]),
            3,  # Must be capped in the data rather than the policy for Tax-Calculator
        )


class tc_nu18(TaxCalcVariableAlias):
    label = "number of people under 18"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        return tax_unit.sum(age < 18)


class tc_n1820(TaxCalcVariableAlias):
    label = "number of people 18-20"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        return tax_unit.sum((age >= 18) & (age <= 20))


class tc_nu13(TaxCalcVariableAlias):
    label = "number of people under 13"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        return tax_unit.sum(age < 13)


class tc_nu06(TaxCalcVariableAlias):
    label = "number of people under 6"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        return tax_unit.sum(age < 6)


class tc_n24(TaxCalcVariableAlias):
    label = "number of people eligible for the CTC"
    adds = ["ctc_qualifying_children"]


class tc_elderly_dependents(TaxCalcVariableAlias):
    label = "number of elderly dependents"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        is_tax_unit_dependent = person("is_tax_unit_dependent", period)
        return tax_unit.sum((age >= 65) * is_tax_unit_dependent)


class tc_f2441(TaxCalcVariableAlias):
    label = "CDCC-qualifying children"
    adds = ["count_cdcc_eligible"]


class tc_e00900p(TaxCalcVariableAlias):
    label = "self-employment income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        self_employment_income = person("self_employment_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(self_employment_income * is_tax_unit_head)


class tc_e00900s(TaxCalcVariableAlias):
    label = "self-employment income (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        self_employment_income = person("self_employment_income", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(self_employment_income * is_tax_unit_spouse)


class tc_e00900(TaxCalcVariableAlias):
    label = "self-employment income"
    adds = ["tc_e00900p", "tc_e00900s"]


class tc_e02100p(TaxCalcVariableAlias):
    label = "farm income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        farm_income = person("farm_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(farm_income * is_tax_unit_head)


class tc_e02100s(TaxCalcVariableAlias):
    label = "farm income (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        farm_income = person("farm_income", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(farm_income * is_tax_unit_spouse)


class tc_e02100(TaxCalcVariableAlias):
    label = "farm income"
    adds = ["tc_e02100p", "tc_e02100s"]


class tc_e01500p(TaxCalcVariableAlias):
    label = "pension income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("pension_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(pension_income * is_tax_unit_head)


class tc_e01500s(TaxCalcVariableAlias):
    label = "pension income (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("pension_income", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(pension_income * is_tax_unit_spouse)


class tc_e01500(TaxCalcVariableAlias):
    label = "pension income"
    adds = ["tc_e01500p", "tc_e01500s"]


class tc_e00800p(TaxCalcVariableAlias):
    label = "alimony income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        alimony_income = person("alimony_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(alimony_income * is_tax_unit_head)


class tc_e00800s(TaxCalcVariableAlias):
    label = "alimony income (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        alimony_income = person("alimony_income", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(alimony_income * is_tax_unit_spouse)


class tc_e00800(TaxCalcVariableAlias):
    label = "alimony income"
    adds = ["tc_e00800p", "tc_e00800s"]


class tc_e02400p(TaxCalcVariableAlias):
    label = "social security income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        social_security_income = person("social_security", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(social_security_income * is_tax_unit_head)


class tc_e02400s(TaxCalcVariableAlias):
    label = "social security income (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        social_security_income = person("social_security", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(social_security_income * is_tax_unit_spouse)


class tc_e02400(TaxCalcVariableAlias):
    label = "social security income"
    adds = ["tc_e02400p", "tc_e02400s"]


class tc_e02300p(TaxCalcVariableAlias):
    label = "unemployment compensation"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        unemployment_compensation = person("unemployment_compensation", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(unemployment_compensation * is_tax_unit_head)


class tc_e02300s(TaxCalcVariableAlias):
    label = "unemployment compensation (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        unemployment_compensation = person("unemployment_compensation", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(unemployment_compensation * is_tax_unit_spouse)


class tc_e02300(TaxCalcVariableAlias):
    label = "unemployment compensation"
    adds = ["tc_e02300p", "tc_e02300s"]


class taxcalc_extension(Reform):
    def apply(self):
        self.add_variables(
            tc_RECID,
            tc_MARS,
            tc_e00200p,
            tc_e00200s,
            tc_e00200,
            tc_age_head,
            tc_age_spouse,
            tc_blind_head,
            tc_blind_spouse,
            tc_fips,
            tc_s006,
            tc_FLPDYR,
            tc_EIC,
            tc_nu18,
            tc_n1820,
            tc_nu13,
            tc_nu06,
            tc_n24,
            tc_elderly_dependents,
            tc_f2441,
            tc_e00900p,
            tc_e00900s,
            tc_e00900,
            tc_e02100p,
            tc_e02100s,
            tc_e02100,
            tc_e01500p,
            tc_e01500s,
            tc_e01500,
            tc_e00800p,
            tc_e00800s,
            tc_e00800,
            tc_e02400p,
            tc_e02400s,
            tc_e02400,
            tc_e02300p,
            tc_e02300s,
            tc_e02300,
        )


def create_flat_file():
    sim = Microsimulation(
        reform=taxcalc_extension, dataset="enhanced_cps_2023"
    )
    df = pd.DataFrame()

    for variable in sim.tax_benefit_system.variables:
        if variable.startswith("tc_"):
            df[variable[3:]] = sim.calculate(variable).values.astype(
                np.float64
            )

    # Extra quality-control checks to do with different data types, nothing major
    df.e00200 = df.e00200p + df.e00200s
    df.e00900 = df.e00900p + df.e00900s
    df.e02100 = df.e02100p + df.e02100s
    df.e01500 = df.e01500p + df.e01500s
    df.e00800 = df.e00800p + df.e00800s
    df.e02400 = df.e02400p + df.e02400s
    df.e02300 = df.e02300p + df.e02300s
    df.RECID = df.RECID.astype(int)
    df.MARS = df.MARS.astype(int)

    print(f"Completed data generation for {len(df.columns)}/68 variables.")

    df.to_csv("tax_microdata.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    create_flat_file()
