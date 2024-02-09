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
        return add(tax_unit, period, ["is_eitc_qualifying_child"])


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
    df.RECID = df.RECID.astype(int)
    df.MARS = df.MARS.astype(int)

    print(f"Completed data generation for {len(df.columns)}/68 variables.")

    df.to_csv("tax_microdata.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    create_flat_file()
