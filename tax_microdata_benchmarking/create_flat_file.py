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
    def formula(tax_unit, period, parameters):
        return tax_unit("tax_unit_id", period)


class tc_MARS(TaxCalcVariableAlias):
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
    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        employment_income = person("employment_income", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(employment_income * is_tax_unit_head)


class tc_e00200s(TaxCalcVariableAlias):
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


class taxcalc_extension(Reform):
    def apply(self):
        self.add_variables(
            tc_RECID,
            tc_MARS,
            tc_e00200p,
            tc_e00200s,
            tc_e00200,
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

    df.to_csv("tax_microdata.csv.gz", index=False, compression="gzip")


if __name__ == "__main__":
    create_flat_file()
