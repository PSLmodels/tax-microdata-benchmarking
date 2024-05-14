"""
This module contains a PolicyEngine-US reform that adds variable outputs aligning with Tax-Calculator inputs.
"""

import warnings

warnings.filterwarnings("ignore")
import taxcalc as tc
from policyengine_us import Microsimulation
from policyengine_us.model_api import *
from policyengine_us.system import system
import numpy as np
import pandas as pd
from policyengine_core.periods import instant
from scipy.optimize import minimize
from tax_microdata_benchmarking.utils.qbi import add_pt_w2_wages
from microdf import MicroDataFrame
import numpy as np

UPRATING_VARIABLES = [
    "employment_income",
    "self_employment_income",
    "farm_income",
    "pension_income",
    "alimony_income",
    "social_security",
    "unemployment_compensation",
    "dividend_income",
    "qualified_dividend_income",
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "taxable_pension_income",
    "non_sch_d_capital_gains",
    "taxable_ira_distributions",
    "self_employed_health_insurance_premiums",
    "cdcc_relevant_expenses",
    "medical_expense",
    "pre_tax_contributions",
    "traditional_ira_contributions",
    "student_loan_interest",
    "short_term_capital_gains",
    "long_term_capital_gains",
    "wic",
]


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
            "SURVIVING_SPOUSE": 5,
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
    adds = ["tc_e00200p", "tc_e00200s"]


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


class tc_s006(TaxCalcVariableAlias):
    label = "tax unit weight"

    def formula(tax_unit, period, parameters):
        return tax_unit.household("household_weight", period)


class tc_FLPDYR(TaxCalcVariableAlias):
    label = "tax year to calculate for"

    def formula(tax_unit, period, parameters):
        return period.start.year


class tc_EIC(TaxCalcVariableAlias):
    label = "EITC-qualifying children"

    def formula(tax_unit, period, parameters):
        return min_(
            add(tax_unit, period, ["is_child_dependent"]),
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


class tc_e01500(TaxCalcVariableAlias):
    label = "pension income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("pension_income", period)
        is_tax_unit_dependent = person("is_tax_unit_dependent", period)
        return tax_unit.sum(pension_income * ~is_tax_unit_dependent)


class tc_e00800(TaxCalcVariableAlias):
    label = "alimony income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("alimony_income", period)
        is_tax_unit_dependent = person("is_tax_unit_dependent", period)
        return tax_unit.sum(pension_income * ~is_tax_unit_dependent)


class tc_e02400(TaxCalcVariableAlias):
    label = "social security income"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("social_security", period)
        is_tax_unit_dependent = person("is_tax_unit_dependent", period)
        return tax_unit.sum(pension_income * ~is_tax_unit_dependent)


class tc_e02300(TaxCalcVariableAlias):
    label = "unemployment compensation"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        pension_income = person("unemployment_compensation", period)
        is_tax_unit_dependent = person("is_tax_unit_dependent", period)
        return tax_unit.sum(pension_income * ~is_tax_unit_dependent)


class tc_XTOT(TaxCalcVariableAlias):
    label = "total exemptions"

    def formula(tax_unit, period, parameters):
        return tax_unit.nb_persons()


class tc_ssi_ben(TaxCalcVariableAlias):
    label = "SSI"
    adds = ["ssi"]


class tc_mcaid_ben(TaxCalcVariableAlias):
    label = "Medicaid"
    adds = ["medicaid"]


class tc_tanf_ben(TaxCalcVariableAlias):
    label = "TANF"
    adds = ["tanf"]


class tc_snap_ben(TaxCalcVariableAlias):
    label = "SNAP"
    adds = ["snap"]


class tc_housing_ben(TaxCalcVariableAlias):
    label = "housing subsidy"
    adds = ["spm_unit_capped_housing_subsidy"]


class tc_DSI(TaxCalcVariableAlias):
    label = "dependent filer"

    def formula(tax_unit, period, parameters):
        return 0


class tc_n21(TaxCalcVariableAlias):
    label = "number of people 21 or over"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        age = person("age", period)
        return tax_unit.sum(age >= 21)


class tc_e00600(TaxCalcVariableAlias):
    label = "ordinary dividends included in AGI"
    adds = ["dividend_income"]


class tc_e18400(TaxCalcVariableAlias):
    label = "State income tax"
    adds = ["state_income_tax"]


class tc_e00650(TaxCalcVariableAlias):
    label = "qualified dividends"
    adds = ["qualified_dividend_income"]


class tc_e00300(TaxCalcVariableAlias):
    label = "taxable interest income"
    adds = ["taxable_interest_income"]


class tc_e00400(TaxCalcVariableAlias):
    label = "tax-exempt interest income"
    adds = ["tax_exempt_interest_income"]


class tc_e01700(TaxCalcVariableAlias):
    label = "taxable pension income"
    adds = ["taxable_pension_income"]


class tc_e01100(TaxCalcVariableAlias):
    label = "capital gains not on Sch. D"
    adds = ["non_sch_d_capital_gains"]


class tc_e01400(TaxCalcVariableAlias):
    label = "taxable IRA distributions"
    adds = ["taxable_ira_distributions"]


class tc_e03270(TaxCalcVariableAlias):
    label = "self-employed health insurance deduction"
    adds = ["self_employed_health_insurance_premiums"]


class tc_e32800(TaxCalcVariableAlias):
    label = "child and dependent care expenses"
    adds = ["cdcc_relevant_expenses"]


class tc_e17500(TaxCalcVariableAlias):
    label = "medical and dental expenses"
    adds = ["medical_expense"]


class tc_pencon_p(TaxCalcVariableAlias):
    label = "pension contributions (filer)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        employment_income = person("pre_tax_contributions", period)
        is_tax_unit_head = person("is_tax_unit_head", period)
        return tax_unit.sum(employment_income * is_tax_unit_head)


class tc_pencon_s(TaxCalcVariableAlias):
    label = "pension contributions (spouse)"

    def formula(tax_unit, period, parameters):
        person = tax_unit.members
        employment_income = person("pre_tax_contributions", period)
        is_tax_unit_spouse = person("is_tax_unit_spouse", period)
        return tax_unit.sum(employment_income * is_tax_unit_spouse)


class tc_e03150(TaxCalcVariableAlias):
    label = "deductible IRA contributions"
    adds = ["traditional_ira_contributions"]


class tc_e03210(TaxCalcVariableAlias):
    label = "student loan interest"
    adds = ["student_loan_interest"]


class tc_p22250(TaxCalcVariableAlias):
    label = "net short-term capital gains"
    adds = ["short_term_capital_gains"]


class tc_p23250(TaxCalcVariableAlias):
    label = "net long-term capital gains"
    adds = ["long_term_capital_gains"]


class tc_wic_ben(TaxCalcVariableAlias):
    label = "WIC"
    adds = ["wic"]


class tc_e18500(TaxCalcVariableAlias):
    label = "real-estate taxes paid"
    adds = ["real_estate_taxes"]


class tc_e19200(TaxCalcVariableAlias):
    label = "interest expense"
    adds = ["interest_expense"]


class is_tax_filer(Variable):
    label = "tax filer"
    value_type = bool
    entity = TaxUnit
    definition_period = YEAR

    """
    (a) General rule
    Returns with respect to income taxes under subtitle A shall be made by the following:
    (1)
    (A) Every individual having for the taxable year gross income which equals or exceeds the exemption amount, except that a return shall not be required of an individual—
    (i) who is not married (determined by applying section 7703), is not a surviving spouse (as defined in section 2(a)), is not a head of a household (as defined in section 2(b)), and for the taxable year has gross income of less than the sum of the exemption amount plus the basic standard deduction applicable to such an individual,
    (ii) who is a head of a household (as so defined) and for the taxable year has gross income of less than the sum of the exemption amount plus the basic standard deduction applicable to such an individual,
    (iii) who is a surviving spouse (as so defined) and for the taxable year has gross income of less than the sum of the exemption amount plus the basic standard deduction applicable to such an individual, or
    (iv) who is entitled to make a joint return and whose gross income, when combined with the gross income of his spouse, is, for the taxable year, less than the sum of twice the exemption amount plus the basic standard deduction applicable to a joint return, but only if such individual and his spouse, at the close of the taxable year, had the same household as their home.
    Clause (iv) shall not apply if for the taxable year such spouse makes a separate return or any other taxpayer is entitled to an exemption for such spouse under section 151(c).
    (B) The amount specified in clause (i), (ii), or (iii) of subparagraph (A) shall be increased by the amount of 1 additional standard deduction (within the meaning of section 63(c)(3)) in the case of an individual entitled to such deduction by reason of section 63(f)(1)(A) (relating to individuals age 65 or more), and the amount specified in clause (iv) of subparagraph (A) shall be increased by the amount of the additional standard deduction for each additional standard deduction to which the individual or his spouse is entitled by reason of section 63(f)(1).
    (C) The exception under subparagraph (A) shall not apply to any individual—
    (i) who is described in section 63(c)(5) and who has—
    (I) income (other than earned income) in excess of the sum of the amount in effect under section 63(c)(5)(A) plus the additional standard deduction (if any) to which the individual is entitled, or
    (II) total gross income in excess of the standard deduction, or
    (ii) for whom the standard deduction is zero under section 63(c)(6).
    (D) For purposes of this subsection—
    (i) The terms “standard deduction”, “basic standard deduction” and “additional standard deduction” have the respective meanings given such terms by section 63(c).
    (ii) The term “exemption amount” has the meaning given such term by section 151(d). In the case of an individual described in section 151(d)(2), the exemption amount shall be zero.
    """

    def formula(tax_unit, period, parameters):
        gross_income = add(tax_unit, period, ["irs_gross_income"])
        exemption_amount = parameters(period).gov.irs.income.exemption.amount

        # (a)(1)(A), (a)(1)(B)

        filing_status = tax_unit("filing_status", period).decode_to_str()
        separate = filing_status == "SEPARATE"
        standard_deduction = tax_unit("standard_deduction", period)
        threshold = where(
            separate,
            exemption_amount,
            standard_deduction + exemption_amount,
        )

        income_over_exemption_amount = gross_income > threshold

        # (a)(1)(C)

        unearned_income_threshold = 500 + tax_unit(
            "additional_standard_deduction", period
        )
        unearned_income = gross_income - add(
            tax_unit, period, ["earned_income"]
        )
        unearned_income_over_threshold = (
            unearned_income > unearned_income_threshold
        )

        required_to_file = (
            income_over_exemption_amount | unearned_income_over_threshold
        )

        tax_refund = tax_unit("income_tax", period) < 0
        not_required_but_likely_filer = ~required_to_file & tax_refund

        # (a)(1)(D) is just definitions

        return required_to_file | not_required_but_likely_filer


EXTRA_PUF_VARIABLES = [
    "e02000",
    "e26270",
    "e19200",
    "e18500",
    "e19800",
    "e20400",
    "e20100",
    "e00700",
    "e03270",
    "e24515",
    "e03300",
    "e07300",
    "e62900",
    "e32800",
    "e87530",
    "e03240",
    "e01100",
    "e01200",
    "e24518",
    "e09900",
    "e27200",
    "e03290",
    "e58990",
    "e03230",
    "e07400",
    "e11200",
    "e07260",
    "e07240",
    "e07600",
    "e03220",
    "p08000",
    "e03400",
    "e09800",
    "e09700",
    "e03500",
    "e87521",
]

for variable in EXTRA_PUF_VARIABLES:
    globals()[f"tc_{variable}"] = type(
        f"tc_{variable}",
        (TaxCalcVariableAlias,),
        {"label": variable, "adds": [variable]},
    )


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
            tc_e01500,
            tc_e00800,
            tc_e02400,
            tc_e02300,
            tc_XTOT,
            tc_ssi_ben,
            tc_mcaid_ben,
            tc_tanf_ben,
            tc_snap_ben,
            tc_housing_ben,
            tc_DSI,
            tc_n21,
            tc_e00600,
            tc_e18400,
            tc_e00650,
            tc_e00300,
            tc_e00400,
            tc_e01700,
            tc_e01100,
            tc_e01400,
            tc_e03270,
            tc_e32800,
            tc_e17500,
            tc_pencon_p,
            tc_pencon_s,
            tc_e03150,
            tc_e03210,
            tc_p22250,
            tc_p23250,
            tc_wic_ben,
            is_tax_filer,
            tc_e18500,
            tc_e19200,
        )

        for variable in EXTRA_PUF_VARIABLES:
            self.update_variable(globals()[f"tc_{variable}"])
