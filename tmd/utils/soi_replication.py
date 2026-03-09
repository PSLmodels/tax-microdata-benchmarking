import pandas as pd
import taxcalc as tc


def tc_to_soi(puf, year):
    policy = tc.Policy()
    data = tc.Records(
        data=puf,
        start_year=year,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
        exact_calculations=True,
    )
    calculator = tc.Calculator(policy=policy, records=data)
    calculator.calc_all()
    puf = calculator.dataframe(None, all_vars=True)

    puf.columns = puf.columns.str.upper()

    df = pd.DataFrame()
    df["adjusted_gross_income"] = puf.C00100
    df["total_income_tax"] = puf.C09200 - puf.REFUND
    df["employment_income"] = puf.E00200
    df["capital_gains_distributions"] = puf.E01100
    df["capital_gains_gross"] = puf["C01000"] * (puf["C01000"] > 0)
    df["capital_gains_losses"] = -puf["C01000"] * (puf["C01000"] < 0)
    df["estate_income"] = 0  # puf.E26390 Not in TC
    df["estate_losses"] = 0  # puf.E26400
    df["exempt_interest"] = puf.E00400
    df["ira_distributions"] = puf.E01400
    df["count_of_exemptions"] = puf.XTOT
    df["ordinary_dividends"] = puf.E00600
    df["partnership_and_s_corp_income"] = puf.E26270 * (puf.E26270 > 0)
    df["partnership_and_s_corp_losses"] = -puf.E26270 * (puf.E26270 < 0)
    df["total_pension_income"] = puf.E01500
    df["taxable_pension_income"] = puf.E01700
    df["qualified_dividends"] = puf.E00650
    df["rent_and_royalty_net_income"] = 0  # puf.E25850 Not in TC
    df["rent_and_royalty_net_losses"] = 0  # puf.E25860
    df["total_social_security"] = puf.E02400
    df["taxable_social_security"] = puf.C02500
    df["income_tax_before_credits"] = puf.C05800
    df["taxable_interest_income"] = puf.E00300
    df["unemployment_compensation"] = puf.E02300
    df["employment_income"] = puf.E00200
    df["charitable_contributions_deduction"] = puf.C19700
    df["interest_paid_deductions"] = puf.E19200
    df["medical_expense_deductions_uncapped"] = puf.E17500
    df["itemized_state_income_and_sales_tax_deductions"] = puf.E18400
    df["itemized_real_estate_tax_deductions"] = puf.E18500
    df["state_and_local_tax_deductions"] = puf.E18400 + puf.E18500
    df["income_tax_after_credits"] = puf.IITAX
    df["business_net_profits"] = puf.E00900 * (puf.E00900 > 0)
    df["business_net_losses"] = -puf.E00900 * (puf.E00900 < 0)
    df["qualified_business_income_deduction"] = puf.QBIDED
    df["taxable_income"] = puf.C04800
    df["is_tax_filer"] = puf.DATA_SOURCE == 1
    df["is_taxable"] = puf.C09200 - puf.REFUND > 0
    df["count"] = 1
    df["filing_status"] = puf.MARS.map(
        {
            0: "SINGLE",  # Assume the aggregate record is single
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    df["weight"] = puf["S006"]
    return df
