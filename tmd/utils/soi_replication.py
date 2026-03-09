import pandas as pd
import taxcalc as tc


def tc_to_soi(puf: pd.DataFrame, year: int) -> pd.DataFrame:
    pol = tc.Policy()
    rec = tc.Records(
        data=puf,
        start_year=year,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
        exact_calculations=True,
    )
    calculator = tc.Calculator(policy=pol, records=rec)
    calculator.calc_all()
    pdf = calculator.dataframe(None, all_vars=True)

    pdf.columns = pdf.columns.str.upper()

    df = pd.DataFrame()
    df["adjusted_gross_income"] = pdf.C00100
    df["total_income_tax"] = pdf.C09200 - pdf.REFUND
    df["employment_income"] = pdf.E00200
    df["capital_gains_distributions"] = pdf.E01100
    df["capital_gains_gross"] = pdf["C01000"] * (pdf["C01000"] > 0)
    df["capital_gains_losses"] = -pdf["C01000"] * (pdf["C01000"] < 0)
    df["estate_income"] = 0  # pdf.E26390 (not in T-C)
    df["estate_losses"] = 0  # pdf.E26400 (not in T-C)
    df["exempt_interest"] = pdf.E00400
    df["ira_distributions"] = pdf.E01400
    df["count_of_exemptions"] = pdf.XTOT
    df["ordinary_dividends"] = pdf.E00600
    df["partnership_and_s_corp_income"] = pdf.E26270 * (pdf.E26270 > 0)
    df["partnership_and_s_corp_losses"] = -pdf.E26270 * (pdf.E26270 < 0)
    df["total_pension_income"] = pdf.E01500
    df["taxable_pension_income"] = pdf.E01700
    df["qualified_dividends"] = pdf.E00650
    df["rent_and_royalty_net_income"] = 0  # pdf.E25850 (not in T-C)
    df["rent_and_royalty_net_losses"] = 0  # pdf.E25860 (not in T-C)
    df["total_social_security"] = pdf.E02400
    df["taxable_social_security"] = pdf.C02500
    df["income_tax_before_credits"] = pdf.C05800
    df["taxable_interest_income"] = pdf.E00300
    df["unemployment_compensation"] = pdf.E02300
    df["charitable_contributions_deduction"] = pdf.C19700
    df["interest_paid_deductions"] = pdf.E19200
    df["medical_expense_deductions_uncapped"] = pdf.E17500
    df["itemized_state_income_and_sales_tax_deductions"] = pdf.E18400
    df["itemized_real_estate_tax_deductions"] = pdf.E18500
    df["state_and_local_tax_deductions"] = pdf.E18400 + pdf.E18500
    df["income_tax_after_credits"] = pdf.IITAX
    df["business_net_profits"] = pdf.E00900 * (pdf.E00900 > 0)
    df["business_net_losses"] = -pdf.E00900 * (pdf.E00900 < 0)
    df["qualified_business_income_deduction"] = pdf.QBIDED
    df["taxable_income"] = pdf.C04800
    df["is_tax_filer"] = pdf.DATA_SOURCE == 1
    df["is_taxable"] = pdf.C09200 - pdf.REFUND > 0
    df["count"] = 1
    df["filing_status"] = pdf.MARS.map(
        {
            0: "SINGLE",  # Assume the aggregate record is single
            1: "SINGLE",
            2: "JOINT",
            3: "SEPARATE",
            4: "HEAD_OF_HOUSEHOLD",
        }
    )
    df["weight"] = pdf["S006"]
    return df
