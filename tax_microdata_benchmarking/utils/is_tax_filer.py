def is_tax_filer(
    irs_gross_income: float,
    filing_status: str,
    earned_income: float,
    total_income_tax: float,
    aged_blind_count: int,
    standard_deduction: float,
    aged_blind_standard_deduction: float,
    exemption_amount: float,
) -> bool:
    """
    Determine whether a tax unit is required to file taxes.

    Args:
        irs_gross_income: Gross income as defined by the IRS.
        filing_status: Filing status of the tax unit.
        earned_income: Total earned income.
        total_income_tax: Total income tax liability.
        aged_blind_count: Number of aged or blind individuals in the tax unit.
        standard_deduction: Standard deduction for the tax unit.
        aged_blind_standard_deduction: Additional standard deduction for aged or blind individuals.
        exemption_amount: Exemption amount for the tax unit.

    Returns:
        bool: Whether the tax unit is required to file taxes.
    """

    # (a)(1)(A), (a)(1)(B)

    separate = filing_status == "SEPARATE"
    threshold = (
        exemption_amount if separate else standard_deduction + exemption_amount
    )
    income_over_exemption_amount = irs_gross_income > threshold

    # (a)(1)(C)

    unearned_income_threshold = 500 + aged_blind_standard_deduction[filing_status] * aged_blind_count
    unearned_income = irs_gross_income - earned_income
    unearned_income_over_threshold = (
        unearned_income > unearned_income_threshold
    )

    required_to_file = (
        income_over_exemption_amount | unearned_income_over_threshold
    )

    tax_refund = total_income_tax < 0
    not_required_but_likely_filer = not required_to_file and tax_refund

    # (a)(1)(D) is just definitions

    return required_to_file or not_required_but_likely_filer

from policyengine_us.system import system

parameters = system.parameters.gov.irs

aged_blind_standard_deduction = parameters.deductions.standard.aged_or_blind.amount