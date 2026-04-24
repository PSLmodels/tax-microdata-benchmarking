"""
Sanity checks that the 2022 TMD data aligns with externally published
Statistics of Income (SOI) figures for the same year.

Five weighted totals are checked against SOI targets read from
``tmd/storage/input/soi.csv``, all with a 1% relative tolerance:

  - Number of filers : Σ s006 on PUF records
                       vs SOI ``count``
  - Wages            : Σ e00200 × s006 on PUF records
                       vs SOI ``employment_income``
  - AGI              : Σ c00100 × s006 on PUF records
                       vs SOI ``adjusted_gross_income``
  - Total income tax : Σ iitax × s006
                       vs SOI ``tottax``
  - Partnership and  : Σ e26270 × s006
    S-corp net income  vs SOI ``partnership_and_s_corp_income``
                          minus ``partnership_and_s_corp_losses``

The first three checks are restricted to the PUF subsample (tax
filers) using ``data_source == 1``; the SOI "All filers, full
population" rows are the concept-aligned targets. CPS-only records in
TMD represent the non-filer population and are excluded.

For the income-tax row, the comparator is SOI ``tottax`` ("Total
income tax" = income tax after credits + NIIT + tax on accumulation
distribution of trusts, per Publication 1304). This aligns with
TaxCalc's ``iitax`` (= ``c09200 - refund``) in scope — both include
NIIT, AMT, and other additional taxes except self-employment tax and
Additional Medicare. The observed 2022 gap is about 0.35%; a looser
``income_tax_after_credits`` comparator would show ~2.3% gap simply
because it excludes NIIT. No PUF-only mask is needed for iitax since
CPS records contribute zero to it.

For the partnership-and-S-corp row, ``e26270`` is the net amount
(income minus losses, line 17 of Form 1040 Schedule E). SOI reports
income and losses separately, so the target is the difference of the
two SOI variables. Same as iitax, no PUF mask is needed because CPS
records contribute zero to ``e26270``.

Replaces the external-benchmark role of the retired
``test_variable_totals.py``, subsumes the retired
``test_misc.py::test_income_tax``, and absorbs the partnership /
S-corp check that was previously in ``test_misc.py`` with a loose
10% tolerance against an unattributed $975 B target.
"""

import pandas as pd
import taxcalc

from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import TAXYEAR, CREDIT_CLAIMING

SOI_YEAR = 2022
RELATIVE_TOLERANCE = 0.01
SOI_CSV = STORAGE_FOLDER / "input" / "soi.csv"


def _soi_target(soi_df, variable, count):
    """Return the "All filers, full population" SOI value for one variable.

    ``count=True`` retrieves the number-of-returns row, ``count=False``
    retrieves the dollar-amount row. Asserts that exactly one row
    matches, so silently-wrong filters fail loudly.
    """
    mask = (
        (soi_df["Year"] == SOI_YEAR)
        & (soi_df["Variable"] == variable)
        & (soi_df["Filing status"] == "All")
        & (soi_df["AGI lower bound"] == float("-inf"))
        & (soi_df["AGI upper bound"] == float("inf"))
        & ~soi_df["Taxable only"]
        & soi_df["Full population"]
        & (soi_df["Count"] == count)
    )
    rows = soi_df[mask]
    assert len(rows) == 1, (
        f"expected exactly one SOI row for variable={variable!r} "
        f"count={count}, found {len(rows)}"
    )
    return float(rows["Value"].iloc[0])


def test_soi_sanity_2022(
    tmd_variables, tmd_weights_path, tmd_growfactors_path
):
    """Five weighted 2022 totals from TMD within 1% of SOI targets."""
    # Run TaxCalc at TAXYEAR to obtain c00100 (AGI) and iitax. e00200,
    # e26270, and s006 are available directly on the input frame but
    # using the post-calc_all arrays keeps one consistent source for
    # all five aggregates.
    pol = taxcalc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    recs = taxcalc.Records(
        data=tmd_variables,
        start_year=TAXYEAR,
        weights=str(tmd_weights_path),
        gfactors=taxcalc.GrowFactors(
            growfactors_filename=str(tmd_growfactors_path)
        ),
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    sim = taxcalc.Calculator(policy=pol, records=recs)
    sim.advance_to_year(TAXYEAR)
    sim.calc_all()

    wght = sim.array("s006")
    e00200 = sim.array("e00200")
    c00100 = sim.array("c00100")
    iitax = sim.array("iitax")
    e26270 = sim.array("e26270")
    puf_mask = sim.array("data_source") == 1

    tmd_filers = float((wght * puf_mask).sum())
    tmd_wages = float((wght * e00200 * puf_mask).sum())
    tmd_agi = float((wght * c00100 * puf_mask).sum())
    # No PUF mask for iitax or e26270: CPS records contribute zero to
    # both, so filtering has no numerical effect. Keeping simpler form.
    tmd_tottax = float((wght * iitax).sum())
    tmd_pship_scorp = float((wght * e26270).sum())

    soi_df = pd.read_csv(SOI_CSV)
    soi_filers = _soi_target(soi_df, "count", count=True)
    soi_wages = _soi_target(soi_df, "employment_income", count=False)
    soi_agi = _soi_target(soi_df, "adjusted_gross_income", count=False)
    soi_tottax = _soi_target(soi_df, "tottax", count=False)
    # SOI reports partnership/S-corp income and losses separately;
    # target is the net (income minus losses), matching e26270.
    soi_pship_scorp_net = _soi_target(
        soi_df, "partnership_and_s_corp_income", count=False
    ) - _soi_target(soi_df, "partnership_and_s_corp_losses", count=False)

    errors = []
    for name, tmd_val, soi_val in [
        ("filers", tmd_filers, soi_filers),
        ("wages (e00200)", tmd_wages, soi_wages),
        ("AGI (c00100)", tmd_agi, soi_agi),
        ("income tax (iitax vs SOI tottax)", tmd_tottax, soi_tottax),
        (
            "partnership / S-corp net (e26270)",
            tmd_pship_scorp,
            soi_pship_scorp_net,
        ),
    ]:
        rel_diff = abs(tmd_val / soi_val - 1.0)
        if rel_diff >= RELATIVE_TOLERANCE:
            errors.append(
                f"{name}: TMD={tmd_val:,.0f} SOI={soi_val:,.0f} "
                f"rel_diff={rel_diff:.4f} "
                f"(tol={RELATIVE_TOLERANCE:.2f})"
            )

    assert not errors, "2022 SOI sanity-check failures:\n" + "\n".join(errors)
