"""
Sanity-check that weighted PUF (filer) totals on the TMD file match CBO's
individual income tax microsimulation projections at four anchor years.

Tested aggregates: number of returns, AGI (`c00100`), and individual
income tax liability (`iitax`).

Comparator is the calendar-year 1040-universe liability series in
CBO's February 2026 Revenue file, sheet 3 — built from the same SOI 2022
PUF the TMD file uses. See `tests/expected_cbo_levels_2022_data.yaml`
for the exact CBO line items and the data source.

Tolerances widen with the projection horizon to reflect compounding
growfactor uncertainty.
"""

import yaml
import pytest
import taxcalc

from tmd.imputation_assumptions import TAXYEAR, CREDIT_CLAIMING

# Per-year relative tolerance for all three aggregates.
RELTOL = {
    2022: 0.01,
    2026: 0.02,
    2031: 0.05,
    2036: 0.06,
}

VARIABLES = ("n_returns_mil", "agi_bil", "iitax_bil")

DUMP = False  # if True, always fail with full output table


@pytest.mark.skipif(
    TAXYEAR != 2022,
    reason="expected values are calibrated to TAXYEAR=2022",
)
def test_revenue_levels_cbo(
    tests_folder, tmd_variables, tmd_weights_path, tmd_growfactors_path
):
    epath = tests_folder / "expected_cbo_levels_2022_data.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        exp = yaml.safe_load(f)

    pol = taxcalc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    rec = taxcalc.Records(
        data=tmd_variables,
        start_year=TAXYEAR,
        gfactors=taxcalc.GrowFactors(
            growfactors_filename=str(tmd_growfactors_path)
        ),
        weights=str(tmd_weights_path),
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    sim = taxcalc.Calculator(policy=pol, records=rec)

    actual = {}
    for year in sorted(RELTOL):
        sim.advance_to_year(year)
        sim.calc_all()
        wt = sim.array("s006")
        puf = sim.array("data_source") == 1
        wpuf = wt * puf
        actual[year] = {
            "n_returns_mil": wpuf.sum() * 1e-6,
            "agi_bil": (wpuf * sim.array("c00100")).sum() * 1e-9,
            "iitax_bil": (wpuf * sim.array("iitax")).sum() * 1e-9,
        }

    emsg = ""
    for year, tol in RELTOL.items():
        for var in VARIABLES:
            act = actual[year][var]
            ex = exp[year][var]
            reldiff = act / ex - 1.0
            if abs(reldiff) >= tol or DUMP:
                emsg += (
                    f"\n{var:<14s} year={year} "
                    f"act={act:10.3f} exp={ex:10.3f} "
                    f"reldiff={reldiff:+7.4f} tol={tol:.3f}"
                )

    if DUMP:
        assert False, f"test_revenue_levels_cbo DUMP output: {emsg}"
    if emsg:
        raise ValueError(emsg)
