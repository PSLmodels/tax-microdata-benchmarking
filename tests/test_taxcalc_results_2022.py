"""
Test that 2022 Tax-Calculator results for EITC and ACTC match
IRS Statistics of Income (SOI) Publication 1304 aggregates.

Comparisons (all tax units, year 2022):
  - Earned Income Credit (EITC): count (millions) and amount (billions)
  - Additional Child Tax Credit (ACTC): count (millions) and amount (billions)
"""

import yaml
import pytest
import taxcalc

from tmd.imputation_assumptions import (
    TAXYEAR,
    SOI_IITAX_SPEC,
    CLAIM_PROB_SCALE_EXPECTED,
)

MAX_RELATIVE_TOLERANCE = {
    "n_returns_mil": 0.088,
    "amount_bil": 0.003,
}


def test_claim_prob_scale_defaults_unchanged():
    """Guard: Tax-Calculator defaults still equal the calibrated values.

    TMD's TAXYEAR EITC/ACTC calibration assumes the Tax-Calculator
    eitc_claim_prob_scale/actc_claim_prob_scale defaults equal the values in
    CLAIM_PROB_SCALE_EXPECTED, so TMD no longer sets them as a reform.  This
    test fails loudly if a future Tax-Calculator release changes those
    upstream defaults and silently de-calibrates TMD.
    """
    pol = taxcalc.Policy()
    pol.set_year(TAXYEAR)
    errors = []
    for name, expected in CLAIM_PROB_SCALE_EXPECTED.items():
        actual = float(getattr(pol, name)[0])
        if actual != expected:
            errors.append(f"{name}: taxcalc={actual} expected={expected}")
    if errors:
        raise ValueError(
            "Tax-Calculator claim_prob_scale defaults have changed; "
            "TMD calibration may need updating:\n" + "\n".join(errors)
        )


@pytest.mark.skipif(
    TAXYEAR != 2022,
    reason="expected values are calibrated to TAXYEAR=2022",
)
def test_taxcalc_results_2022(
    tests_folder, tmd_variables, tmd_weights_path, tmd_growfactors_path
):
    epath = tests_folder / "expected_taxcalc_results_2022.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        expect = yaml.safe_load(f)

    pol = taxcalc.Policy()
    pol.implement_reform(SOI_IITAX_SPEC)
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

    wt = sim.array("s006")
    eitc = sim.array("eitc")
    actc = sim.array("c11070")

    actual = {
        "eitc": {
            "n_returns_mil": float((wt * (eitc > 0)).sum() * 1e-6),
            "amount_bil": float((wt * eitc).sum() * 1e-9),
        },
        "actc": {
            "n_returns_mil": float((wt * (actc > 0)).sum() * 1e-6),
            "amount_bil": float((wt * actc).sum() * 1e-9),
        },
    }

    errors = []
    for credit in ("eitc", "actc"):
        for metric in ("n_returns_mil", "amount_bil"):
            act = actual[credit][metric]
            exp = expect[credit][metric]
            rel_diff = act / exp - 1.0
            if abs(rel_diff) >= MAX_RELATIVE_TOLERANCE[metric]:
                errors.append(
                    f"{credit}.{metric}: "
                    f"act={act:.2f} exp={exp:.2f} "
                    f"rel_diff={rel_diff:+.4f} "
                    f"tol={MAX_RELATIVE_TOLERANCE[metric]:.4f}"
                )

    if errors:
        raise ValueError(
            "\nACT-vs-EXP TAXCALC 2022 DIFFERENCES:\n" + "\n".join(errors)
        )
