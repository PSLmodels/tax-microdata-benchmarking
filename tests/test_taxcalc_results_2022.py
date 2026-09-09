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
from tmd.storage import (
    POLICY_GROWFACTORS_PATH,
    TMD_GROWFACTORS_PATH,
    TMD_WEIGHTS_PATH,
)
from tmd.imputation_assumptions import TAXYEAR, SOI_IITAX_SPEC

MAX_RELATIVE_TOLERANCE = {
    "n_returns_mil": 0.010,
    "amount_bil": 0.005,
}


@pytest.mark.skipif(
    TAXYEAR != 2022,
    reason="expected values are calibrated to TAXYEAR=2022",
)
def test_taxcalc_results_2022(tests_folder, tmd_variables):
    epath = tests_folder / "expected_taxcalc_results_2022.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        expect = yaml.safe_load(f)

    policy_gf = taxcalc.GrowFactors(
        growfactors_filename=str(POLICY_GROWFACTORS_PATH)
    )
    pol = taxcalc.Policy(gfactor=policy_gf)
    pol.implement_reform(SOI_IITAX_SPEC)
    recs = taxcalc.Records(
        data=tmd_variables,
        start_year=TAXYEAR,
        weights=str(TMD_WEIGHTS_PATH),
        gfactors=taxcalc.GrowFactors(
            growfactors_filename=str(TMD_GROWFACTORS_PATH)
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
