"""
Compare weighted TMD aggregate individual income tax revenue against CBO's
individual income tax microsimulation projections for each calendar year
in the 2022 through 2035 range.

Expected values are the calendar-year 1040-universe liability series in
CBO's February 2026 Revenue file, sheet 3, read from
`tests/expected_cbo_revenues.yaml` (see that file for the exact CBO line
item and the data source).  Actual values are weighted totals of the
TaxCalc `iitax` variable over the filer (PUF / data_source==1) records on
the TMD file, which is the same universe the CBO series covers and the
same universe used by tests/test_revenue_levels_cbo.py.

Unlike test_revenue_levels_cbo.py, this test applies no per-year
tolerance.  Instead it writes the full year-by-year comparison to
`tests/actual_tmd_revenues` and requires that file to be identical to the
version-controlled `tests/expect_tmd_revenues` file.  Any change in the
TMD file, its weights, or its growfactors that moves annual revenue will
therefore fail this test, showing exactly which years moved and by how
much.  When the change is intended, copy `tests/actual_tmd_revenues` over
`tests/expect_tmd_revenues` and commit the difference.

The written `tests/actual_tmd_revenues` file is removed when the test
passes, and is left in place when the test fails so that it can be
inspected and, if the new results are correct, copied.
"""

import difflib
import yaml
import pytest
import taxcalc
from tmd.imputation_assumptions import TAXYEAR, SOI_IITAX_SPEC

FIRST_YEAR = TAXYEAR
LAST_YEAR = 2035

ACTUAL_FILENAME = "actual_tmd_revenues"
EXPECT_FILENAME = "expect_tmd_revenues"


def revenue_table(actual, expect):
    """
    Return, as a list of text lines, the year-by-year comparison of the
    actual and expected income tax revenue dictionaries, followed by the
    sum over years of the squared percentage differences.
    """
    lines = [
        "TMD vs CBO aggregate individual income tax revenue",
        "(revenue in billions of calendar-year dollars)",
        "",
        "YEAR" + "ACTUAL".rjust(11) + "EXPECT".rjust(11) + "PCTDIFF".rjust(9),
    ]
    sum_sq_pctdiff = 0.0
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        act = actual[year]
        exp = expect[year]
        pctdiff = 100.0 * (act / exp - 1.0)
        sum_sq_pctdiff += pctdiff * pctdiff
        lines.append(f"{year:4d} {act:10.1f} {exp:10.1f} {pctdiff:+8.2f}")
    lines.append("")
    lines.append(
        f"sum over {FIRST_YEAR}-{LAST_YEAR} of squared PCTDIFF: "
        f"iitax = {sum_sq_pctdiff:.2f}"
    )
    return lines


@pytest.mark.skipif(
    TAXYEAR != 2022,
    reason="expected values are calibrated to TAXYEAR=2022",
)
def test_tmd_revenues(
    tests_folder,
    tmd_variables,
    tmd_weights_path,
    tmd_growfactors_path,
    policy_growfactors_path,
):
    # read expected annual income tax revenue
    epath = tests_folder / "expected_cbo_revenues.yaml"
    with open(epath, "r", encoding="utf-8") as f:
        edict = yaml.safe_load(f)
    expect = {}
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        assert year in edict, f"{epath.name} has no {year} data"
        expect[year] = edict[year]["iitax_bil"]

    # calculate actual annual income tax revenue on the TMD file
    policy_gf = taxcalc.GrowFactors(
        growfactors_filename=str(policy_growfactors_path)
    )
    pol = taxcalc.Policy(gfactor=policy_gf)
    pol.implement_reform(SOI_IITAX_SPEC)
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
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        sim.advance_to_year(year)
        sim.calc_all()
        wght = sim.array("s006") * (sim.array("data_source") == 1)
        actual[year] = (wght * sim.array("iitax")).sum() * 1e-9

    # write actual output file
    apath = tests_folder / ACTUAL_FILENAME
    atext = "\n".join(revenue_table(actual, expect)) + "\n"
    with open(apath, "w", encoding="utf-8") as f:
        f.write(atext)

    # compare actual output file with expected output file
    xpath = tests_folder / EXPECT_FILENAME
    if not xpath.exists():
        raise ValueError(
            f"tests/{EXPECT_FILENAME} file does not exist:\n"
            f"  if tests/{ACTUAL_FILENAME} contains correct results,\n"
            f"  copy it to tests/{EXPECT_FILENAME} and add it to git"
        )
    with open(xpath, "r", encoding="utf-8") as f:
        xtext = f.read()
    if atext != xtext:
        diffs = difflib.unified_diff(
            xtext.splitlines(keepends=True),
            atext.splitlines(keepends=True),
            fromfile=f"tests/{EXPECT_FILENAME}",
            tofile=f"tests/{ACTUAL_FILENAME}",
            n=0,
        )
        raise ValueError(
            f"tests/{ACTUAL_FILENAME} differs from "
            f"tests/{EXPECT_FILENAME}:\n" + "".join(diffs)
        )

    # remove actual output file because it is identical to expected file
    apath.unlink()
