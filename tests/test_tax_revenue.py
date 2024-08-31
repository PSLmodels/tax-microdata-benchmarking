"""
Tests of income/payroll tax revenues generated by the tmd files.
"""

import yaml
import taxcalc as tc


FIRST_CYR = 2022
LAST_CYR = 2033

RELTOL_ITAX = 0.20
RELTOL_PTAX = 0.09


DUMP = False  # True implies test always fails with complete output


def fy2cy(fy1, fy2):
    return fy1 + 0.25 * (fy2 - fy1)


def test_tax_revenue(
    tests_folder, tmd_variables, tmd_weights_path, tmd_growfactors_path
):
    # read expected fiscal year revenues and convert to calendar year revenues
    with open(tests_folder / "expected_itax_revenue.yaml") as f:
        fy_itax = yaml.safe_load(f)
    with open(tests_folder / "expected_ptax_revenue.yaml") as f:
        fy_ptax = yaml.safe_load(f)
    exp_itax = {}
    exp_ptax = {}
    for year in range(FIRST_CYR, LAST_CYR + 1):
        exp_itax[year] = round(fy2cy(fy_itax[year], fy_itax[year + 1]), 3)
        exp_ptax[year] = round(fy2cy(fy_ptax[year], fy_ptax[year + 1]), 3)
    # calculate actual tax revenues for each calendar year
    wghts = str(tmd_weights_path)
    growf = tc.GrowFactors(growfactors_filename=str(tmd_growfactors_path))
    input_data = tc.Records(
        data=tmd_variables,
        start_year=2021,
        weights=wghts,
        gfactors=growf,
        adjust_ratios=None,
        exact_calculations=True,
    )
    sim = tc.Calculator(records=input_data, policy=tc.Policy())
    act_itax = {}
    act_ptax = {}
    for year in range(FIRST_CYR, LAST_CYR + 1):
        sim.advance_to_year(year)
        sim.calc_all()
        wght = sim.array("s006")
        itax = sim.array("iitax")  # includes refundable credit amounts
        refc = sim.array("refund")  # refundable credits considered expenditure
        itax_cbo = itax + refc  # itax revenue comparable to CBO estimates
        act_itax[year] = (wght * itax_cbo).sum() * 1e-9
        act_ptax[year] = (wght * sim.array("payrolltax")).sum() * 1e-9
    # compare actual vs expected tax revenues in each calendar year
    emsg = ""
    for year in range(FIRST_CYR, LAST_CYR + 1):
        reldiff = act_itax[year] / exp_itax[year] - 1
        same = abs(reldiff) < RELTOL_ITAX
        if not same or DUMP:
            msg = (
                f"\nITAX:cyr,act,exp,rdiff= {year} "
                f"{act_itax[year]:9.3f} {exp_itax[year]:9.3f} {reldiff:7.4f}"
            )
            emsg += msg
        reldiff = act_ptax[year] / exp_ptax[year] - 1
        same = abs(reldiff) < RELTOL_PTAX
        if not same or DUMP:
            msg = (
                f"\nPTAX:cyr,act,exp,rdiff= {year} "
                f"{act_ptax[year]:9.3f} {exp_ptax[year]:9.3f} {reldiff:7.4f}"
            )
            emsg += msg
    if DUMP:
        assert False, f"test_tax_revenue DUMP output: {emsg}"
    else:
        if emsg:
            emsg += f"\nRELTOL_ITAX= {RELTOL_ITAX:4.2f}"
            emsg += f"\nRELTOL_PTAX= {RELTOL_PTAX:4.2f}"
            raise ValueError(emsg)
