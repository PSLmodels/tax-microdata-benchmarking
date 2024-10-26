"""
Tests of tmd/areas/create_area_weights.py script.
"""

import yaml
import pandas as pd
import taxcalc as tc
from tmd.storage import STORAGE_FOLDER
from tmd.areas import AREAS_FOLDER
from tmd.areas.create_area_weights import create_area_weights_file

YEAR = 2021


def test_area_xx(tests_folder):
    """
    Optimize national weights for faux xx area using the faux xx area targets
    and compare actual Tax-Calculator results with expected results when
    using area weights along with national input data and growfactors.
    """
    rc = create_area_weights_file("xx", write_log=True, write_file=True)
    assert rc == 0, "create_areas_weights_file has non-zero return code"
    # compare actual vs expected results for faux area xx
    # ... instantiate Tax-Calculator object for {area}
    idpath = STORAGE_FOLDER / "output" / "tmd.csv.gz"
    gfpath = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"
    wtpath = AREAS_FOLDER / "weights" / "xx_tmd_weights.csv.gz"
    input_data = tc.Records(
        data=pd.read_csv(idpath),
        start_year=YEAR,
        gfactors=tc.GrowFactors(growfactors_filename=str(gfpath)),
        weights=str(wtpath),
        adjust_ratios=None,
        exact_calculations=True,
    )
    sim = tc.Calculator(records=input_data, policy=tc.Policy())
    # ... calculate tax variables for YEAR
    sim.advance_to_year(YEAR)
    sim.calc_all()
    vdf = sim.dataframe([], all_vars=True)
    # ... calculate actual results and store in act dictionary
    puf = vdf.data_source == 1
    wght = vdf.s006 * puf
    act = {
        "popall": (vdf.s006 * vdf.XTOT).sum() * 1e-6,
        "e00300": (wght * vdf.e00300[puf]).sum() * 1e-9,
        "e00900": (wght * vdf.e00900[puf]).sum() * 1e-9,
        "e00200": (wght * vdf.e00200[puf]).sum() * 1e-9,
        "e02000": (wght * vdf.e02000[puf]).sum() * 1e-9,
        "e02400": (wght * vdf.e02400[puf]).sum() * 1e-9,
        "c00100": (wght * vdf.c00100[puf]).sum() * 1e-9,
        "agihic": (wght * (vdf.c00100 >= 1e6)).sum() * 1e-3,
        "e00400": (wght * vdf.e00400[puf]).sum() * 1e-9,
        "e00600": (wght * vdf.e00600[puf]).sum() * 1e-9,
        "e00650": (wght * vdf.e00650[puf]).sum() * 1e-9,
        "e01700": (wght * vdf.e01700[puf]).sum() * 1e-9,
        "e02300": (wght * vdf.e02300[puf]).sum() * 1e-9,
        "e17500": (wght * vdf.e17500[puf]).sum() * 1e-9,
        "e18400": (wght * vdf.e18400[puf]).sum() * 1e-9,
        "e18500": (wght * vdf.e18500[puf]).sum() * 1e-9,
    }
    # ... read expected results into exp dictionary
    exp_path = tests_folder / "test_area_weights_expect.yaml"
    with open(exp_path, "r", encoding="utf-8") as efile:
        exp = yaml.safe_load(efile.read())
    # compare actual with expected results
    default_rtol = 0.005
    rtol = {
        # "res": 0.011,
    }
    if set(act.keys()) != set(exp.keys()):
        print("sorted(act.keys())=", sorted(act.keys()))
        print("sorted(exp.keys())=", sorted(exp.keys()))
        raise ValueError("act.keys() != exp.keys()")
    emsg = ""
    for res in exp.keys():
        reldiff = act[res] / exp[res] - 1
        reltol = rtol.get(res, default_rtol)
        ok = abs(reldiff) < reltol
        if not ok:
            emsg += (
                f"FAIL:res,act,exp,rdiff,rtol= {res} {act[res]:.5f}"
                f" {exp[res]:.5f} {reldiff:.4f} {reltol:.4f}\n"
            )
    if emsg:
        print(emsg)
        raise ValueError("ACT vs EXP diffs in test_area_weights")
