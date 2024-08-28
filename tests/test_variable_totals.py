"""
This module adds basic tests for several tmd file variables,
checking that the variable total is within the ballpark of
the taxdata 2021 PUF's totals.
"""

import yaml
import pytest
from tmd.storage import STORAGE_FOLDER


@pytest.mark.vartotals
def test_variable_totals(tests_folder, tmd_variables):
    with open(STORAGE_FOLDER / "input" / "tc_variable_metadata.yaml") as f:
        tc_variable_metadata = yaml.safe_load(f)
    with open(tests_folder / "taxdata_variable_totals.yaml") as f:
        td_variable_totals = yaml.safe_load(f)
    test_exempted_variables = [
        "DSI",  # Issue here but deprioritized
        "EIC",  # PUF-PE file more correct by including CPS data
        "MIDR",  # Issue here but deprioritized
        "RECID",  # No reason to compare.
        "a_lineno",  # No reason to compare.
        "agi_bin",  # No reason to compare.
        "blind_spouse",  # Issue here but deprioritized.
        "cmbtp",  # No reason to compare.
        "data_source",  # No reason to compare.
        "s006",  # No reason to compare.
        "h_seq",  # No reason to compare.
        "fips",  # No reason to compare.
        "ffpos",  # No reason to compare.
        "p22250",  # PE-PUF closer to truth than taxdata (needs checking).
        "p23250",  # PE-PUF closer to truth than taxdata (needs checking).
        "e01200",  # Unknown but deprioritized for now.
        "e17500",  # Unknown but deprioritized for now.
        "e18500",  # Unknown but deprioritized for now.
        "e02100",  # Farm income, unsure who's closer.
        "e02300",  # UI exploded in 2021
        "e02400",  # SS benefits, TD is out
        "e18400",
        "e19200",
        "e20100",
    ]
    # also exempt any variable split between head and spouse
    test_exempted_variables += [
        variable
        for variable in tc_variable_metadata["read"]
        if variable.endswith("p") or variable.endswith("s")
    ]
    variables_to_test = [
        variable
        for variable in td_variable_totals.keys()
        if variable not in test_exempted_variables
    ]
    weight = tmd_variables.s006
    puf_record = tmd_variables.data_source == 1
    emsg = ""
    for var in variables_to_test:
        meta = tc_variable_metadata["read"][var]
        name = meta.get("desc")
        total = (tmd_variables[var] * weight * puf_record).sum()
        if td_variable_totals[var] == 0:
            # if the variable_totals file has a zero total,
            # assume the tmd_variables dataframe is correct
            continue
        # 45% and more than $30bn off taxdata is a failure
        ok = (
            abs(total / td_variable_totals[var] - 1) < 0.45
            or abs(total / 1e9 - td_variable_totals[var] / 1e9) < 30
        )
        if not ok:
            msg = (
                f"\n{var} ({name}) differs from expected by "
                f"{(total / td_variable_totals[var] - 1):.1%} : "
                f"or $B {(total / 1e9):.1f} vs "
                f"{(td_variable_totals[var] / 1e9):.1f}"
            )
            emsg += msg
    if emsg:
        raise ValueError(emsg)
