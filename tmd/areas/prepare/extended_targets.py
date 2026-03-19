"""
Extended targets — build additional SOI-shared, Census-shared, and
aggregate credit targets for area weighting.

These targets supplement the base recipe targets with variables that
use different geographic distribution sources:

  1. **SOI-shared by AGI stub**: TMD national sum × (state SOI / US SOI)
     for variables where TMD and SOI definitions align well.

  2. **Census-shared by AGI stub**: TMD national sum × Census tax share,
     distributed across AGI stubs by SOI bin proportions.  Used for SALT
     variables where Census provides better geographic distribution.

  3. **SOI aggregate**: One amount + one count target per state (no AGI
     breakdown) for credits like EITC and CTC.

All stub-level targets are restricted to high-income AGI stubs (default:
$50K+, stubs 5–10) to avoid noisy low-income bins.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from tmd.areas.prepare.constants import (
    ALL_STATES,
    STATE_AGI_CUTS,
    AreaType,
    SOI_STATE_CSV_PATTERNS,
    build_agi_labels,
)

# --- Extended target specs ---

# SOI-shared by AGI stub: (tmd_varname, soi_amount_varname)
SOI_SHARED_SPECS: List[Tuple[str, str]] = [
    # c18300 is NOT here — it's already in the base recipe with full
    # AGI stub coverage (stubs 3-10), so adding it here would duplicate.
    ("e01700", "a01700"),  # Taxable pensions
    ("c02500", "a02500"),  # Taxable Social Security
    ("e01400", "a01400"),  # Taxable IRA distributions
    ("capgains_net", "a01000"),  # Net capital gains (p22250+p23250)
    ("e00600", "a00600"),  # Ordinary dividends
    ("e00900", "a00900"),  # Business/professional net income
    ("c19200", "a19300"),  # Interest deduction (mortgage dominates)
    ("c19700", "a19700"),  # Charitable contributions deduction
]

# SOI aggregate (one target per state, no AGI breakdown):
SOI_AGGREGATE_SPECS: List[Tuple[str, str]] = [
    ("eitc", "a59660"),  # Earned Income Tax Credit
    ("ctc_total", "a_ctc_total"),  # Total CTC (a07225 + a11070)
]

# Census-shared by AGI stub: (tmd_varname, census_type)
CENSUS_SHARED_SPECS: List[Tuple[str, str]] = [
    ("e18400", "combined"),  # SALT income/sales → Census property+sales
    ("e18500", "property"),  # SALT real estate → Census property only
]

# Default AGI stubs for extended targets ($50K+)
DEFAULT_EXTENDED_STUBS = [5, 6, 7, 8, 9, 10]

# --- Census data loading ---

_CENSUS_DATA_PATH = (
    Path(__file__).parent / "data" / "census_2022_state_local_finance.xlsx"
)

_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def load_census_shares(
    census_path: Path = _CENSUS_DATA_PATH,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Load Census state/local finance data and compute state tax shares.

    Returns
    -------
    (combined_shares, property_shares) : tuple of dicts
        combined_shares: state → (property + sales) / US total
        property_shares: state → property / US property total
    """
    cdf = pd.read_excel(census_path, sheet_name="2022_US_WY", header=None)
    sl_cols = {}
    seen = set()
    for col_idx in range(2, len(cdf.columns)):
        n = (
            str(cdf.iloc[8, col_idx]).strip()
            if pd.notna(cdf.iloc[8, col_idx])
            else ""
        )
        gt = (
            str(cdf.iloc[9, col_idx]).strip()
            if pd.notna(cdf.iloc[9, col_idx])
            else ""
        )
        at = (
            str(cdf.iloc[11, col_idx]).strip()
            if pd.notna(cdf.iloc[11, col_idx])
            else ""
        )
        if n and gt == "State & local" and at == "amount1" and n not in seen:
            seen.add(n)
            sl_cols[n] = col_idx

    us_col = sl_cols["United States Total"]
    us_property = float(cdf.iloc[25, us_col])
    us_sales = float(cdf.iloc[27, us_col])

    combined_shares = {}
    property_shares = {}
    for name, col in sl_cols.items():
        abbr = _NAME_TO_ABBR.get(name)
        if abbr is None:
            continue
        pt = float(cdf.iloc[25, col]) if pd.notna(cdf.iloc[25, col]) else 0
        gs = float(cdf.iloc[27, col]) if pd.notna(cdf.iloc[27, col]) else 0
        combined_shares[abbr] = (pt + gs) / (us_property + us_sales)
        property_shares[abbr] = pt / us_property

    return combined_shares, property_shares


# --- SOI data loading ---


def _load_soi_by_stub(soi_year: int) -> pd.DataFrame:
    """
    Load raw SOI state CSV for the given year.

    Returns DataFrame with lowercase columns, filtered to agi_stub > 0,
    with a 'stabbr' column (uppercase state abbreviation).
    """
    soi_raw_dir = Path(__file__).parent / "data" / "soi_states"
    fname = SOI_STATE_CSV_PATTERNS.get(soi_year)
    if fname is None:
        raise ValueError(
            f"No SOI state CSV for year {soi_year}. "
            f"Available: {sorted(SOI_STATE_CSV_PATTERNS.keys())}"
        )
    soi = pd.read_csv(soi_raw_dir / fname, thousands=",")
    soi.columns = [c.lower() for c in soi.columns]
    soi_by_stub = soi[soi["agi_stub"] > 0].copy()
    soi_by_stub["stabbr"] = soi_by_stub["state"].str.strip().str.upper()
    # Derived: total CTC = nonrefundable CTC/ODC + refundable ACTC
    if "a07225" in soi_by_stub.columns and "a11070" in soi_by_stub.columns:
        soi_by_stub["a_ctc_total"] = (
            soi_by_stub["a07225"] + soi_by_stub["a11070"]
        )
    if "n07225" in soi_by_stub.columns and "n11070" in soi_by_stub.columns:
        soi_by_stub["n_ctc_total"] = (
            soi_by_stub["n07225"] + soi_by_stub["n11070"]
        )
    return soi_by_stub


# --- TMD national sums ---


def _compute_tmd_stub_sums(
    cached_allvars_path: Path,
    target_vars: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Compute PUF-weighted national sums by AGI stub for each variable.

    Returns dict: varname → {stub: weighted_sum}.
    """
    # capgains_net is synthetic (p22250 + p23250), not in cached_allvars
    file_cols = [v for v in target_vars if v != "capgains_net"]
    if "capgains_net" in target_vars:
        file_cols.extend(["p22250", "p23250"])
    needed_cols = list({"data_source", "s006", "c00100"} | set(file_cols))
    allvars = pd.read_csv(cached_allvars_path, usecols=needed_cols)
    puf = allvars[allvars["data_source"] == 1].copy()

    if "capgains_net" in target_vars:
        puf["capgains_net"] = puf["p22250"] + puf["p23250"]

    puf["agistub"] = (
        pd.cut(
            puf["c00100"], bins=STATE_AGI_CUTS, right=False, labels=False
        ).astype(int)
        + 1
    )

    result = {}
    for var in target_vars:
        result[var] = {}
        for stub in range(1, 11):
            mask = puf["agistub"] == stub
            result[var][stub] = float(
                (puf.loc[mask, "s006"] * puf.loc[mask, var]).sum()
            )
    return result


def _compute_tmd_aggregate_sums(
    cached_allvars_path: Path,
    aggregate_vars: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute PUF-weighted national totals (amount and nonzero count)
    for aggregate-targeted variables.

    Returns (amounts, counts) dicts.
    """
    needed_cols = ["data_source", "s006"] + aggregate_vars
    allvars = pd.read_csv(cached_allvars_path, usecols=needed_cols)
    puf = allvars[allvars["data_source"] == 1]
    s006 = puf["s006"]
    amounts = {}
    counts = {}
    for var in aggregate_vars:
        amounts[var] = float((s006 * puf[var]).sum())
        counts[var] = float((s006 * (puf[var] != 0)).sum())
    return amounts, counts


# --- Target row builders ---


def _build_soi_stub_rows(
    st: str,
    varname: str,
    soi_varname: str,
    tmd_stub_sums: Dict[int, float],
    soi_by_stub: pd.DataFrame,
    agi_labels: pd.DataFrame,
    stubs: List[int],
) -> List[dict]:
    """Build target rows for one SOI-shared variable and one state."""
    rows = []
    st_soi = soi_by_stub[soi_by_stub["stabbr"] == st]
    us_soi = soi_by_stub[soi_by_stub["stabbr"] == "US"]
    for stub in stubs:
        sr = st_soi[st_soi["agi_stub"] == stub]
        usr = us_soi[us_soi["agi_stub"] == stub]
        if sr.empty or usr.empty:
            continue
        us_val = float(usr[soi_varname].values[0])
        if us_val <= 0:
            continue
        share = float(sr[soi_varname].values[0]) / us_val
        target = tmd_stub_sums[stub] * share
        agi_row = agi_labels[agi_labels["agistub"] == stub].iloc[0]
        rows.append(
            {
                "varname": varname,
                "count": 0,
                "scope": 1,
                "agilo": agi_row["agilo"],
                "agihi": agi_row["agihi"],
                "fstatus": 0,
                "target": target,
            }
        )
    return rows


def _build_soi_aggregate_rows(
    st: str,
    varname: str,
    soi_varname: str,
    tmd_national_amount: float,
    tmd_national_count: float,
    soi_by_stub: pd.DataFrame,
) -> List[dict]:
    """Build aggregate amount + nonzero count target rows for one state."""
    rows = []
    st_soi = soi_by_stub[soi_by_stub["stabbr"] == st]
    us_soi = soi_by_stub[soi_by_stub["stabbr"] == "US"]
    st_total = st_soi[soi_varname].sum()
    us_total = us_soi[soi_varname].sum()
    if us_total <= 0:
        return rows
    share = st_total / us_total
    # Amount target
    rows.append(
        {
            "varname": varname,
            "count": 0,
            "scope": 1,
            "agilo": -9e99,
            "agihi": 9e99,
            "fstatus": 0,
            "target": tmd_national_amount * share,
        }
    )
    # Nonzero count target
    soi_nvarname = "n" + soi_varname[1:]
    if soi_nvarname in st_soi.columns:
        st_n = st_soi[soi_nvarname].sum()
        us_n = us_soi[soi_nvarname].sum()
        if us_n > 0:
            n_share = st_n / us_n
            rows.append(
                {
                    "varname": varname,
                    "count": 2,
                    "scope": 1,
                    "agilo": -9e99,
                    "agihi": 9e99,
                    "fstatus": 0,
                    "target": tmd_national_count * n_share,
                }
            )
    return rows


def _build_census_stub_rows(
    st: str,
    varname: str,
    census_share: float,
    soi_by_stub: pd.DataFrame,
    soi_varname: str,
    tmd_stub_sums: Dict[int, float],
    agi_labels: pd.DataFrame,
    stubs: List[int],
) -> List[dict]:
    """Build target rows for one Census-shared variable and one state.

    Census provides the state's total share; SOI stub proportions
    distribute that share across AGI bins.
    """
    if census_share <= 0:
        return []
    # Get SOI bin proportions for this state
    st_soi = soi_by_stub[soi_by_stub["stabbr"] == st]
    bin_vals = {}
    for stub in stubs:
        sr = st_soi[st_soi["agi_stub"] == stub]
        if not sr.empty and soi_varname in sr.columns:
            bin_vals[stub] = float(sr[soi_varname].values[0])
        else:
            bin_vals[stub] = 0
    total = sum(bin_vals.values())
    if total <= 0:
        return []
    bin_props = {s: v / total for s, v in bin_vals.items()}

    # TMD total across targeted stubs
    tmd_total = sum(tmd_stub_sums[s] for s in stubs)
    st_total = tmd_total * census_share

    rows = []
    for stub, prop in bin_props.items():
        agi_row = agi_labels[agi_labels["agistub"] == stub].iloc[0]
        rows.append(
            {
                "varname": varname,
                "count": 0,
                "scope": 1,
                "agilo": agi_row["agilo"],
                "agihi": agi_row["agihi"],
                "fstatus": 0,
                "target": st_total * prop,
            }
        )
    return rows


# --- Main entry point ---


def build_extended_targets(
    cached_allvars_path: Path,
    soi_year: int = 2022,
    stubs: Optional[List[int]] = None,
    areas: Optional[List[str]] = None,
    soi_specs: Optional[List[Tuple[str, str]]] = None,
    census_specs: Optional[List[Tuple[str, str]]] = None,
    aggregate_specs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build extended target rows for all areas.

    Returns a dict mapping area code (uppercase) → DataFrame of extra
    target rows, with the same columns as a target CSV:
        varname, count, scope, agilo, agihi, fstatus, target

    This can be passed to ``write_area_target_files()`` as extra_targets.

    Parameters
    ----------
    cached_allvars_path : Path
        Path to ``cached_allvars.csv``.
    soi_year : int
        SOI data year for geographic shares.
    stubs : list of int, optional
        AGI stubs to target (default: 5–10, i.e., $50K+).
    areas : list of str, optional
        Area codes to process (default: ALL_STATES).
    soi_specs : list of (varname, soi_varname), optional
        SOI-shared target specs (default: SOI_SHARED_SPECS).
    census_specs : list of (varname, census_type), optional
        Census-shared target specs (default: CENSUS_SHARED_SPECS).
    aggregate_specs : list of (varname, soi_varname), optional
        SOI aggregate target specs (default: SOI_AGGREGATE_SPECS).

    Returns
    -------
    dict
        Mapping of area code → DataFrame of extra target rows.
    """
    if stubs is None:
        stubs = DEFAULT_EXTENDED_STUBS
    if areas is None:
        areas = ALL_STATES
    if soi_specs is None:
        soi_specs = SOI_SHARED_SPECS
    if census_specs is None:
        census_specs = CENSUS_SHARED_SPECS
    if aggregate_specs is None:
        aggregate_specs = SOI_AGGREGATE_SPECS

    agi_labels = build_agi_labels(AreaType.STATE)

    # Compute TMD national sums by stub
    all_stub_vars = [v for v, _ in soi_specs] + [v for v, _ in census_specs]
    tmd_by_stub = _compute_tmd_stub_sums(cached_allvars_path, all_stub_vars)

    # Compute TMD aggregate sums for credit variables
    agg_vars = [v for v, _ in aggregate_specs]
    tmd_agg_amounts, tmd_agg_counts = _compute_tmd_aggregate_sums(
        cached_allvars_path, agg_vars
    )

    # Load SOI data
    soi_by_stub = _load_soi_by_stub(soi_year)

    # Load Census shares
    combined_shares, property_shares = load_census_shares()

    # SOI variable names for Census bin proportions
    census_soi_map = {
        "e18400": "a18425",  # SALT income/sales component
        "e18500": "a18500",  # Property tax component
    }

    # Build per-area extended targets
    result = {}
    for st in areas:
        new_rows = []

        # SOI-shared by stub
        for varname, soi_varname in soi_specs:
            new_rows.extend(
                _build_soi_stub_rows(
                    st,
                    varname,
                    soi_varname,
                    tmd_by_stub[varname],
                    soi_by_stub,
                    agi_labels,
                    stubs,
                )
            )

        # Census-shared by stub
        for varname, census_type in census_specs:
            if census_type == "combined":
                share = combined_shares.get(st, 0)
            else:
                share = property_shares.get(st, 0)
            soi_var = census_soi_map.get(varname, f"a{varname[1:]}")
            new_rows.extend(
                _build_census_stub_rows(
                    st,
                    varname,
                    share,
                    soi_by_stub,
                    soi_var,
                    tmd_by_stub[varname],
                    agi_labels,
                    stubs,
                )
            )

        # SOI aggregate (credits)
        for varname, soi_varname in aggregate_specs:
            new_rows.extend(
                _build_soi_aggregate_rows(
                    st,
                    varname,
                    soi_varname,
                    tmd_agg_amounts[varname],
                    tmd_agg_counts[varname],
                    soi_by_stub,
                )
            )

        if new_rows:
            result[st] = pd.DataFrame(new_rows)

    return result
