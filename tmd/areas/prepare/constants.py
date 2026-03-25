"""
Shared constants for area target preparation.

AGI cut points, SOI file patterns, variable mappings, and area
type definitions used across the preparation pipeline for
states and congressional districts.
"""

from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class AreaType(Enum):
    """Type of sub-national area."""

    STATE = "state"
    CD = "cd"


# --- AGI range definitions ---

# State AGI stubs: 10 bins (IRS SOI Historical Table 2)
# agistub 0 = total, agistubs 1-10 = bins
STATE_AGI_CUTS: List[float] = [
    -np.inf,
    1,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    np.inf,
]

# Number of non-total AGI stubs
STATE_NUM_AGI_STUBS = len(STATE_AGI_CUTS) - 1  # 10

# CD AGI stubs: 9 bins (IRS SOI Congressional District data)
# Same as state stubs 1-8, with stubs 9+10 merged into "$500K+"
# agistub 0 = total, agistubs 1-9 = bins
CD_AGI_CUTS: List[float] = [
    -np.inf,
    1,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    200_000,
    500_000,
    np.inf,
]

CD_NUM_AGI_STUBS = len(CD_AGI_CUTS) - 1  # 9


def build_agi_labels(area_type: AreaType) -> pd.DataFrame:
    """
    Build DataFrame with AGI stub definitions.

    Returns DataFrame with columns: agistub, agilo, agihi, agilabel
    where agistub 0 is the total row and 1..N are the bins.
    """
    if area_type == AreaType.STATE:
        cuts = STATE_AGI_CUTS
    elif area_type == AreaType.CD:
        cuts = CD_AGI_CUTS
    else:
        raise ValueError(f"Unsupported area_type: {area_type}")

    rows = [
        {
            "agistub": 0,
            "agilo": -9e99,
            "agihi": 9e99,
            "agilabel": "Total",
        }
    ]
    for i in range(len(cuts) - 1):
        lo = cuts[i]
        hi = cuts[i + 1]
        agilo = -9e99 if np.isneginf(lo) else lo
        agihi = 9e99 if np.isposinf(hi) else hi
        if np.isneginf(lo):
            label = f"Under ${hi:,.0f}"
        elif np.isposinf(hi):
            label = f"${lo:,.0f} or more"
        else:
            label = f"${lo:,.0f} under ${hi:,.0f}"
        rows.append(
            {
                "agistub": i + 1,
                "agilo": agilo,
                "agihi": agihi,
                "agilabel": label,
            }
        )
    return pd.DataFrame(rows)


# --- SOI file naming patterns ---

# State SOI CSV files by year (2-digit year substituted into pattern)
SOI_STATE_CSV_PATTERNS: Dict[int, str] = {
    2015: "15in54cmcsv.csv",
    2016: "16in54cmcsv.csv",
    2017: "17in54cmcsv.csv",
    2018: "18in55cmagi.csv",
    2019: "19in55cmcsv.csv",
    2020: "20in55cmcsv.csv",
    2021: "21in55cmcsv.csv",
    2022: "22in55cmcsv.csv",
}

# CD SOI CSV files by year
SOI_CD_CSV_PATTERNS: Dict[int, str] = {
    2021: "21incd.csv",
    2022: "22incd.csv",
}

# At-large states: single CD coded as CONG_DISTRICT=0 in SOI data
AT_LARGE_STATES: List[str] = [
    "AK",
    "DC",
    "DE",
    "MT",
    "ND",
    "SD",
    "VT",
    "WY",
]


# --- Variable classifications ---

# Count variables that represent "count of all returns"
# (as opposed to "count of nonzero" for other count variables)
ALLCOUNT_VARS: List[str] = ["n1", "n2", "mars1", "mars2", "mars4"]


# --- TMD-SOI sharing mappings ---
# Every targeted variable uses SOI for geographic distribution
# and TMD for national levels.
# Format: (tmd_varname, soi_base_varname, count_type, fstatus, desc)
#   count_type: 0=amount, 1=allcount, 2=nonzero count
#   fstatus: 0=all, 1=single, 2=MFJ, 4=HoH
ALL_SHARING_MAPPINGS: List[Tuple[str, str, int, int, str]] = [
    # Amounts (count=0): TMD sum shared by SOI amounts
    ("c00100", "00100", 0, 0, "AGI"),
    ("e00200", "00200", 0, 0, "Wages"),
    ("e00300", "00300", 0, 0, "Taxable interest"),
    ("e01500", "01700", 0, 0, "Pensions total by taxable"),
    ("e02400", "02500", 0, 0, "SS total by taxable"),
    ("c18300", "18300", 0, 0, "Actual SALT deduction"),
    ("e26270", "26270", 0, 0, "Partnership/S corp"),
    # Nonzero counts (count=2): TMD nonzero count shared by SOI
    ("e00200", "00200", 2, 0, "Wages nonzero count"),
    ("c18300", "18300", 2, 0, "Actual SALT nz count"),
    # All-return counts (count=1): TMD total count shared by SOI
    ("c00100", "n1", 1, 0, "Total returns"),
    ("c00100", "mars1", 1, 1, "Single returns"),
    ("c00100", "mars2", 1, 2, "MFJ returns"),
    ("c00100", "mars4", 1, 4, "HoH returns"),
]


# --- State and Congressional District information ---

# Number of Congressional districts per state by Census year.
# Source: 2020 Census Apportionment Results, April 26, 2021.
STATE_INFO: Dict[str, Dict[int, int]] = {
    "AL": {2020: 7, 2010: 7},
    "AK": {2020: 1, 2010: 1},
    "AZ": {2020: 9, 2010: 9},
    "AR": {2020: 4, 2010: 4},
    "CA": {2020: 52, 2010: 53},
    "CO": {2020: 8, 2010: 7},
    "CT": {2020: 5, 2010: 5},
    "DE": {2020: 1, 2010: 1},
    "DC": {2020: 0, 2010: 0},
    "FL": {2020: 28, 2010: 27},
    "GA": {2020: 14, 2010: 14},
    "HI": {2020: 2, 2010: 2},
    "ID": {2020: 2, 2010: 2},
    "IL": {2020: 17, 2010: 18},
    "IN": {2020: 9, 2010: 9},
    "IA": {2020: 4, 2010: 4},
    "KS": {2020: 4, 2010: 4},
    "KY": {2020: 6, 2010: 6},
    "LA": {2020: 6, 2010: 6},
    "ME": {2020: 2, 2010: 2},
    "MD": {2020: 8, 2010: 8},
    "MA": {2020: 9, 2010: 9},
    "MI": {2020: 13, 2010: 14},
    "MN": {2020: 8, 2010: 8},
    "MS": {2020: 4, 2010: 4},
    "MO": {2020: 8, 2010: 8},
    "MT": {2020: 2, 2010: 1},
    "NE": {2020: 3, 2010: 3},
    "NV": {2020: 4, 2010: 4},
    "NH": {2020: 2, 2010: 2},
    "NJ": {2020: 12, 2010: 12},
    "NM": {2020: 3, 2010: 3},
    "NY": {2020: 26, 2010: 27},
    "NC": {2020: 14, 2010: 13},
    "ND": {2020: 1, 2010: 1},
    "OH": {2020: 15, 2010: 16},
    "OK": {2020: 5, 2010: 5},
    "OR": {2020: 6, 2010: 5},
    "PA": {2020: 17, 2010: 18},
    "RI": {2020: 2, 2010: 2},
    "SC": {2020: 7, 2010: 7},
    "SD": {2020: 1, 2010: 1},
    "TN": {2020: 9, 2010: 9},
    "TX": {2020: 38, 2010: 36},
    "UT": {2020: 4, 2010: 4},
    "VT": {2020: 1, 2010: 1},
    "VA": {2020: 11, 2010: 11},
    "WA": {2020: 10, 2010: 10},
    "WV": {2020: 2, 2010: 3},
    "WI": {2020: 8, 2010: 8},
    "WY": {2020: 1, 2010: 1},
}

# All valid 2-letter state codes (uppercase)
ALL_STATES: List[str] = sorted(STATE_INFO.keys())


def get_agi_cuts(area_type: AreaType) -> List[float]:
    """Return AGI cut points for the given area type."""
    if area_type == AreaType.STATE:
        return STATE_AGI_CUTS
    if area_type == AreaType.CD:
        return CD_AGI_CUTS
    raise ValueError(f"Unsupported area_type: {area_type}")


def get_num_agi_stubs(area_type: AreaType) -> int:
    """Return the number of non-total AGI stubs for the given area type."""
    if area_type == AreaType.STATE:
        return STATE_NUM_AGI_STUBS
    if area_type == AreaType.CD:
        return CD_NUM_AGI_STUBS
    raise ValueError(f"Unsupported area_type: {area_type}")


# Faux area prefixes used for testing
FAUX_AREA_PREFIXES: List[str] = [
    "xx",
    "xy",
    "xz",
]
