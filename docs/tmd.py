import pandas as pd
import numpy as np


def clean_agi_bounds(
    agi_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Adds cleaned AGI bounds to Don's scraped SOI statistics file.

    Args:
        agi_targets (pd.DataFrame): DataFrame with AGI targets.

    Returns:
        pd.DataFrame: DataFrame with cleaned AGI bounds.
    """

    agi_targets = agi_targets.copy()

    agi_bound_map = {
        "All returns": (-np.inf, np.inf),
        "No adjusted gross income": (-np.inf, 0),
        "$1 under $5,000": (1, 5_000),
        "$5,000 under $10,000": (5_000, 10_000),
        "$10,000 under $15,000": (10_000, 15_000),
        "$15,000 under $20,000": (15_000, 20_000),
        "$20,000 under $25,000": (20_000, 25_000),
        "$25,000 under $30,000": (25_000, 30_000),
        "$30,000 under $40,000": (30_000, 40_000),
        "$40,000 under $50,000": (40_000, 50_000),
        "$50,000 under $75,000": (50_000, 75_000),
        "$75,000 under $100,000": (75_000, 100_000),
        "$100,000 under $200,000": (100_000, 200_000),
        "$200,000 under $500,000": (200_000, 500_000),
        "$500,000 under $1,000,000": (500_000, 1_000_000),
        "$1,000,000 under $1,500,000": (1_000_000, 1_500_000),
        "$1,500,000 under $2,000,000": (1_500_000, 2_000_000),
        "$2,000,000 under $5,000,000": (2_000_000, 5_000_000),
        "$5,000,000 under $10,000,000": (5_000_000, 10_000_000),
        "$10,000,000 or more": (10_000_000, np.inf),
        "All returns, total": (-np.inf, np.inf),
        "Taxable returns, total": (-np.inf, np.inf),
        "No adjusted gross income (includes deficits)": (-np.inf, 0),
        "$1,000,000 or more": (1_000_000, np.inf),
        "Under $5,000": (0, 5_000),
        "$30,000 under $35,000": (30_000, 35_000),
        "$35,000 under $40,000": (35_000, 40_000),
        "$40,000 under $45,000": (40_000, 45_000),
        "$45,000 under $50,000": (45_000, 50_000),
        "$50,000 under $55,000": (50_000, 55_000),
        "$55,000 under $60,000": (55_000, 60_000),
        "$60,000 under $75,000": (60_000, 75_000),
    }

    agi_targets["agi_lower"] = agi_targets["incrange"].map(
        lambda x: agi_bound_map[x][0]
    )
    agi_targets["agi_upper"] = agi_targets["incrange"].map(
        lambda x: agi_bound_map[x][1]
    )

    return agi_targets


def clean_filing_status(
    agi_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Adds cleaned filing status values to Don's scraped SOI statistics file.

    Args:
        agi_targets (pd.DataFrame): DataFrame with AGI targets.

    Returns:
        pd.DataFrame: DataFrame with cleaned filing status values.
    """
    agi_targets = agi_targets.copy()
    agi_targets["is_total"] = "nret" in agi_targets.vname
    # if none of single, mfs, mfjss, hoh in vname, then mars_subset = False

    def get_filing_status(name):
        if "single" in name:
            return "Single"
        elif "mfs" in name:
            return "Married Filing Separately"
        elif "mfjss" in name:
            return "Married Filing Jointly/Surviving Spouse"
        elif "hoh" in name:
            return "Head of Household"
        else:
            return "All"

    agi_targets["filing_status"] = agi_targets.vname.apply(get_filing_status)

    return agi_targets


def clean_agi_targets_file(agi_targets):
    agi_targets["is_total"] = "nret" in agi_targets.vname

    agi_targets = clean_agi_bounds(agi_targets)
    agi_targets = clean_filing_status(agi_targets)

    agi
