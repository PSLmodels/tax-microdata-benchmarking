import pandas as pd
import numpy as np

# Add Don's SOI target scrape

tmd = pd.read_csv(
    "../tax_microdata_benchmarking/storage/input/agi_targets.csv"
)

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


def clean_vname(vname):
    REMOVED = ["nret", "single", "mfs", "mfjss", "hoh", "_"]

    for r in REMOVED:
        vname = vname.replace(r, "")

    if vname == "" or vname == "all":
        return "count"

    VARIABLE_RENAMES = {
        "agi": "adjusted gross income",
        "all": "all",
        "exemption": "exemptions",
        "itemded": "itemized deductions",
        "count": "count",
        "sd": "standard deduction",
        "taxac": "federal tax after credits",
        "taxtot": "unknown",
        "ti": "total income",
        "amt": "alternative minimum tax",
        "busprofnetinc": "business net profits",
        "busprofnetloss": "business net losses",
        "cgdist": "capital gains distributions",
        "cggross": "capital gains gross",
        "cgloss": "capital gains losses",
        "estateinc": "estate income",
        "estateloss": "estate losses",
        "exemptint": "exempt interest",
        "iradist": "IRA distributions",
        "nexemptions": "unknown",
        "orddiv": "ordinary dividends",
        "partnerscorpinc": "partnership and S corp income",
        "partnerscorploss": "partnership and S corp losses",
        "pensions": "total pension income",
        "pensionstaxable": "taxable pension income",
        "qualdiv": "qualified dividends",
        "rentroyinc": "rent and royalty net income",
        "rentroyloss": "rent and royalty net losses",
        "socsectaxable": "taxable Social Security",
        "socsectot": "total Social Security",
        "taxbc": "federal tax before credits",
        "taxint": "taxable interest income",
        "unempcomp": "unemployment compensation",
        "wages": "employment income",
        "partnerloss": "partnership net losses",
        "partnerpinc": "partnership net income",
        "qbid": "qualified business income deduction",
        "scorpinc": "S-corporation net income",
        "scorploss": "S-corporation net losses",
        "partnerinc": "partnership net income",
        "idcontributions": "charitable contributions deductions",
        "idgst": "unknown",
        "idintpaid": "interest paid deductions",
        "idmedicalcapped": "medical expense deductions (capped)",
        "idmedicaluncapped": "medical expense deductions (uncapped)",
        "idmortgage": "mortgage interest deductions",
        "idpit": "unknown",
        "idretax": "unknown",
        "idsalt": "state and local tax deductions",
        "idtaxpaid": "unknown",
    }

    if vname in VARIABLE_RENAMES:
        return VARIABLE_RENAMES[vname]

    return vname


def clean_agi_targets_file(agi_targets):
    agi_targets["Count"] = agi_targets.vname.apply(lambda x: "nret" in x)
    agi_targets["Taxable only"] = agi_targets.datatype == "taxable"
    agi_targets.ptarget[~agi_targets.Count] *= 1e3

    agi_targets = clean_agi_bounds(agi_targets)
    agi_targets = clean_filing_status(agi_targets)

    agi_targets["SOI table"] = agi_targets.table.map(
        {
            "tab11": "Table 1.1",
            "tab12": "Table 1.2",
            "tab14": "Table 1.4",
            "tab21": "Table 2.1",
        }
    )

    agi_targets.vname = agi_targets.vname.apply(clean_vname)

    agi_targets = agi_targets.rename(
        columns={
            "agi_lower": "AGI lower bound",
            "agi_upper": "AGI upper bound",
            "filing_status": "Filing status",
            "ptarget": "Value",
            "year": "Year",
            "vname": "Variable",
        }
    )

    columns = [
        "Year",
        "SOI table",
        "Variable",
        "Filing status",
        "AGI lower bound",
        "AGI upper bound",
        "Count",
        "Taxable only",
        "Value",
    ]

    agi_targets.Variable = agi_targets.Variable.apply(
        lambda x: x.replace(" ", "_").replace("-", "_").lower()
    )

    return agi_targets[columns]


tms = clean_agi_targets_file(tmd)

tms.to_csv("soi.csv", index=False)
