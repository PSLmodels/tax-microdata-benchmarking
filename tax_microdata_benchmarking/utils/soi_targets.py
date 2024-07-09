import pandas as pd
import numpy as np
import warnings
from tax_microdata_benchmarking.storage import STORAGE_FOLDER

warnings.filterwarnings("ignore")

soi = pd.read_csv(STORAGE_FOLDER / "input" / "agi_targets.csv")


def clean_agi_bounds(
    soi: pd.DataFrame,
) -> pd.DataFrame:
    """Adds cleaned AGI bounds to Don's scraped SOI statistics file.

    Args:
        soi (pd.DataFrame): DataFrame with AGI targets.

    Returns:
        pd.DataFrame: DataFrame with cleaned AGI bounds.
    """

    soi = soi.copy()

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

    soi["agi_lower"] = soi["incrange"].map(lambda x: agi_bound_map[x][0])
    soi["agi_upper"] = soi["incrange"].map(lambda x: agi_bound_map[x][1])

    return soi


def clean_filing_status(
    soi: pd.DataFrame,
) -> pd.DataFrame:
    """Adds cleaned filing status values to Don's scraped SOI statistics file.

    Args:
        soi (pd.DataFrame): DataFrame with AGI targets.

    Returns:
        pd.DataFrame: DataFrame with cleaned filing status values.
    """
    soi = soi.copy()
    soi["is_total"] = "nret" in soi.vname
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

    soi["filing_status"] = soi.vname.apply(get_filing_status)

    return soi


def clean_vname(vname):
    REMOVED = ["nret", "single", "mfs", "mfjss", "hoh", "_"]

    for r in REMOVED:
        vname = vname.replace(r, "")

    if vname == "" or vname == "all":
        return "count"

    VARIABLE_RENAMES = {
        "agi": "adjusted gross income",
        "exemption": "exemptions",
        "itemded": "itemized deductions",
        "count": "count",
        "sd": "standard deduction",
        "taxac": "income tax after credits",
        "taxtot": "total income tax",
        "ti": "taxable income",
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
        "nexemptions": "count of exemptions",
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
        "taxbc": "income tax before credits",
        "taxint": "taxable interest income",
        "unempcomp": "unemployment compensation",
        "wages": "employment income",
        "partnerloss": "partnership and S corp losses",
        "partnerpinc": "partnership and S corp income",
        "qbid": "qualified business income deduction",
        "scorpinc": "S-corporation net income",
        "scorploss": "S-corporation net losses",
        "partnerinc": "partnership and S corp income",
        "idcontributions": "charitable contributions deductions",
        "idgst": "itemized general sales tax deduction",
        "idintpaid": "interest paid deductions",
        "idmedicalcapped": "medical expense deductions (capped)",
        "idmedicaluncapped": "medical expense deductions (uncapped)",
        "idmortgage": "mortgage interest deductions",
        "id_pitgst": "itemized state income and sales tax deductions",
        "id_retax": "itemized real estate tax deductions",
        "idsalt": "state and local tax deductions",
        "idtaxpaid": "itemized taxes paid deductions",  # federal tax payments
    }

    if vname in VARIABLE_RENAMES:
        return VARIABLE_RENAMES[vname]

    return vname


def clean_soi_file(soi):
    soi["Count"] = soi.vname.apply(lambda x: "nret" in x)
    soi["Taxable only"] = soi.datatype == "taxable"
    soi.ptarget[~soi.Count] *= 1e3

    soi = clean_agi_bounds(soi)
    soi = clean_filing_status(soi)

    soi["SOI table"] = soi.table.map(
        {
            "tab11": "Table 1.1",
            "tab12": "Table 1.2",
            "tab14": "Table 1.4",
            "tab21": "Table 2.1",
        }
    )

    soi.vname = soi.vname.apply(clean_vname)

    soi.ptarget[soi.vname == "count of exemptions"] /= 1e3

    soi = soi.rename(
        columns={
            "agi_lower": "AGI lower bound",
            "agi_upper": "AGI upper bound",
            "filing_status": "Filing status",
            "ptarget": "Value",
            "year": "Year",
            "vname": "Variable",
            "xlcolumn": "XLSX column",
            "xlrownum": "XLSX row",
        }
    )

    columns = [
        "Year",
        "SOI table",
        "XLSX column",
        "XLSX row",
        "Variable",
        "Filing status",
        "AGI lower bound",
        "AGI upper bound",
        "Count",
        "Taxable only",
        "Full population",
        "Value",
    ]

    soi.Variable = soi.Variable.apply(
        lambda x: x.replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )

    # Add aggregate (all filers, all filing statuses, all AGI bins) label

    soi["Full population"] = (
        (soi["Filing status"] == "All")
        & (soi["AGI lower bound"] == -np.inf)
        & (soi["AGI upper bound"] == np.inf)
        & ~soi["Taxable only"]
    )

    # De-duplicate along
    unique_columns = [
        "Year",
        "Variable",
        "Filing status",
        "AGI lower bound",
        "AGI upper bound",
        "Count",
        "Taxable only",
        "Value",
    ]

    soi = soi.groupby(unique_columns).first().reset_index()

    return soi[columns]


soi = clean_soi_file(soi)

soi.to_csv(STORAGE_FOLDER / "input" / "soi.csv", index=False)
