"""IRS SOI Excel file layout configuration.

For each table and year, specifies:
  - filename
  - data row range (1-indexed, inclusive)
  - column specs: which Excel column letter(s) hold each variable

This is the single source of truth for reading IRS Excel files.
Update here when IRS adds a new year or restructures a spreadsheet.

Column letters come from direct inspection of the spreadsheets and
cross-validation against SOI published totals; see
tmd/national_targets/docs/variable_validation_methodology.md.

Units: IRS amount columns are in thousands of dollars.
       extract_irs_to_csv.py multiplies amounts by 1000 → dollars.
       Counts and numbers are already in full units (no conversion).
"""

YEARS = (2015, 2021, 2022)

# ── Filenames ────────────────────────────────────────────────────────────────

FILE_NAMES = {
    ("tab11", 2015): "15in11si.xls",
    ("tab11", 2021): "21in11si.xls",
    ("tab11", 2022): "22in11si.xls",
    ("tab12", 2015): "15in12ms.xls",
    ("tab12", 2021): "21in12ms.xls",
    ("tab12", 2022): "22in12ms.xls",
    ("tab14", 2015): "15in14ar.xls",
    ("tab14", 2021): "21in14ar.xls",
    ("tab14", 2022): "22in14ar.xls",
    ("tab21", 2015): "15in21id.xls",
    ("tab21", 2021): "21in21id.xls",
    ("tab21", 2022): "22in21id.xls",
}

TABLE_DESCRIPTIONS = {
    "tab11": "Table 1.1: All Returns — Returns and AGI by Income Size",
    "tab12": "Table 1.2: All Returns — Marital Status",
    "tab14": "Table 1.4: All Returns — Sources of Income and Adjustments",
    "tab21": "Table 2.1: Returns with Itemized Deductions",
}

# ── Data row ranges (1-indexed Excel row numbers, inclusive) ─────────────────
# First data row = "All returns, total" row
# Last data row  = highest income bracket row

DATA_ROWS = {
    ("tab11", 2015): (10, 29),
    ("tab11", 2021): (10, 29),
    ("tab11", 2022): (10, 29),
    ("tab12", 2015): (9, 28),
    ("tab12", 2021): (9, 28),
    ("tab12", 2022): (9, 28),
    ("tab14", 2015): (9, 28),
    ("tab14", 2021): (9, 28),
    ("tab14", 2022): (9, 28),
    ("tab21", 2015): (10, 32),
    ("tab21", 2021): (10, 32),
    ("tab21", 2022): (10, 32),
}

# ── Column specifications ────────────────────────────────────────────────────
# Each entry defines one (variable × type × filter × marstat) combination.
# "cols" maps year → Excel column letter for that year's file.
# Omit a year from "cols" if the variable doesn't exist in that year.
#
# var_type:
#   "amount"  — dollar amount (thousands in IRS file, converted to dollars)
#   "count"   — number of returns
#   "number"  — item count (e.g. number of exemptions), not a return count
#
# value_filter:
#   "all"  — all returns regardless of sign or zero value
#   "nz"   — returns with nonzero value
#   "gt0"  — returns with strictly positive value (income side)
#   "lt0"  — returns with strictly negative value (loss side)
#
# marstat:
#   "all" | "single" | "mfjss" | "mfs" | "hoh"
#
# Year-availability notes:
#   2015-only:  exemption, exemptions_n (tab14/12), partnerscorpincome (tab14)
#   2021+ only: qbid, partnerincome, scorpincome (tab14), id_pitgst (tab21)


# ── Table 1.1: All Returns — AGI by income size ──────────────────────────────
# Only two useful columns; "Percent of total" (col C) is intentionally skipped.

TAB11_COLUMNS = [
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "all",
        "description": "Number of returns",
        "cols": {2015: "B", 2021: "B", 2022: "B"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "all",
        "description": "Adjusted gross income",
        "cols": {2015: "D", 2021: "D", 2022: "D"},
    },
]


# ── Table 1.2: All Returns — Marital Status ──────────────────────────────────
# 2015 has a personal exemption column (col D) not present in 2021/2022.
# This shifts all 2015 column letters by +1 for the "all" block and
# by an additional +1 per marstat block (each block also had an exemption col).

TAB12_COLUMNS = [
    # ── All returns ──────────────────────────────────────────────────────────
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "all",
        "description": "Number of returns",
        "cols": {2015: "B", 2021: "B", 2022: "B"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "all",
        "description": "Adjusted gross income",
        "cols": {2015: "C", 2021: "C", 2022: "C"},
    },
    # Personal exemption: 2015 only (TCJA eliminated for 2018+)
    {
        "var_name": "exemption",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "all",
        "description": "Personal exemption (2015 only)",
        "cols": {2015: "D"},
    },
    {
        "var_name": "itemded",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Itemized deductions — number of returns",
        "cols": {2015: "E", 2021: "D", 2022: "D"},
    },
    {
        "var_name": "itemded",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Total itemized deductions",
        "cols": {2015: "F", 2021: "E", 2022: "E"},
    },
    {
        "var_name": "sd",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Standard deduction — number of returns",
        "cols": {2015: "G", 2021: "F", 2022: "F"},
    },
    {
        "var_name": "sd",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Standard deduction",
        "cols": {2015: "H", 2021: "G", 2022: "G"},
    },
    {
        "var_name": "ti",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxable income — number of returns",
        "cols": {2015: "I", 2021: "H", 2022: "H"},
    },
    {
        "var_name": "ti",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxable income",
        "cols": {2015: "J", 2021: "I", 2022: "I"},
    },
    {
        "var_name": "taxac",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Income tax after credits — number of returns",
        "cols": {2015: "K", 2021: "J", 2022: "J"},
    },
    {
        "var_name": "taxac",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Income tax after credits",
        "cols": {2015: "L", 2021: "K", 2022: "K"},
    },
    {
        "var_name": "tottax",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Total tax — number of returns",
        "cols": {2015: "M", 2021: "L", 2022: "L"},
    },
    {
        "var_name": "tottax",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Total tax",
        "cols": {2015: "N", 2021: "M", 2022: "M"},
    },
    # ── Married filing jointly / surviving spouse ────────────────────────────
    # Each marstat block in 2015 also has an exemption col, so the 2015 shift
    # grows by 1 per block: mfjss +1, mfs +2, hoh +3, single +4.
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "mfjss",
        "description": "Number of returns — married filing jointly/SS",
        "cols": {2015: "O", 2021: "N", 2022: "N"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "mfjss",
        "description": "Adjusted gross income — married filing jointly/SS",
        "cols": {2015: "P", 2021: "O", 2022: "O"},
    },
    # ── Married filing separately ────────────────────────────────────────────
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "mfs",
        "description": "Number of returns — married filing separately",
        "cols": {2015: "AB", 2021: "Z", 2022: "Z"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "mfs",
        "description": "Adjusted gross income — married filing separately",
        "cols": {2015: "AC", 2021: "AA", 2022: "AA"},
    },
    # ── Head of household ────────────────────────────────────────────────────
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "hoh",
        "description": "Number of returns — head of household",
        "cols": {2015: "AO", 2021: "AL", 2022: "AL"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "hoh",
        "description": "Adjusted gross income — head of household",
        "cols": {2015: "AP", 2021: "AM", 2022: "AM"},
    },
    # ── Single ───────────────────────────────────────────────────────────────
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "single",
        "description": "Number of returns — single",
        "cols": {2015: "BB", 2021: "AX", 2022: "AX"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "single",
        "description": "Adjusted gross income — single",
        "cols": {2015: "BC", 2021: "AY", 2022: "AY"},
    },
]


# ── Table 1.4: All Returns — Sources of Income and Adjustments ───────────────
# Column positions are stable for 2015 and 2021 for most variables, but
# IRS added several new columns in 2022 before the existing variables,
# shifting many of them. See column comments where 2022 differs.
# Additionally:
#   - 2015 has partnerscorpincome (combined) but not partnerincome/scorpincome
#   - 2021+ have partnerincome and scorpincome (split) but not
#     partnerscorpincome
#   - 2015 has exemption/exemptions_n (pre-TCJA)
#   - 2021+ have qbid (post-TCJA)

TAB14_COLUMNS = [
    # ── Number of returns and AGI ────────────────────────────────────────────
    {
        "var_name": "agi",
        "var_type": "count",
        "value_filter": "all",
        "marstat": "all",
        "description": "Number of returns",
        "cols": {2015: "B", 2021: "B", 2022: "B"},
    },
    {
        "var_name": "agi",
        "var_type": "amount",
        "value_filter": "all",
        "marstat": "all",
        "description": "Adjusted gross income",
        "cols": {2015: "C", 2021: "C", 2022: "C"},
    },
    # ── Wages and salaries ───────────────────────────────────────────────────
    {
        "var_name": "wages",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Salaries and wages — number of returns",
        "cols": {2015: "F", 2021: "F", 2022: "F"},
    },
    {
        "var_name": "wages",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Salaries and wages",
        "cols": {2015: "G", 2021: "G", 2022: "G"},
    },
    # ── Taxable interest ─────────────────────────────────────────────────────
    # 2022: IRS inserted new columns (tips, overtime) before taxint
    {
        "var_name": "taxint",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxable interest — number of returns",
        "cols": {2015: "H", 2021: "H", 2022: "T"},
    },
    {
        "var_name": "taxint",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxable interest",
        "cols": {2015: "I", 2021: "I", 2022: "U"},
    },
    # ── Tax-exempt interest ──────────────────────────────────────────────────
    {
        "var_name": "exemptint",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Tax-exempt interest — number of returns",
        "cols": {2015: "J", 2021: "J", 2022: "V"},
    },
    {
        "var_name": "exemptint",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Tax-exempt interest",
        "cols": {2015: "K", 2021: "K", 2022: "W"},
    },
    # ── Ordinary dividends ───────────────────────────────────────────────────
    {
        "var_name": "orddiv",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Ordinary dividends — number of returns",
        "cols": {2015: "L", 2021: "L", 2022: "X"},
    },
    {
        "var_name": "orddiv",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Ordinary dividends",
        "cols": {2015: "M", 2021: "M", 2022: "Y"},
    },
    # ── Qualified dividends ──────────────────────────────────────────────────
    {
        "var_name": "qualdiv",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Qualified dividends — number of returns",
        "cols": {2015: "N", 2021: "N", 2022: "Z"},
    },
    {
        "var_name": "qualdiv",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Qualified dividends",
        "cols": {2015: "O", 2021: "O", 2022: "AA"},
    },
    # ── Business / profession net income (income and loss reported
    # ── separately)
    {
        "var_name": "busprofincome",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Business income — number of returns with profit",
        "cols": {2015: "T", 2021: "T", 2022: "AF"},
    },
    {
        "var_name": "busprofincome",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Business net income (profit)",
        "cols": {2015: "U", 2021: "U", 2022: "AG"},
    },
    {
        "var_name": "busprofincome",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Business income — number of returns with loss",
        "cols": {2015: "V", 2021: "V", 2022: "AH"},
    },
    {
        "var_name": "busprofincome",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Business net loss",
        "cols": {2015: "W", 2021: "W", 2022: "AI"},
    },
    # ── Capital gain distributions (Schedule D) ──────────────────────────────
    {
        "var_name": "cgdist",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Capital gain distributions — number of returns",
        "cols": {2015: "X", 2021: "X", 2022: "AJ"},
    },
    {
        "var_name": "cgdist",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Capital gain distributions",
        "cols": {2015: "Y", 2021: "Y", 2022: "AK"},
    },
    # ── Net capital gain / loss ──────────────────────────────────────────────
    {
        "var_name": "cgtaxable",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Capital gains — number of returns with net gain",
        "cols": {2015: "Z", 2021: "Z", 2022: "AL"},
    },
    {
        "var_name": "cgtaxable",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Net capital gain",
        "cols": {2015: "AA", 2021: "AA", 2022: "AM"},
    },
    {
        "var_name": "cgtaxable",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Capital gains — number of returns with net loss",
        "cols": {2015: "AB", 2021: "AB", 2022: "AN"},
    },
    {
        "var_name": "cgtaxable",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Net capital loss",
        "cols": {2015: "AC", 2021: "AC", 2022: "AO"},
    },
    # ── IRA distributions ────────────────────────────────────────────────────
    {
        "var_name": "iradist",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "IRA distributions — number of returns",
        "cols": {2015: "AH", 2021: "AH", 2022: "AT"},
    },
    {
        "var_name": "iradist",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "IRA distributions",
        "cols": {2015: "AI", 2021: "AI", 2022: "AU"},
    },
    # ── Pensions and annuities ───────────────────────────────────────────────
    {
        "var_name": "pensions",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Pensions and annuities — number of returns",
        "cols": {2015: "AJ", 2021: "AJ", 2022: "AV"},
    },
    {
        "var_name": "pensions",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Pensions and annuities (total)",
        "cols": {2015: "AK", 2021: "AK", 2022: "AW"},
    },
    {
        "var_name": "pensions_taxable",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Pensions and annuities (taxable) — number of returns",
        "cols": {2015: "AL", 2021: "AL", 2022: "AX"},
    },
    {
        "var_name": "pensions_taxable",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Pensions and annuities (taxable)",
        "cols": {2015: "AM", 2021: "AM", 2022: "AY"},
    },
    # ── Rent and royalty net income / loss ───────────────────────────────────
    {
        "var_name": "rentroyalty",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Rent/royalty — number of returns with income",
        "cols": {2015: "AZ", 2021: "AZ", 2022: "BL"},
    },
    {
        "var_name": "rentroyalty",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Rent and royalty net income",
        "cols": {2015: "BA", 2021: "BA", 2022: "BM"},
    },
    {
        "var_name": "rentroyalty",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Rent/royalty — number of returns with loss",
        "cols": {2015: "BB", 2021: "BB", 2022: "BN"},
    },
    {
        "var_name": "rentroyalty",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Rent and royalty net loss",
        "cols": {2015: "BC", 2021: "BC", 2022: "BO"},
    },
    # ── Partnership and S-corp: 2015 combined, 2021+ split ───────────────────
    # IRS reported partnership+S-corp combined in 2015; split into two series
    # starting 2021.  Use partnerscorpincome for 2015 targets; use
    # partnerincome + scorpincome for 2021+ (potential_targets_to_soi.py
    # aggregates them back into partnership_and_s_corp for the optimizer).
    {
        "var_name": "partnerscorpincome",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Partnership+S-corp income (2015 only, combined)",
        "cols": {2015: "BD"},
    },
    {
        "var_name": "partnerscorpincome",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Partnership+S-corp net income (2015 only, combined)",
        "cols": {2015: "BE"},
    },
    {
        "var_name": "partnerscorpincome",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Partnership+S-corp loss (2015 only, combined)",
        "cols": {2015: "BF"},
    },
    {
        "var_name": "partnerscorpincome",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Partnership and S-corp net loss (2015 only, combined)",
        "cols": {2015: "BG"},
    },
    {
        "var_name": "partnerincome",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Partnership income — number with income (2021+)",
        "cols": {2021: "BD", 2022: "BP"},
    },
    {
        "var_name": "partnerincome",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Partnership net income (2021+)",
        "cols": {2021: "BE", 2022: "BQ"},
    },
    {
        "var_name": "partnerincome",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Partnership income — number with loss (2021+)",
        "cols": {2021: "BF", 2022: "BR"},
    },
    {
        "var_name": "partnerincome",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Partnership net loss (2021+)",
        "cols": {2021: "BG", 2022: "BS"},
    },
    {
        "var_name": "scorpincome",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "S-corporation income — number with income (2021+)",
        "cols": {2021: "BH", 2022: "BT"},
    },
    {
        "var_name": "scorpincome",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "S-corporation net income (2021+)",
        "cols": {2021: "BI", 2022: "BU"},
    },
    {
        "var_name": "scorpincome",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "S-corporation income — number with loss (2021+)",
        "cols": {2021: "BJ", 2022: "BV"},
    },
    {
        "var_name": "scorpincome",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "S-corporation net loss (2021+)",
        "cols": {2021: "BK", 2022: "BW"},
    },
    # ── Estate and trust income / loss ───────────────────────────────────────
    # Positions differ all 3 years because 2021 inserted scorpincome cols
    # and 2022 shifted further.
    {
        "var_name": "estateincome",
        "var_type": "count",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Estate/trust income — number with income",
        "cols": {2015: "BH", 2021: "BL", 2022: "BX"},
    },
    {
        "var_name": "estateincome",
        "var_type": "amount",
        "value_filter": "gt0",
        "marstat": "all",
        "description": "Estate and trust net income",
        "cols": {2015: "BI", 2021: "BM", 2022: "BY"},
    },
    {
        "var_name": "estateincome",
        "var_type": "count",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Estate/trust income — number with loss",
        "cols": {2015: "BJ", 2021: "BN", 2022: "BZ"},
    },
    {
        "var_name": "estateincome",
        "var_type": "amount",
        "value_filter": "lt0",
        "marstat": "all",
        "description": "Estate and trust net loss",
        "cols": {2015: "BK", 2021: "BO", 2022: "CA"},
    },
    # ── Unemployment compensation ────────────────────────────────────────────
    {
        "var_name": "unempcomp",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Unemployment compensation — number of returns",
        "cols": {2015: "BP", 2021: "BT", 2022: "CF"},
    },
    {
        "var_name": "unempcomp",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Unemployment compensation",
        "cols": {2015: "BQ", 2021: "BU", 2022: "CG"},
    },
    # ── Social security benefits ─────────────────────────────────────────────
    {
        "var_name": "socsectot",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Social security benefits (total) — number of returns",
        "cols": {2015: "BR", 2021: "BV", 2022: "CH"},
    },
    {
        "var_name": "socsectot",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Social security benefits (total)",
        "cols": {2015: "BS", 2021: "BW", 2022: "CI"},
    },
    {
        "var_name": "socsectaxable",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Social security benefits (taxable) — num. returns",
        "cols": {2015: "BT", 2021: "BX", 2022: "CJ"},
    },
    {
        "var_name": "socsectaxable",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Social security benefits (taxable)",
        "cols": {2015: "BU", 2021: "BY", 2022: "CK"},
    },
    # ── Personal exemptions: 2015 only (TCJA eliminated after 2017) ──────────
    {
        "var_name": "exemption",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Personal exemptions (2015 only)",
        "cols": {2015: "DY"},
    },
    {
        "var_name": "exemptions_n",
        "var_type": "number",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Number of exemptions (2015 only)",
        "cols": {2015: "DX"},
    },
    # ── Itemized deductions (total) ──────────────────────────────────────────
    {
        "var_name": "itemded",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Itemized deductions — number of returns",
        "cols": {2015: "DV", 2021: "DV", 2022: "EF"},
    },
    {
        "var_name": "itemded",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Total itemized deductions",
        "cols": {2015: "DW", 2021: "DW", 2022: "EG"},
    },
    # ── Qualified business income deduction: 2021+ (post-TCJA) ───────────────
    {
        "var_name": "qbid",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "QBI deduction — number of returns (2021+)",
        "cols": {2021: "DX", 2022: "EH"},
    },
    {
        "var_name": "qbid",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Qualified business income deduction (2021+)",
        "cols": {2021: "DY", 2022: "EI"},
    },
    # ── Alternative minimum tax ──────────────────────────────────────────────
    {
        "var_name": "amt",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Alternative minimum tax — number of returns",
        "cols": {2015: "ED", 2021: "ED", 2022: "EN"},
    },
    {
        "var_name": "amt",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Alternative minimum tax",
        "cols": {2015: "EE", 2021: "EE", 2022: "EO"},
    },
    # ── Income tax before credits ────────────────────────────────────────────
    {
        "var_name": "taxbc",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Income tax before credits — number of returns",
        "cols": {2015: "EH", 2021: "EH", 2022: "ER"},
    },
    {
        "var_name": "taxbc",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Income tax before credits",
        "cols": {2015: "EI", 2021: "EI", 2022: "ES"},
    },
]


# ── Table 2.1: Returns with Itemized Deductions ──────────────────────────────
# Covers taxable returns with itemized deductions only.
# Column positions differ across all three years because IRS regularly
# adds or reorganizes itemized deduction sub-components.
# id_pitgst (combined state income/sales tax) was added in 2021.

TAB21_COLUMNS = [
    # ── Count of returns with itemized deductions ────────────────────────────
    {
        "var_name": "id",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Number of returns with itemized deductions",
        "cols": {2015: "B", 2021: "B", 2022: "B"},
    },
    # ── Medical and dental expenses ──────────────────────────────────────────
    {
        "var_name": "id_medical_capped",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Medical expenses (capped) — number of returns",
        "cols": {2015: "BK", 2021: "BI", 2022: "BU"},
    },
    {
        "var_name": "id_medical_capped",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Medical expenses (deductible/capped)",
        "cols": {2015: "BL", 2021: "BJ", 2022: "BV"},
    },
    {
        "var_name": "id_medical_uncapped",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Medical expenses (total/uncapped) — number of returns",
        "cols": {2015: "BM", 2021: "BK", 2022: "BW"},
    },
    {
        "var_name": "id_medical_uncapped",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Medical expenses (total/uncapped)",
        "cols": {2015: "BN", 2021: "BL", 2022: "BX"},
    },
    # ── Taxes paid ───────────────────────────────────────────────────────────
    {
        "var_name": "id_taxpaid",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxes paid deduction — number of returns",
        "cols": {2015: "BQ", 2021: "BO", 2022: "CA"},
    },
    {
        "var_name": "id_taxpaid",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Taxes paid deduction",
        "cols": {2015: "BR", 2021: "BP", 2022: "CB"},
    },
    # ── State and local taxes (SALT) ─────────────────────────────────────────
    {
        "var_name": "id_salt",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State and local taxes (SALT) — number of returns",
        "cols": {2015: "BS", 2021: "BQ", 2022: "CC"},
    },
    {
        "var_name": "id_salt",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State and local taxes (SALT)",
        "cols": {2015: "BT", 2021: "BR", 2022: "CD"},
    },
    # ── State income/sales taxes combined: 2021+ only ────────────────────────
    # In 2021 IRS began reporting the combined income-or-sales-tax election
    # as a separate line (id_pitgst), in addition to the individual components.
    {
        "var_name": "id_pitgst",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State income/sales taxes — number of returns (2021+)",
        "cols": {2021: "BS", 2022: "CE"},
    },
    {
        "var_name": "id_pitgst",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State income or sales taxes combined (2021+)",
        "cols": {2021: "BT", 2022: "CF"},
    },
    # ── State/local income taxes ─────────────────────────────────────────────
    {
        "var_name": "id_pit",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State income taxes — number of returns",
        "cols": {2015: "BU", 2021: "BU", 2022: "CG"},
    },
    {
        "var_name": "id_pit",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "State income taxes",
        "cols": {2015: "BV", 2021: "BV", 2022: "CH"},
    },
    # ── General sales taxes ──────────────────────────────────────────────────
    {
        "var_name": "id_gst",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "General sales taxes — number of returns",
        "cols": {2015: "BW", 2021: "BW", 2022: "CI"},
    },
    {
        "var_name": "id_gst",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "General sales taxes",
        "cols": {2015: "BX", 2021: "BX", 2022: "CJ"},
    },
    # ── Real estate taxes ────────────────────────────────────────────────────
    {
        "var_name": "id_retax",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Real estate taxes — number of returns",
        "cols": {2015: "BY", 2021: "BY", 2022: "CK"},
    },
    {
        "var_name": "id_retax",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Real estate taxes",
        "cols": {2015: "BZ", 2021: "BZ", 2022: "CL"},
    },
    # ── Interest paid ────────────────────────────────────────────────────────
    # 2021/2022 column positions are offset from 2015 because id_pitgst
    # (2 cols) was inserted before id_pit/id_gst in 2021.
    {
        "var_name": "id_intpaid",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Interest paid — number of returns",
        "cols": {2015: "CE", 2021: "CG", 2022: "CS"},
    },
    {
        "var_name": "id_intpaid",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Interest paid",
        "cols": {2015: "CF", 2021: "CH", 2022: "CT"},
    },
    # ── Home mortgage interest ───────────────────────────────────────────────
    {
        "var_name": "id_mortgage",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Home mortgage interest — number of returns",
        "cols": {2015: "CG", 2021: "CI", 2022: "CW"},
    },
    {
        "var_name": "id_mortgage",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Home mortgage interest",
        "cols": {2015: "CH", 2021: "CJ", 2022: "CX"},
    },
    # ── Charitable contributions ─────────────────────────────────────────────
    {
        "var_name": "id_contributions",
        "var_type": "count",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Charitable contributions — number of returns",
        "cols": {2015: "CS", 2021: "CW", 2022: "DG"},
    },
    {
        "var_name": "id_contributions",
        "var_type": "amount",
        "value_filter": "nz",
        "marstat": "all",
        "description": "Charitable contributions",
        "cols": {2015: "CT", 2021: "CX", 2022: "DH"},
    },
]


# ── Aggregate lookup ─────────────────────────────────────────────────────────

COLUMNS = {
    "tab11": TAB11_COLUMNS,
    "tab12": TAB12_COLUMNS,
    "tab14": TAB14_COLUMNS,
    "tab21": TAB21_COLUMNS,
}
