import os
from pathlib import Path
import json
from io import BytesIO
from zipfile import ZipFile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import taxcalc
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import (
    CREDIT_CLAIMING,
    CPS_FILER_MIN_INCOME,
    CPS_TAXABLE_INTEREST_FRACTION,
    CPS_QUALIFIED_DIVIDEND_FRACTION,
    CPS_TAXABLE_PENSION_FRACTION,
    CPS_LONG_TERM_CAPGAIN_FRACTION,
)

TAX_UNIT_COLUMNS = [
    "ACTC_CRD",
    "AGI",
    "CTC_CRD",
    "EIT_CRED",
    "FEDTAX_AC",
    "FEDTAX_BC",
    "MARG_TAX",
    "STATETAX_A",
    "STATETAX_B",
    "TAX_INC",
]
SPM_UNIT_COLUMNS = [
    "ACTC",
    "BBSUBVAL",
    "CAPHOUSESUB",
    "CAPWKCCXPNS",
    "CHILDCAREXPNS",
    "CHILDSUPPD",
    "EITC",
    "ENGVAL",
    "EQUIVSCALE",
    "FAMTYPE",
    "FEDTAX",
    "FEDTAXBC",
    "FICA",
    "GEOADJ",
    "HAGE",
    "HHISP",
    "HMARITALSTATUS",
    "HRACE",
    "MEDXPNS",
    "NUMADULTS",
    "NUMKIDS",
    "NUMPER",
    "POOR",
    "POVTHRESHOLD",
    "RESOURCES",
    "SCHLUNCH",
    "SNAPSUB",
    "STTAX",
    "TENMORTSTATUS",
    "TOTVAL",
    "WCOHABIT",
    "WEIGHT",
    "WFOSTER22",
    "WICVAL",
    "WKXPNS",
    "WNEWHEAD",
    "WNEWPARENT",
    "WUI_LT15",
    "ID",
]
SPM_UNIT_COLUMNS = ["SPM_" + column for column in SPM_UNIT_COLUMNS]
PERSON_COLUMNS = [
    "PH_SEQ",
    "PF_SEQ",
    "P_SEQ",
    "TAX_ID",
    "SPM_ID",
    "A_FNLWGT",
    "A_LINENO",
    "A_SPOUSE",
    "A_AGE",
    "A_SEX",
    "PEDISEYE",
    "MRK",
    "WSAL_VAL",
    "INT_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "SS_VAL",
    "UC_VAL",
    "ANN_VAL",
    "PNSN_VAL",
    "OI_OFF",
    "OI_VAL",
    "CSP_VAL",
    "PAW_VAL",
    "SSI_VAL",
    "RETCB_VAL",
    "CAP_VAL",
    "WICYN",
    "VET_VAL",
    "WC_VAL",
    "DIS_VAL1",
    "DIS_VAL2",
    "CHSP_VAL",
    "PHIP_VAL",
    "MOOP",
    "PEDISDRS",
    "PEDISEAR",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
    "PEPAR1",
    "PEPAR2",
    "DIS_SC1",
    "DIS_SC2",
    "DST_SC1",
    "DST_SC2",
    "DST_SC1_YNG",
    "DST_SC2_YNG",
    "DST_VAL1",
    "DST_VAL2",
    "DST_VAL1_YNG",
    "DST_VAL2_YNG",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "PERIDNUM",
    "I_ERNVAL",
    "I_SEVAL",
    "A_HSCOL",
    "HRSWK",
    "WKSWORK",
]

TAXCALC_CPS_AGED_RNG = np.random.default_rng(seed=374651932)

CPS_URL_BY_YEAR = {
    2018: (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        "2019/march/asecpub19csv.zip"
    ),
    2019: (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        "2020/march/asecpub20csv.zip"
    ),
    2020: (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        "2021/march/asecpub21csv.zip"
    ),
    2021: (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        "2022/march/asecpub22csv.zip"
    ),
    2022: (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        "2023/march/asecpub23csv.zip"
    ),
}


def _cached_raw_cps_path(taxyear: int):
    """Return path for the cached raw CPS HDF5 file for taxyear."""
    return STORAGE_FOLDER / "input" / f"raw_cps_{taxyear}.h5"


def _download_raw_cps(taxyear: int) -> str:
    """
    Download raw CPS ASEC data for taxyear and cache in an HDF5 file.
    Return the path to the HDF5 file.
    """
    h5path = _cached_raw_cps_path(taxyear)
    if os.path.exists(h5path):
        return h5path
    if taxyear not in CPS_URL_BY_YEAR:
        raise ValueError(f"No raw CPS data URL known for year {taxyear}.")
    url = CPS_URL_BY_YEAR[taxyear]
    file_year = taxyear + 1
    file_year_code = str(file_year)[-2:]
    spm_unit_columns = SPM_UNIT_COLUMNS
    if taxyear <= 2020:
        spm_unit_columns = [
            col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
        ]
    print(f"Downloading raw {taxyear} CPS ASEC data...")
    response = requests.get(
        url,
        stream=True,
        verify=False,
        timeout=(20, 600),
    )
    total_size_in_bytes = int(response.headers.get("content-length", 200e6))
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc="Downloading ASEC",
    )
    if response.status_code == 404:
        raise FileNotFoundError(
            "Received a 404 response when fetching the data."
        )
    with BytesIO() as file:
        content_length_actual = 0
        for data in response.iter_content(int(1e6)):
            progress_bar.update(len(data))
            content_length_actual += len(data)
            file.write(data)
        progress_bar.set_description("Downloaded ASEC")
        progress_bar.total = content_length_actual
        progress_bar.close()
        zipfile = ZipFile(file)  # pylint: disable=consider-using-with
        if file_year_code == "19":
            file_prefix = "cpspb/asec/prod/data/2019/"
        else:
            file_prefix = ""
        with zipfile.open(f"{file_prefix}pppub{file_year_code}.csv") as f:
            person = pd.read_csv(
                f,
                usecols=PERSON_COLUMNS + spm_unit_columns + TAX_UNIT_COLUMNS,
            ).fillna(0)
        with pd.HDFStore(str(h5path), mode="w") as storage:
            storage["person"] = person
    return h5path


def load_raw_cps_person_data(taxyear: int) -> pd.DataFrame:
    """
    Load the person data from the cached raw CPS HDF5 file.
    """
    h5path = _download_raw_cps(taxyear)
    with pd.HDFStore(str(h5path), mode="r") as storage:
        return storage["person"]


def _identify_head_spouse_dependent(person: pd.DataFrame):
    """
    For each person, determine whether they are the tax-unit head,
    spouse, or a dependent using CPS person variables.
    Returns three boolean numpy arrays: is_head, is_spouse, is_dependent.
    """
    # Within each tax unit (TAX_ID), the head is the person with
    # the lowest A_LINENO. The spouse is the person whose A_LINENO
    # matches the head's A_SPOUSE (if A_SPOUSE > 0). Everyone else
    # is a dependent.
    head_lineno = person.groupby("TAX_ID")["A_LINENO"].transform("min")
    is_head = (person["A_LINENO"] == head_lineno).values
    # Map each tax unit's head A_SPOUSE value to all persons in that unit
    head_a_spouse = person.loc[is_head, ["TAX_ID", "A_SPOUSE"]].set_index(
        "TAX_ID"
    )["A_SPOUSE"]
    head_a_spouse_per_person = person["TAX_ID"].map(head_a_spouse).values
    is_spouse = (
        ~is_head
        & (head_a_spouse_per_person > 0)
        & (person["A_LINENO"].values == head_a_spouse_per_person)
    )
    is_dependent = ~is_head & ~is_spouse
    return is_head, is_spouse, is_dependent


def _derive_filing_status(person: pd.DataFrame, is_head, is_spouse):
    """
    Derive MARS filing status for each tax unit from CPS A_MARITL.
    Returns an integer array with one entry per tax unit.
    MARS: 1=single, 2=joint, 3=separate, 4=head of household, 5=widow(er)
    """
    head_persons = person[is_head].copy().sort_values("TAX_ID")
    head_tax_ids = head_persons["TAX_ID"].values
    # Determine if tax unit has a spouse
    spouse_counts = person[is_spouse].groupby("TAX_ID").size()
    has_spouse = spouse_counts.reindex(head_tax_ids).fillna(0).values > 0
    # Determine if tax unit has dependents
    dep_counts = person[~is_head & ~is_spouse].groupby("TAX_ID").size()
    has_deps = dep_counts.reindex(head_tax_ids).fillna(0).values > 0
    # A_MARITL codes:
    # 1=married, civilian spouse present
    # 2=married, Armed Forces spouse present
    # 3=married, spouse absent (except separated)
    # 4=widowed
    # 5=divorced
    # 6=separated
    # 7=never married
    maritl = head_persons["A_MARITL"].values
    mars = np.ones(len(head_persons), dtype=int)  # default: single
    # Married filing jointly (has spouse present)
    mars[has_spouse] = 2
    # Widowed without spouse => surviving spouse if has dependents
    mars[(maritl == 4) & ~has_spouse & has_deps] = 5
    # Head of household: unmarried (or separated) with dependents
    unmarried = np.isin(maritl, [4, 5, 6, 7])
    mars[unmarried & ~has_spouse & has_deps & (mars != 5)] = 4
    # Single: unmarried without dependents (already default=1)
    return mars


def _derive_age(person: pd.DataFrame) -> np.ndarray:
    """
    Apply the same age-80 randomization as the existing CPS code.
    """
    return np.where(
        person.A_AGE == 80,
        TAXCALC_CPS_AGED_RNG.integers(
            low=80, high=85, endpoint=False, size=len(person)
        ),
        person.A_AGE.values,
    )


def _is_tax_filer(tcdf: pd.DataFrame, taxyear: int) -> pd.Series:
    """
    This function approximates the tax filer logic used by the Census
    ASEC Tax Model, documentation for which is at this URL:
    https://www.census.gov/content/dam/Census/library/working-papers/2022/demo/
            sehsd-wp2022-18.pdf

    From page 15 of that documentation, we have this text:
     The CPS ASEC Tax Model defines a set of filing requirements.
     The tax model assumes a tax unit files a return if it meets
     at least one of the following requirements:
     (1) income above IRS filing threshold determined by age and filing status;
     (2) positive Earned Income Tax Credit (EITC);
     (3) positive self-employment income;
     (4) gross income less than $0;
     (5) self-employment income less than $0;
     (6) positive Additional Child Tax Credit;
     (7) positive self-employment income for either spouse; or
     (8) has total income above $2,000.
    """
    assert taxyear in [2021, 2022]
    income = (
        tcdf["e00200"]
        + tcdf["e00300"]
        + tcdf["e00600"]
        + tcdf["e00800"]
        + tcdf["e00900"]
        + tcdf["e01400"]
        + tcdf["e01500"]
        + tcdf["e02100"]
        + tcdf["e02300"]
        + tcdf["e02400"]
        + tcdf["p22250"]
        + tcdf["p23250"]
    )
    rec = taxcalc.Records(
        data=tcdf,
        start_year=taxyear,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
        exact_calculations=True,
        weights_scale=1.0,
    )
    pol = taxcalc.Policy()
    pol.implement_reform(CREDIT_CLAIMING)
    calc = taxcalc.Calculator(records=rec, policy=pol)
    calc.advance_to_year(taxyear)
    calc.calc_all()
    output = calc.dataframe(["eitc", "c11070"])
    filer = income > CPS_FILER_MIN_INCOME  # req (1) and (8)
    filer |= output["eitc"] > 0  # req (2)
    filer |= (tcdf["e00900p"] > 0) | (tcdf["e00900s"] > 0)  # req (3) and (7)
    filer |= income < 0  # req (4)
    filer |= tcdf["e00900"] < 0  # req (5)
    if taxyear != 2021:
        # skip because CTC was more generous and fully refundable in 2021
        filer |= output["c11070"] > 0  # req (6)
    return filer


def create_taxcalc_cps(taxyear: int) -> (pd.DataFrame, pd.Series):
    """
    Create a Tax-Calculator-compatible CPS DataFrame for the given taxyear
    directly from the Census raw CPS data.
    """
    person = load_raw_cps_person_data(taxyear)
    print(f"Creating CPS dataframe for year {taxyear}...")

    # identify head, spouse, dependent for each person
    is_head, is_spouse, is_dependent = _identify_head_spouse_dependent(person)
    is_non_dep = ~is_dependent
    age = _derive_age(person)

    # helper: sum a person-level array over nondependents per tax unit
    tax_unit_ids = person["TAX_ID"].values

    def sum_nondep(values):
        s = pd.Series(values * is_non_dep).groupby(tax_unit_ids).sum()
        return s.values

    def sum_all(values):
        return pd.Series(values).groupby(tax_unit_ids).sum().values

    def map_head(values):
        return pd.Series(values * is_head).groupby(tax_unit_ids).sum().values

    def map_spouse(values):
        return pd.Series(values * is_spouse).groupby(tax_unit_ids).sum().values

    # number of tax units
    tu_ids_unique = pd.Series(tax_unit_ids).groupby(tax_unit_ids).first()
    n_tu = len(tu_ids_unique)
    zeros = np.zeros(n_tu, dtype=int)
    ones = np.ones(n_tu, dtype=int)

    # derive tax filing status
    mars = _derive_filing_status(person, is_head, is_spouse)

    # derive tax-unit weight from person weight
    # (use the head's A_FNLWGT / 100 as the tax-unit weight)
    tu_weight = map_head(person["A_FNLWGT"].values) / 1e2

    # income variables from person table
    employment_income = person["WSAL_VAL"].values
    self_employment_income = person["SEMP_VAL"].values
    farm_income = person["FRSE_VAL"].values
    interest_income = person["INT_VAL"].values
    dividend_income = person["DIV_VAL"].values
    rental_income = person["RNT_VAL"].values
    ss_val = person["SS_VAL"].values
    uc_val = person["UC_VAL"].values
    pension_val = person["PNSN_VAL"].values + person["ANN_VAL"].values
    cap_val = person["CAP_VAL"].values
    oi_off = person["OI_OFF"].values
    oi_val = person["OI_VAL"].values
    retcb_val = person["RETCB_VAL"].values

    # build the Tax-Calculator variable dictionary
    var = {}
    var["RECID"] = tu_ids_unique.values
    var["S006"] = tu_weight
    var["FLPDYR"] = ones * taxyear
    var["MARS"] = mars
    var["data_source"] = zeros  # CPS data

    # ... employment income (sum of nondependents)
    var["E00200"] = sum_nondep(employment_income)
    var["e00200p"] = map_head(employment_income)
    var["e00200s"] = map_spouse(employment_income)

    # ... self-employment income
    var["E00900"] = sum_nondep(self_employment_income)
    var["e00900p"] = map_head(self_employment_income)
    var["e00900s"] = map_spouse(self_employment_income)

    # ... farm income
    var["E02100"] = sum_nondep(farm_income)
    var["e02100p"] = map_head(farm_income)
    var["e02100s"] = map_spouse(farm_income)

    # ... interest income
    var["E00300"] = sum_nondep(interest_income * CPS_TAXABLE_INTEREST_FRACTION)
    var["E00400"] = sum_nondep(
        interest_income * (1 - CPS_TAXABLE_INTEREST_FRACTION)
    )

    # ... dividend income
    var["E00650"] = sum_nondep(
        dividend_income * CPS_QUALIFIED_DIVIDEND_FRACTION
    )
    non_qual_div = sum_nondep(
        dividend_income * (1 - CPS_QUALIFIED_DIVIDEND_FRACTION)
    )
    var["E00600"] = non_qual_div + var["E00650"]

    # ... rental income (included in e02000)
    var_rental = sum_nondep(rental_income)

    # ... social security
    var["E02400"] = sum_nondep(ss_val)

    # ... unemployment compensation
    var["E02300"] = sum_nondep(uc_val)

    # ... pensions and annuities
    taxable_pension = sum_nondep(pension_val * CPS_TAXABLE_PENSION_FRACTION)
    tax_exempt_pension = sum_nondep(
        pension_val * (1 - CPS_TAXABLE_PENSION_FRACTION)
    )
    var["E01700"] = taxable_pension
    var["E01500"] = taxable_pension + tax_exempt_pension

    # ... capital gains
    var["P23250"] = sum_nondep(cap_val * CPS_LONG_TERM_CAPGAIN_FRACTION)
    var["P22250"] = sum_nondep(cap_val * (1 - CPS_LONG_TERM_CAPGAIN_FRACTION))

    # ... alimony income (OI_OFF code 20)
    var["E00800"] = sum_nondep((oi_off == 20) * oi_val)

    # ... IRA distributions
    RETIREMENT_CODES = {
        1: "401k",
        2: "403b",
        3: "roth_ira",
        4: "regular_ira",
        5: "keogh",
        6: "sep",
        7: "other_type_retirement_account",
    }
    retirement_dist = {}
    for code, description in RETIREMENT_CODES.items():
        tmp = np.zeros(len(person))
        for i in ["1", "2", "1_YNG", "2_YNG"]:
            tmp += (person["DST_SC" + i].values == code) * (
                person["DST_VAL" + i].values
            )
        retirement_dist[description] = tmp

    var["E01400"] = sum_nondep(retirement_dist["regular_ira"])

    # ... IRA and 401(k) contributions
    LIMIT_401K = 20_500
    LIMIT_401K_CATCH_UP = 6_500
    LIMIT_IRA = 6_000
    LIMIT_IRA_CATCH_UP = 1_000
    CATCH_UP_AGE = 50
    se_pension = np.where(self_employment_income > 0, retcb_val, 0)
    remaining = np.maximum(retcb_val - se_pension, 0)
    catch_up_eligible = person["A_AGE"].values >= CATCH_UP_AGE
    limit_401k = LIMIT_401K + catch_up_eligible * LIMIT_401K_CATCH_UP
    limit_ira = LIMIT_IRA + catch_up_eligible * LIMIT_IRA_CATCH_UP
    trad_401k = np.where(
        employment_income > 0,
        np.minimum(remaining, limit_401k),
        0,
    )
    remaining = np.maximum(remaining - trad_401k, 0)
    roth_401k = np.where(
        employment_income > 0,
        np.minimum(remaining, limit_401k),
        0,
    )
    remaining = np.maximum(remaining - roth_401k, 0)
    trad_ira_contrib = np.where(
        employment_income > 0,
        np.minimum(remaining, limit_ira),
        0,
    )
    var["E03150"] = sum_nondep(trad_ira_contrib)

    # ... self-employed pension contribution deduction
    var["E03300"] = sum_nondep(se_pension)

    # ... e02000 = rental + partnership_s_corp + estate + farm_rent
    #     (CPS has no partnership/estate/farm_rent, so just rental income)
    var["e02000"] = var_rental

    # ... pension contributions (zero for CPS)
    var["pencon_p"] = np.zeros(n_tu)
    var["pencon_s"] = np.zeros(n_tu)

    # ... demographics variables
    var["age_head"] = map_head(age)
    var["age_spouse"] = map_spouse(age)
    blind = (person["PEDISEYE"].values == 1).astype(float)
    var["blind_head"] = map_head(blind)
    var["blind_spouse"] = map_spouse(blind)

    dep_flag = is_dependent.astype(float)
    var["nu18"] = sum_all((age < 18) * dep_flag)
    var["nu13"] = sum_all((age < 13) * dep_flag)
    var["nu06"] = sum_all((age < 6) * dep_flag)
    var["n1820"] = sum_all(((age >= 18) & (age < 21)) * dep_flag)
    var["n21"] = sum_all((age >= 21) * dep_flag)
    var["n24"] = sum_all((age < 17) * dep_flag)
    var["elderly_dependents"] = sum_all((age >= 65) * dep_flag)

    # ... total exemptions = number of non-dependents + dependents in unit
    var["XTOT"] = sum_all(np.ones(len(person)))

    # ... EIC: number of EITC qualifying children (max 3)
    #     (heuristic: dependents under 19, or under 24 if full-time student)
    is_student = person["A_HSCOL"].values == 2
    eitc_qual = dep_flag * ((age < 19) | ((age < 24) & is_student))
    var["EIC"] = np.minimum(sum_all(eitc_qual), 3).astype(int)

    # ... f2441: count of CDCC qualifying persons (dependents under 13)
    var["f2441"] = sum_all((age < 13) * dep_flag).astype(int)

    # variables with no CPS source are set to zero
    zero_taxcalc_names = [
        "a_lineno",
        "agi_bin",
        "h_seq",
        "ffpos",
        "fips",
        "DSI",
        "MIDR",
        "PT_SSTB_income",
        "PT_ubia_property",
        "PT_binc_w2_wages",
        "cmbtp",
        "f6251",
        "k1bx14p",
        "k1bx14s",
        "tanf_ben",
        "vet_ben",
        "wic_ben",
        "snap_ben",
        "housing_ben",
        "ssi_ben",
        "mcare_ben",
        "mcaid_ben",
        "other_ben",
        "E03500",  # alimony expense
        "G20500",  # casualty loss
        "E32800",  # cdcc relevant expenses
        "E19800",  # charitable cash donations
        "E20100",  # charitable non-cash donations
        "E03240",  # domestic production ald
        "E03400",  # early withdrawal penalty
        "E03220",  # educator expense
        "E03290",  # health savings account ald
        "E19200",  # interest deduction
        "E24518",  # long-term cap gains on collectibles
        "E17500",  # medical expense
        "E26270",  # partnership/s-corp income
        "E03230",  # qualified tuition expenses
        "e87530",  # qualified tuition expenses (lowercase)
        "E18500",  # real estate taxes
        "E03270",  # self-employed health insurance ald
        "E18400",  # state and local sales/income tax
        "E03210",  # student loan interest
        "E24515",  # unrecaptured section 1250 gain
        "E27200",  # farm rent income
        "e20400",  # misc deduction
        "e07300",  # foreign tax credit
        "e62900",  # AMT foreign tax credit
        "e01200",  # miscellaneous income
        "e00700",  # salt refund income
        "e58990",  # investment income elected form 4952
        "e07400",  # general business credit
        "e07600",  # prior year minimum tax credit
        "e11200",  # excess withheld payroll tax
        "e01100",  # non-sch D capital gains
        "e87521",  # american opportunity credit
        "e07260",  # energy efficient home improvement credit
        "e09900",  # qualified retirement penalty
        "p08000",  # other credits
        "e07240",  # savers credit
        "e09700",  # recapture of investment credit
        "e09800",  # unreported payroll tax
    ]
    for tcname in zero_taxcalc_names:
        if tcname not in var:
            var[tcname] = zeros

    # ... create DataFrame
    tcdf = pd.DataFrame(var)

    # correct variable name casing for Tax-Calculator
    json_file_path = Path(taxcalc.__file__).parent / "records_variables.json"
    with open(json_file_path, "r", encoding="utf-8") as jfile:
        taxcalc_variable_metadata = json.load(jfile)
    renames = {}
    for variable in tcdf.columns:
        if variable.upper() in taxcalc_variable_metadata["read"]:
            renames[variable] = variable.upper()
        elif variable.lower() in taxcalc_variable_metadata["read"]:
            renames[variable] = variable.lower()
    tcdf.rename(columns=renames, inplace=True)

    # drop tax units with zero weight
    nonzero_wt = tcdf["s006"] > 0
    tcdf = tcdf[nonzero_wt].reset_index(drop=True)

    # identify tax filers and nonfilers
    filer = _is_tax_filer(tcdf, taxyear)
    nonfiler = ~filer

    # return Tax-Calculator DataFrame and nonfiler boolean Series
    return tcdf, nonfiler
