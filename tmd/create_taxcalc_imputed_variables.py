"""
Augment tmd.csv, a Tax-Calculator-style input variable file for TAXYEAR, that
is created in the create_taxcalc_input_variables.py module, with imputed
values of overtime_income and tip_income (derived from SIPP data) and
imputed values of auto_loan_interest (derived from CEX data), writing the
augmented data to the tmda.csv.gz file in the tmd/storage/output folder.
"""

import sys
import shutil
import sqlite3
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from tmd.imputation_assumptions import (
    TAXYEAR,
    OTM_convert_zero_prob,
    OTM_scale,
    TIP_convert_zero_prob,
    TIP_scale,
    ALI_convert_zero_prob,
    ALI_scale,
)
from tmd.storage import STORAGE_FOLDER
from tmd.utils.mice import MICE

TMD_PATH = STORAGE_FOLDER / "output" / "tmd.csv.gz"
TMD_YEAR = TAXYEAR
SIPP_PATH = STORAGE_FOLDER / "input" / "SIPP24" / "pu2024.csv.gz"
SIPP_YEAR = 2023  # 2024 SIPP data contain calandar year 2023 information
CEX_FOLDER = STORAGE_FOLDER / "input" / "CEX23"
CEX_FILES = [
    "fmli232.csv.gz",
    "fmli233.csv.gz",
    "fmli234.csv.gz",
    "fmli241.csv.gz",
]
CEX_YEAR = 2023
TMD_GROWFACTORS_PATH = STORAGE_FOLDER / "output" / "tmd_growfactors.csv"


def growfactor(name: str, year1: int, year2: int) -> float:
    """
    Returns compound growth factor from year1 to year2 for named factor.
    """
    assert year1 <= year2
    columns_to_read = ["YEAR", name]
    gfdf = pd.read_csv(TMD_GROWFACTORS_PATH, usecols=columns_to_read)
    gfdf.set_index("YEAR", inplace=True)
    cgf = 1.0
    for year in range(year1 + 1, year2 + 1):
        cgf *= gfdf.loc[year, name]
    return cgf


def read_sipp_for_imputation() -> pd.DataFrame:
    """
    Return SIPP dataframe with one row per individual with positive weight.
        SIPP data is for each month over the 2023 year with a
        row for each month for each individual and with separate
        monthly variables for each job held by the individual
        during the month with the jobs numbered from 1 thru 7.
    """

    def cols_to_read(
        header_columns: List[str],  # SIPP file header contents
        job_slots: List[int],  # list of job slot numbers
        ind_cols: List[str],  # non-job SIPP variables
    ) -> List[str]:
        """
        Determine which columns to read from SIPP file given
        SIPP header columns, available job slot numbers, and ind_cols.
        """
        job_cols = []
        for slot_num in job_slots:
            for suffix in ("MSUM", "TXAMT", "OXAMT"):
                colname = f"TJB{slot_num}_{suffix}"
                if colname in header_columns:
                    job_cols.append(colname)
            colname = f"EJB{slot_num}_CLWRK"
            if colname in header_columns:
                job_cols.append(colname)
        cols = [c for c in ind_cols + job_cols if c in header_columns]
        return cols

    def sum_job_arrays(
        xdf: pd.DataFrame,  # SIPP extract dataframe
        job_slot_numbers: List[int],  # job slot numbers
        variable_suffix: str,  # suffix for wage, overtime or tip variable
        selfemp_mask: Dict[int, np.ndarray],  # self-employment indicators
    ) -> np.ndarray:
        """
        Aggregate job-level columns across job slots into a single array.
        """
        tot = np.zeros(len(xdf), dtype="float64")
        for job_slot_number in job_slot_numbers:
            colname = f"TJB{job_slot_number}_{variable_suffix}"
            values = (
                pd.to_numeric(sdf[colname], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype="float64")
            )
            mask = selfemp_mask.get(job_slot_number)
            values = np.where(mask, 0.0, values)
            tot += values
        return tot

    # begin main read_sipp_for_imputation function logic
    job_slot_numbers = [1, 2, 3, 4, 5, 6, 7]  # as many as seven jobs per month
    header = pd.read_csv(SIPP_PATH, sep="|", nrows=0)
    ind_cols = ["SSUID", "PNUM", "WPFINWGT", "TAGE", "EMS"]
    missing = set(ind_cols).difference(header.columns)
    if missing:
        raise ValueError(f"Missing SIPP individual columns: {missing}")
    columns_to_read = cols_to_read(header.columns, job_slot_numbers, ind_cols)

    sdf = pd.read_csv(SIPP_PATH, sep="|", usecols=columns_to_read)

    # build a self-employment mask for each job slot
    # > codes in the "EJB{job_slot_number}_CLWRK" variable are as follows:
    # > 1: Federal government employee
    # > 2: Active duty military
    # > 3: State government employee
    # > 4: Local government employee
    # > 5: Employee of a private, for-profit company
    # > 6: Employee of a private, not-for-profit company
    # > 7: Self-employed in own incorporated business
    # > 8: Self-employed in own not incorporated business
    # > exclude self-employed codes 7 and 8 from wage,otime,tip aggregates
    selfemp_mask: Dict[int, np.ndarray] = {}
    for job_slot_number in job_slot_numbers:
        colname = f"EJB{job_slot_number}_CLWRK"
        val = pd.to_numeric(sdf[colname], errors="coerce").to_numpy(
            dtype="float64"
        )
        selfemp_mask[job_slot_number] = np.isin(val, [7, 8])

    # aggregate the several job amounts into sdf-length individual-month arrays
    wage = sum_job_arrays(sdf, job_slot_numbers, "MSUM", selfemp_mask)
    otm = sum_job_arrays(sdf, job_slot_numbers, "OXAMT", selfemp_mask)
    tip = sum_job_arrays(sdf, job_slot_numbers, "TXAMT", selfemp_mask)
    if not np.any(wage) and not np.any(otm) and not np.any(tip):
        raise ValueError(
            "Encountered a SIPP sdf without any TJB-prefixed wage or "
            "overtime or tip data; "
            "ensure the SIPP extract contains TJB columns."
        )

    # specify non-job data as sdf-length individual-month arrays
    ssuid = (
        pd.to_numeric(sdf["SSUID"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int64")
    )
    pnum = (
        pd.to_numeric(sdf["PNUM"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int64")
    )
    wght = (
        pd.to_numeric(sdf["WPFINWGT"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    age = (
        pd.to_numeric(sdf["TAGE"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int32")
    )
    ems = (
        pd.to_numeric(sdf["EMS"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int32")
    )
    mars2 = np.where(ems == 1, 1, 0)  # TMD MARS==2 dummy variable

    # aggregate individual-month arrays into an individual-year dataframe
    people: Dict[Tuple[int, int], Dict[str, float]] = {}
    for ss, pn, wt, age_val, mars2_val, wage_val, tip_val, otm_val in zip(
        ssuid, pnum, wght, age, mars2, wage, tip, otm
    ):
        assert ss > 0 and pn > 0
        key = (ss, pn)
        rec = people.get(key)
        if rec is None:
            rec = {
                "wage": 0.0,
                "tip_amt": 0.0,
                "otm_amt": 0.0,
                "age": 0,
                "mars2": 0,
                "weight": 0.0,
            }
            people[key] = rec
        rec["wage"] += wage_val
        rec["otm_amt"] += otm_val
        rec["tip_amt"] += tip_val
        rec["age"] = age_val
        rec["mars2"] = mars2_val
        rec["weight"] = wt
    if not people:
        raise ValueError("SIPP reading produced no individual annual records")
    adf = pd.DataFrame.from_records(list(people.values()))

    # remove individual-year records with zero weight
    adf = adf[adf["weight"] > 0].copy()

    # return individual-year dataframe
    return adf


def read_tmd_for_sipp_imputation() -> pd.DataFrame:
    """
    Return TMD dataframe suitable for use in imputation using SIPP data.
    """

    def convert_tudf_to_idf(tudf: pd.DataFrame) -> pd.DataFrame:
        """
        Return individual dataframe converted from tax-unit dataframe, tudf.
        """
        # create SQLite3 database with unit table containing tudf data
        conn = sqlite3.connect(":memory:")
        tudf.to_sql("unit", conn, index=False)
        # create empty idf-style database table indv
        sql = """
        CREATE TABLE indv (
          RECID          INTEGER  CHECK(RECID>=1),
          weight         REAL     CHECK(weight>0),
          mars2          INTEGER  CHECK(mars2>=0 AND mars2<=1),
          age            INTEGER  CHECK(age>=0),
          e00200         REAL     CHECK(e00200>=0),
          overtime_frac  REAL     CHECK(overtime_frac>=0),
          tip_frac       REAL     CHECK(tip_frac>=0)
        )
        """
        conn.execute(sql)
        conn.commit()
        # populate indv table using unit table information
        sql = """
        INSERT INTO indv
          SELECT -- rows for married heads
            RECID, s006, 1, age_head, e00200p, 0.0, 0.0
          FROM unit WHERE mars=2
          UNION ALL
          SELECT -- rows for married spouses
            RECID, s006, 1, age_spouse, e00200s, 0.0, 0.0
          FROM unit WHERE mars=2
          UNION ALL
          SELECT -- rows for unmarried individuals
            RECID, s006, 0, age_head, e00200p, 0.0, 0.0
          FROM unit WHERE mars!=2
        """
        conn.executescript(sql)
        conn.commit()
        # convert indv table to idf dataframe
        idf = pd.read_sql_query("SELECT * FROM indv", conn)
        # close in-memory database and return idf dataframe
        conn.close()
        return idf

    # begin main read_tmd_for_sipp_imputation function logic
    udf = pd.read_csv(TMD_PATH)
    # edit udf removing unneeded variables
    needed_vars = [
        "RECID",
        "s006",
        "MARS",
        "age_head",
        "age_spouse",
        "e00200p",
        "e00200s",
        "overtime_income",
        "tip_income",
    ]
    tudf = udf[needed_vars].copy()
    del udf
    # convert tax-unit dataframe into an individual idf dataframe
    idf = convert_tudf_to_idf(tudf)
    # return idf dataframe
    return idf


def prep_sipp_for_imputation(adf: pd.DataFrame) -> pd.DataFrame:
    """
    Return SIPP dataframe suitable for imputating missing TMD variables
    using annual SIPP dataframe generated by read_sipp_for_imputation().
    """
    # compute overtime and tip fractions of e00200 using SIPP_YEAR values
    # (those with zero e00200 are assigned zero otm and tip fractions)
    wage = adf["wage"].to_numpy(dtype="float64")
    otm = adf["otm_amt"].to_numpy(dtype="float64")
    tip = adf["tip_amt"].to_numpy(dtype="float64")
    bad_data_mask = (wage <= 0.0) & ((otm + tip) > 0.0)
    assert np.all(~bad_data_mask)
    e00200 = wage + otm + tip
    assert np.all(e00200 >= 0.0)
    valid_division_mask = e00200 > 0.0
    otm_frac = np.zeros_like(otm)
    np.divide(otm, e00200, out=otm_frac, where=valid_division_mask)
    tip_frac = np.zeros_like(tip)
    np.divide(tip, e00200, out=tip_frac, where=valid_division_mask)

    # convert e00200 amount from SIPP_YEAR to TMD_YEAR
    e00200 /= growfactor("AWAGE", TMD_YEAR, SIPP_YEAR)

    # construct prepared SIPP dataframe
    df_dict = {
        "RECID": np.zeros_like(wage, dtype="int32"),
        "weight": adf["weight"].to_numpy(dtype="float64"),
        "mars2": adf["mars2"].to_numpy(dtype="int32"),
        "age": adf["age"].to_numpy(dtype="int32"),
        "e00200": e00200,
        "overtime_frac": otm_frac,
        "tip_frac": tip_frac,
    }
    return pd.DataFrame(df_dict)


def read_cex_for_imputation() -> pd.DataFrame:
    """
    Return CEX dataframe with one row per unit with positive weight.
    The returned dataframe contains units from all four quarterly surveys.
    The income and auto loan interest variables are scaled from CEX_YEAR
    to TMD_YEAR values.
    """
    # read quarterly files into list of quarterly dataframes
    qframes = []
    for fname in CEX_FILES:
        qdf = pd.read_csv(
            CEX_FOLDER / fname,
            usecols=[
                "FINLWT21",  # sampling weight of unit in quarterly survey
                "AGE_REF",  # age of reference person in unit
                "MARITAL1",  # MARITAL1==1 implied married
                "FINCBTAX",  # annual income before tax
                "VEHFINCQ",  # quarterly vehicle finance charges
            ],
            low_memory=False,
        ).rename(
            columns={
                "FINLWT21": "weight",
                "AGE_REF": "age",
                "MARITAL1": "marital_status",
                "FINCBTAX": "income",
                "VEHFINCQ": "auto_loan_interest",
            }
        )
        qframes.append(qdf)

    # combine quarterly dataframes
    xdf = pd.concat(qframes, ignore_index=True)

    # convert variables to numeric values
    xdf["weight"] = (
        pd.to_numeric(xdf["weight"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        * 0.25  # scale by 1/4 because quarterly weights give national totals
    )
    cdf = xdf[xdf["weight"] > 0.0].copy()
    del xdf
    cdf["age"] = pd.to_numeric(cdf["age"], errors="coerce")
    cdf["mars2"] = np.where(cdf["marital_status"] == 1, 1, 0)
    cdf.drop(columns=["marital_status"], inplace=True)
    cdf["income"] = (
        pd.to_numeric(cdf["income"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    cgf = growfactor("AWAGE", TMD_YEAR, CEX_YEAR)
    cdf["income"] = (cdf["income"] / cgf).round()
    cdf["auto_loan_interest"] = (
        pd.to_numeric(cdf["auto_loan_interest"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        * 4.0  # scale by 4 because auto_loan_interest is a quarterly variable
    )
    cgf = growfactor("ATXPY", TMD_YEAR, CEX_YEAR)
    cdf["auto_loan_interest"] = (cdf["auto_loan_interest"] / cgf).round()
    cdf["RECID"] = 0  # set RECID values to zero in order to identify CEX units

    # return dataframe in specified column_order
    column_order = [
        "RECID",
        "weight",
        "mars2",
        "age",
        "income",
        "auto_loan_interest",
    ]
    return cdf[column_order]


def read_tmd_for_cex_imputation(column_order: list[str]) -> pd.DataFrame:
    """
    Return TMD dataframe suitable for use in imputation using CEX data.
    """
    core_needed_vars = [
        "RECID",
        "s006",
        "MARS",
        "age_head",
        "auto_loan_interest",
    ]
    # specify components of constructed income variable
    income_components = [
        "e00200",  # wages and salaries
        "pencon_p",  # head DC pension contribution
        "pencon_s",  # spouse DC pension contribution
        "e00300",  # taxable interest income
        "e00400",  # tax-exempt interest income
        "e00600",  # dividends included in AGI
        "e00900",  # Sch C business net profit/loss for filing unit
        "e01400",  # taxable IRA distributions
        "e01500",  # total pensions and annuities
        "e02000",  # Sch E rental, royalty, partnership, S-corp income/loss
        "e02100",  # Sch F farm net income/loss
        "e02300",  # unemployment insurance benefits
        "e02400",  # social security (OASDI) benefits
    ]
    used_cols = core_needed_vars + income_components
    udf = pd.read_csv(TMD_PATH, usecols=used_cols)
    # rename some variables
    udf.rename(columns={"s006": "weight", "age_head": "age"}, inplace=True)
    # construct mars2 variable
    udf["mars2"] = np.where(udf["MARS"] == 2, 1, 0)
    udf.drop(columns=["MARS"], inplace=True)
    # construct income variable and drop income components
    income = np.zeros_like(udf["RECID"])
    for icomp in income_components:
        income += udf[icomp].to_numpy()
    udf["income"] = np.round(np.clip(income, 0.0, np.inf)).astype("int64")
    udf.drop(columns=income_components, inplace=True)
    # specify auto_loan_interest as missing
    udf["auto_loan_interest"] = np.nan
    # return udf dataframe using specified column_order
    return udf[column_order]


def print_weighted_bin_percents(
    xdf: pd.DataFrame,
    varname: str,
    label: str,
) -> None:
    """
    Print weighted _frac bin percents for varname tabulated from xdf.
    """
    print(f"Weighed distribution of {varname}:")
    print("dataframe.shape=", xdf.shape)
    wght = xdf["weight"].to_numpy(dtype="float64")
    wght_tot = wght.sum()
    var = xdf[varname].to_numpy(dtype="float64")
    tot_bin_pct = 0.0
    bin_edges = [-1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 999.0]
    for idx in range(0, len(bin_edges) - 1):
        bot = bin_edges[idx]
        top = bin_edges[idx + 1]
        bin_mask = (var > bot) & (var <= top)
        bin_pct = 100.0 * wght[bin_mask].sum() / wght_tot
        tot_bin_pct += bin_pct
        print(f"{label}:bot,top,pct= >{bot} <={top} ==> {bin_pct:.3f}")
    print(f"all_bin_pct= {tot_bin_pct:.3f}")


def print_sipp_idf_tabulations(idf: pd.DataFrame) -> None:
    """
    Print various idf tabulations.
    """
    wght = idf["weight"].to_numpy(dtype="float64")
    e00200 = idf["e00200"].to_numpy(dtype="float64")
    print(f"wghted_e00200($b)= {((wght * e00200).sum() * 1e-9):.3f}")
    otm_f = idf["overtime_frac"].to_numpy(dtype="float64")
    otm = otm_f * e00200
    print(f"wghted_otm($b)= {((wght * otm).sum() * 1e-9):.3f}")
    tip_f = idf["tip_frac"].to_numpy(dtype="float64")
    tip = tip_f * e00200
    print(f"wghted_tip($b)= {((wght * tip).sum() * 1e-9):.3f}")
    cnt, bins = np.histogram(otm_f)
    print("otm_f.cnt=", cnt)
    print("otm_f.bin=", bins)
    print("otm_f.max=", np.amax(otm_f))
    cnt, bins = np.histogram(tip_f)
    print("tip_f.cnt=", cnt)
    print("tip_f.bin=", bins)
    print("tip_f.max=", np.amax(tip_f))
    dump_high_frac_rows = False
    if dump_high_frac_rows:
        hdf = idf[idf["overtime_frac"] >= 0.5]
        print("high otm_frac rows:\n", hdf.head())
        hdf = idf[idf["tip_frac"] >= 0.5]
        print("high tip_frac rows:\n", hdf.head())
    # tabulate weighted _frac distribution for those with positive e00200
    xdf = idf[idf["e00200"] > 0.0].copy()
    print_weighted_bin_percents(xdf, "overtime_frac", "otm_f")
    print_weighted_bin_percents(xdf, "tip_frac", "tip_f")


def create_sipp_imputed_tmd(
    tmd_idf: pd.DataFrame,
    sipp_idf: pd.DataFrame,
    verbose: bool,
) -> pd.DataFrame:
    """
    Use MICE algorithm to impute overtime_income and tip_income values
    from SIPP individual data to TMD individual data, returning a TMD
    tax-unit dataframe containing imputed variable values.
    """

    def augmented_taxunit_tmd_imputed_dataframe(
        aug_tmd_idf: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert aug_tmd_idf to augmented tax-unit dataframe while
        translating individual _frac variables to individual _income variables.
        """
        # create SQLite3 database with indx table containing aug_tmd_idf data
        conn = sqlite3.connect(":memory:")
        aug_tmd_idf.to_sql("indx", conn, index=False)
        # create empty taxunit-style database table unit
        sql = """
        CREATE TABLE unit (
          RECID            INTEGER  CHECK(RECID>=1),
          overtime_income  REAL     CHECK(overtime_income>=0),
          tip_income       REAL     CHECK(tip_income>=0)
        )
        """
        conn.execute(sql)
        conn.commit()
        # populate unit table using indx table information
        sql = """
        INSERT INTO unit
          SELECT
            RECID,
            SUM(overtime_frac * e00200) AS overtime_income,
            SUM(tip_frac * e00200) AS tip_income
          FROM indx
          GROUP BY RECID
        """
        conn.execute(sql)
        conn.commit()
        # convert unit table to udf dataframe
        udf = pd.read_sql_query("SELECT * FROM unit", conn)
        # close in-memory database and return udf dataframe
        conn.close()
        return udf

    # begin main function logic
    # split tmd_idf frame into zero e00200 and positive e00200 frames
    zero_tmd_idf = tmd_idf[tmd_idf["e00200"] <= 0.0].copy()
    pos_tmd_idf = tmd_idf[tmd_idf["e00200"] > 0.0].copy()
    # specify positive e00200 SIPP frame
    pos_sipp_idf = sipp_idf[sipp_idf["e00200"] > 0.0].copy()
    # use MICE with pos_sipp_idf to impute values to pos_tmd_idf
    pos_tmd_idf["overtime_frac"] = np.nan
    pos_tmd_idf["tip_frac"] = np.nan
    mdf = pd.concat([pos_sipp_idf, pos_tmd_idf], axis=0).copy()
    colnames = mdf.columns.tolist()
    idx_order = [colnames.index("overtime_frac"), colnames.index("tip_frac")]
    ignore_vars = [colnames.index("RECID"), colnames.index("weight")]
    mice = MICE(
        mdf.shape[0],
        mdf.shape[1],
        idx_order,
        ignore_vars,
        monotone=True,
        iters=1,  # MICE M (number of iterations) is 1 because monotone=True
        verbose=verbose,  # if True, write impute progress to stdout
        seed=192837465,
        # post-MICE imputation adjustment parameters:
        # ... convert zero to nonzero with prob
        convert_zero_prob=[
            OTM_convert_zero_prob[TAXYEAR],
            TIP_convert_zero_prob[TAXYEAR],
        ],
        # ... multiplicative scaling done after convert
        scale=[
            OTM_scale[TAXYEAR],
            TIP_scale[TAXYEAR],
        ],
    )
    iarray = mice.impute(mdf.to_numpy())
    idf = pd.DataFrame(iarray, columns=mdf.columns)
    pos_tmd_idf = idf[idf["RECID"] > 0]  # remove SIPP data
    # cap imputed overtime_frac and tip_frac values
    for col in ["overtime_frac", "tip_frac"]:
        pos_tmd_idf.loc[:, col] = pos_tmd_idf[col].clip(lower=0.0, upper=0.5)
    if verbose:
        print_weighted_bin_percents(pos_tmd_idf, "overtime_frac", "I:otm_frac")
        print_weighted_bin_percents(pos_tmd_idf, "tip_frac", "I:tip_frac")
    # recombine zero_tmd_idf and pos_tmd_idf into aug_tmd_idf frame
    aug_tmd_idf = pd.concat([zero_tmd_idf, pos_tmd_idf], axis=0)
    # convert aug_tmd_idf to augmented tax-unit tmd_udf dataframe
    # (where tmd_udf contains only RECID, overtime_income, tip_income columns)
    return augmented_taxunit_tmd_imputed_dataframe(aug_tmd_idf)


def create_cex_imputed_tmd(
    tmd_udf: pd.DataFrame,
    cex_udf: pd.DataFrame,
    verbose: bool,
) -> pd.DataFrame:
    """
    Use MICE algorithm to impute auto_loan_interest values
    from CEX unit data to TMD unit data, returning a TMD
    tax-unit dataframe containing imputed variable values.
    """
    mdf = pd.concat([cex_udf, tmd_udf], axis=0).copy()
    colnames = mdf.columns.tolist()
    idx_order = [colnames.index("auto_loan_interest")]
    ignore_vars = [colnames.index("RECID"), colnames.index("weight")]
    mice = MICE(
        mdf.shape[0],
        mdf.shape[1],
        idx_order,
        ignore_vars,
        monotone=True,
        iters=1,  # MICE M (number of iterations) is 1 because monotone=True
        verbose=verbose,  # if True, write impute progress to stdout
        seed=192837465,
        # post-MICE imputation adjustment parameters:
        # ... convert zero to nonzero with prob
        convert_zero_prob=[ALI_convert_zero_prob[TAXYEAR]],
        # ... multiplicative scaling done after convert
        scale=[ALI_scale[TAXYEAR]],
    )
    iarray = mice.impute(mdf.to_numpy())
    idf = pd.DataFrame(iarray, columns=mdf.columns)
    tdf = idf[idf["RECID"] > 0]  # removes CEX data leaving imputed TMD data
    assert (
        not tdf["auto_loan_interest"].isna().any()
    ), "Some imputed auto_loan_interest values are NaN"
    return tdf


def create_augmented_file(
    write_file: bool = True,
    verbose: bool = False,
) -> None:
    """
    Create Tax-Calculator-style input variable file for TMD_YEAR augmented
    with imputed values for the overtime_income, tip_income, and
    auto_loan_interest variables, which do exist in the unaugmented file
    but are zero for each tax unit.
    """
    print("Preparing external data for imputing missing variables...")

    # create SIPP dataframe for imputing missing TMD
    # overtime_income and tip_income variables
    sipp_adf = read_sipp_for_imputation()
    # tabulate sipp_adf dataframe
    if verbose:
        print("\nsipp_adf.shape", sipp_adf.shape)
        print(
            "sipp_adf_stats=\n", sipp_adf.drop(columns=["weight"]).describe()
        )
        print()
        age_mean = sipp_adf["age"].mean()
        print(f"age.unwghted_mean= {age_mean:.3f}")
        otm_mean = sipp_adf["otm_amt"].mean()
        print(f"otm.unwghted_mean= {otm_mean:.3f}")
        tip_mean = sipp_adf["tip_amt"].mean()
        print(f"tip.unwghted_mean= {tip_mean:.3f}")
        wght = sipp_adf["weight"].to_numpy()
        print(f"weight.sum(#M)= {(wght.sum() * 1e-6):.3f}")
    sipp_idf = prep_sipp_for_imputation(sipp_adf)
    if verbose:
        print("sipp_idf.shape=", sipp_idf.shape)
        print(
            "sipp_idf_stats=\n",
            sipp_idf.drop(columns=["RECID", "weight"]).describe(),
        )
        print()
        print_sipp_idf_tabulations(sipp_idf)

    # create TMD dataframe for use in SIPP imputation
    tmd_idf = read_tmd_for_sipp_imputation()
    if verbose:
        print("tmd_idf.shape=", tmd_idf.shape)
        print(
            "tmd_idf_stats=\n",
            tmd_idf.drop(columns=["RECID", "weight"]).describe(),
        )

    # use MICE class to impute missing TMD overtime/tip variables
    print("Imputing overtime and tip data from SIPP to TMD...")
    assert sipp_idf.columns.tolist() == tmd_idf.columns.tolist()
    tmd_udf = create_sipp_imputed_tmd(tmd_idf.copy(), sipp_idf.copy(), verbose)

    # apply tmd_udf imputed values to whole TMD dataframe, all_udf
    all_udf = pd.read_csv(TMD_PATH)
    all_udf.sort_values(by="RECID", inplace=True)
    tmd_udf.sort_values(by="RECID", inplace=True)
    all_udf["overtime_income"] = (
        tmd_udf["overtime_income"].round(0).astype("int32")
    )
    all_udf["tip_income"] = tmd_udf["tip_income"].round(0).astype("int32")
    if verbose:
        wght = all_udf["s006"].to_numpy()
        inc = all_udf["overtime_income"].to_numpy()
        print(f"I:all_udf:wght_otm($b)= {((wght * inc).sum() * 1e-9):.3f}")
        inc = all_udf["tip_income"].to_numpy()
        print(f"I:all_udf:wght_tip($b)= {((wght * inc).sum() * 1e-9):.3f}")

    # create CEX dataframe for imputing missing TMD auto_loan_interest variable
    cex_udf = read_cex_for_imputation()
    if verbose:
        print("cex_udf.head=\n", cex_udf.head())
        print("cex_udf.shape=", cex_udf.shape)
        wt_units = cex_udf["weight"].sum() * 1e-6
        print(f"cex_wt_units(#m)= {wt_units:.3f}")
        print("cex_udf.describe()=\n", cex_udf.describe())
        ali = cex_udf["auto_loan_interest"].to_numpy()
        ali_bins = [0, 1e-2, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, np.inf]
        cnt, edges = np.histogram(ali, bins=ali_bins)
        print("ALI:cnt", cnt)
        print("ALI:bin", edges)
        inc = cex_udf["income"].to_numpy()
        inc_bins = [-np.inf, 0, 1e-2, 25e3, 50e3, 100e3, 200e3, 500e3, np.inf]
        cnt, edges = np.histogram(inc, bins=inc_bins)
        print("INC:cnt", cnt)
        print("INC:bin", edges)
        print("cnt[INC>0 & INC<=ALI]=", ((inc > 0) & (inc <= ali)).sum())
        wght = cex_udf["weight"].to_numpy()
        wt_total = wght.sum() * 1e-6
        hi_inc = inc > 500e3
        wt_hi_inc = (wght * hi_inc).sum() * 1e-6
        print("cnt[INC>500e3]=", hi_inc.sum())
        print(f"wt[INC>500e3](#m)= {wt_hi_inc: .3f}")
        print(f"wt[INC>500e3](%)= {(100 * wt_hi_inc / wt_total): .2f}")
        print(f"wt_cex_ALI($B)= {((wght * ali).sum() * 1e-9): .3f}")

    # create TMD dataframe for use in CEX imputation
    tmd_udf = read_tmd_for_cex_imputation(cex_udf.columns.tolist())
    if verbose:
        print("tmd_udf.shape=", tmd_udf.shape)
        print(
            "tmd_udf_stats=\n",
            tmd_udf.drop(columns=["RECID", "weight"]).describe(),
        )
        inc = tmd_udf["income"].to_numpy()
        inc_bins = [-np.inf, 0, 1e-2, 25e3, 50e3, 100e3, 200e3, 500e3, np.inf]
        cnt, edges = np.histogram(inc, bins=inc_bins)
        print("INC:cnt", cnt)
        print("INC:bin", edges)
        wght = tmd_udf["weight"].to_numpy()
        wt_total = wght.sum() * 1e-6
        hi_inc = inc > 500e3
        wt_hi_inc = (wght * hi_inc).sum() * 1e-6
        print("cnt[INC>500e3]=", hi_inc.sum())
        print(f"wt[INC>500e3](#m)= {wt_hi_inc: .3f}")
        print(f"wt[INC>500e3](%)= {(100 * wt_hi_inc / wt_total): .2f}")

    # use MICE class to impute missing TMD auto_loan_interest variable
    print("Imputing auto loan interest data from CEX to TMD...")
    assert cex_udf.columns.tolist() == tmd_udf.columns.tolist()
    tmd_udf = create_cex_imputed_tmd(tmd_udf.copy(), cex_udf.copy(), verbose)
    if verbose:
        print("tmd_udf:I.shape=", tmd_udf.shape)
        ali = tmd_udf["auto_loan_interest"].to_numpy()
        ali_bins = [0, 1e-2, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, np.inf]
        cnt, edges = np.histogram(ali, bins=ali_bins)
        print("ALI:I:cnt", cnt)
        print("ALI:I:bin", edges)
        wght = tmd_udf["weight"].to_numpy()
        wt_total = wght.sum() * 1e-6
        print(f"ALI:I:wt_ALI($B)= {((wght * ali).sum() * 1e-9): .3f}")

    # apply tmd_udf imputed values to whole TMD dataframe, all_udf
    all_udf.sort_values(by="RECID", inplace=True)
    tmd_udf.sort_values(by="RECID", inplace=True)
    all_udf["auto_loan_interest"] = np.round(
        tmd_udf["auto_loan_interest"].to_numpy(), 0
    ).astype("int32")

    # write TMD tax-unit data file including imputed variable values
    # leaving pre-impute TMD data file as preimpute_tmd.csv.gz
    if write_file:
        preimpute_path = STORAGE_FOLDER / "output" / "preimpute_tmd.csv.gz"
        print(f"Writing preimpute TMD file... [{preimpute_path}]")
        shutil.move(TMD_PATH, preimpute_path)
        print(f"Writing augmented TMD file... [{TMD_PATH}]")
        all_udf.to_csv(TMD_PATH, index=False, float_format="%.5f")
        shutil.copystat(preimpute_path, TMD_PATH)
        preimpute_path.touch(exist_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(create_augmented_file(write_file=True, verbose=False))
