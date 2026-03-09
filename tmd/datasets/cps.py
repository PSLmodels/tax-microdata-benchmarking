import os
from io import BytesIO
from typing import Type
from zipfile import ZipFile
import yaml
import requests
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm
import h5py
from policyengine_core.data import Dataset
from tmd.storage import STORAGE_FOLDER

AGED_RNG = np.random.default_rng(seed=374651932)


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


class RawCPS(Dataset):
    name = "raw_cps"
    label = "Raw CPS"
    time_period = None
    data_format = Dataset.TABLES

    def generate(self) -> pd.DataFrame:
        # Generates the raw CPS dataset.
        # Files are named for a year after the year the survey represents.
        # For example, the 2020 CPS was administered in March 2021, so it's
        # named 2021.
        file_year = int(self.time_period) + 1
        file_year_code = str(file_year)[-2:]

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

        if self.time_period not in CPS_URL_BY_YEAR:
            raise ValueError(
                f"No raw CPS data URL known for year {self.time_period}."
            )

        url = CPS_URL_BY_YEAR[self.time_period]

        spm_unit_columns = SPM_UNIT_COLUMNS
        if self.time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]

        response = requests.get(
            url,
            stream=True,
            verify=False,
            timeout=(20, 600),
        )
        total_size_in_bytes = int(
            response.headers.get("content-length", 200e6)
        )
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
        try:
            with BytesIO() as file, pd.HDFStore(
                self.file_path, mode="w"
            ) as storage:
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
                    # In the 2018 CPS, the file is within prod/data/2019
                    # instead of at the top level.
                    file_prefix = "cpspb/asec/prod/data/2019/"
                else:
                    file_prefix = ""
                with zipfile.open(
                    f"{file_prefix}pppub{file_year_code}.csv"
                ) as f:
                    storage["person"] = pd.read_csv(
                        f,
                        usecols=PERSON_COLUMNS
                        + spm_unit_columns
                        + TAX_UNIT_COLUMNS,
                    ).fillna(0)
                    person = storage["person"]
                with zipfile.open(
                    f"{file_prefix}ffpub{file_year_code}.csv"
                ) as f:
                    person_family_id = person.PH_SEQ * 10 + person.PF_SEQ
                    family = pd.read_csv(f).fillna(0)
                    family_id = family.FH_SEQ * 10 + family.FFPOS
                    family = family[family_id.isin(person_family_id)]
                    storage["family"] = family
                with zipfile.open(
                    f"{file_prefix}hhpub{file_year_code}.csv"
                ) as f:
                    person_household_id = person.PH_SEQ
                    household = pd.read_csv(f).fillna(0)
                    household_id = household.H_SEQ
                    household = household[
                        household_id.isin(person_household_id)
                    ]
                    storage["household"] = household
                storage["tax_unit"] = RawCPS._create_tax_unit_table(person)
                storage["spm_unit"] = RawCPS._create_spm_unit_table(
                    person, self.time_period
                )
        except Exception as e:
            raise ValueError(
                "Attempted to extract and save the CSV files, "
                f"but encountered an error: {e} "
                "(removed the intermediate dataset)."
            ) from e

    @staticmethod
    def _create_tax_unit_table(person: pd.DataFrame) -> pd.DataFrame:
        tax_unit_df = person[TAX_UNIT_COLUMNS].groupby(person.TAX_ID).sum()
        tax_unit_df["TAX_ID"] = tax_unit_df.index
        return tax_unit_df

    @staticmethod
    def _create_spm_unit_table(
        person: pd.DataFrame, time_period: int
    ) -> pd.DataFrame:
        spm_unit_columns = SPM_UNIT_COLUMNS
        if time_period <= 2020:
            spm_unit_columns = [
                col for col in spm_unit_columns if col != "SPM_BBSUBVAL"
            ]
        return person[spm_unit_columns].groupby(person.SPM_ID).first()


class RawCPS_2021(RawCPS):
    time_period = 2021
    name = "raw_cps_2021"
    label = "Raw CPS 2021"
    file_path = STORAGE_FOLDER / "input" / "raw_cps_2021.h5"


class CPS(Dataset):
    name = "cps"
    label = "CPS"
    raw_cps: Type[RawCPS] = None
    previous_year_raw_cps: Type[RawCPS] = None
    data_format = Dataset.ARRAYS

    def generate(self):
        # Generates a Current Population Survey dataset for PE-US microsims
        # Technical documentation and codebook at this URL:
        #  https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar21.pdf
        raw_data = self.raw_cps(  # pylint: disable=not-callable
            require=True
        ).load()
        cps = h5py.File(self.file_path, mode="w")

        ENTITIES = ("person", "tax_unit", "family", "spm_unit", "household")
        person, tax_unit, family, spm_unit, household = [
            raw_data[entity] for entity in ENTITIES
        ]

        add_id_variables(cps, person, tax_unit, family, spm_unit, household)
        add_personal_variables(cps, person)
        add_personal_income_variables(cps, person)
        add_previous_year_income(self, cps)
        add_spm_variables(cps, spm_unit)
        add_household_variables(cps, household)

        raw_data.close()
        cps.close()

        cps = h5py.File(self.file_path, mode="a")
        cps.close()


def add_id_variables(
    cps: h5py.File,
    person: DataFrame,
    tax_unit: DataFrame,
    family: DataFrame,
    spm_unit: DataFrame,
    household: DataFrame,
) -> None:
    """Add basic ID and weight variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The person table of the ASEC.
        tax_unit (DataFrame): The tax unit table created from the person table
            of the ASEC.
        family (DataFrame): The family table of the ASEC.
        spm_unit (DataFrame): The SPM unit table created from the person table
            of the ASEC.
        household (DataFrame): The household table of the ASEC.
    """
    # Add primary and foreign keys
    cps["person_id"] = person.PH_SEQ * 100 + person.P_SEQ
    cps["family_id"] = family.FH_SEQ * 10 + family.FFPOS
    cps["household_id"] = household.H_SEQ
    cps["person_tax_unit_id"] = person.TAX_ID
    cps["person_spm_unit_id"] = person.SPM_ID
    cps["tax_unit_id"] = tax_unit.TAX_ID
    cps["spm_unit_id"] = spm_unit.SPM_ID
    cps["person_household_id"] = person.PH_SEQ
    cps["person_family_id"] = person.PH_SEQ * 10 + person.PF_SEQ

    # Add weights
    # Weights are multiplied by 100 to avoid decimals
    cps["person_weight"] = person.A_FNLWGT / 1e2
    cps["family_weight"] = family.FSUP_WGT / 1e2

    # Tax unit weight is the weight of the containing family.
    family_weight = Series(
        cps["family_weight"][...], index=cps["family_id"][...]
    )
    person_family_id = cps["person_family_id"][...]
    persons_family_weight = Series(family_weight[person_family_id])
    cps["tax_unit_weight"] = persons_family_weight.groupby(
        cps["person_tax_unit_id"][...]
    ).first()

    cps["spm_unit_weight"] = spm_unit.SPM_WEIGHT / 1e2

    cps["household_weight"] = household.HSUP_WGT / 1e2

    # Marital units

    marital_unit_id = person.PH_SEQ * 1e6 + np.maximum(
        person.A_LINENO, person.A_SPOUSE
    )

    # marital_unit_id is not the household ID, zero padded and followed
    # by the index within household (of each person, or their spouse if
    # one exists earlier in the survey).

    marital_unit_id = Series(marital_unit_id).rank(
        method="dense"
        # simplifies to a natural number sequence
        # with repetitions [0, 1, 1, 2, 3, ...]
    )

    cps["person_marital_unit_id"] = marital_unit_id.values
    cps["marital_unit_id"] = marital_unit_id.drop_duplicates().values


def add_personal_variables(cps: h5py.File, person: DataFrame) -> None:
    """Add personal demographic variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The CPS person table.
    """

    # The CPS provides age as follows:
    # 00-79 = 0-79 years of age
    # 80 = 80-84 years of age
    # 85 = 85+ years of age
    # We assign the 80 ages randomly between 80 and 84 to avoid bunching at 80
    cps["age"] = np.where(
        person.A_AGE == 80,
        AGED_RNG.integers(low=80, high=85, endpoint=False, size=len(person)),
        person.A_AGE,
    )
    # A_SEX is 1 -> male, 2 -> female.
    cps["is_female"] = person.A_SEX == 2
    # "Is...blind or does...have serious difficulty seeing even when Wearing
    #  glasses?" 1 -> Yes
    cps["is_blind"] = person.PEDISEYE == 1
    DISABILITY_FLAGS = [
        "PEDIS" + i for i in ["DRS", "EAR", "EYE", "OUT", "PHY", "REM"]
    ]
    cps["is_disabled"] = (person[DISABILITY_FLAGS] == 1).any(axis=1)

    def children_per_parent(col: str) -> pd.DataFrame:
        """Calculate number of children in the household using parental
            pointers.

        Args:
            col (str): Either PEPAR1 and PEPAR2, which correspond to A_LINENO
            of the person's first and second parent in the household,
            respectively.
        """
        return (
            person[person[col] > 0]
            .groupby(["PH_SEQ", col])
            .size()
            .reset_index()
            .rename(columns={col: "A_LINENO", 0: "children"})
        )

    # Aggregate to parent.
    res = (
        pd.concat(
            [children_per_parent("PEPAR1"), children_per_parent("PEPAR2")]
        )
        .groupby(["PH_SEQ", "A_LINENO"])
        .children.sum()
        .reset_index()
    )
    tmp = person[["PH_SEQ", "A_LINENO"]].merge(
        res, on=["PH_SEQ", "A_LINENO"], how="left"
    )
    cps["own_children_in_household"] = tmp.children.fillna(0)

    cps["has_marketplace_health_coverage"] = person.MRK == 1

    cps["cps_race"] = person.PRDTRACE
    cps["is_hispanic"] = person.PRDTHSP != 0

    cps["is_widowed"] = person.A_MARITL == 4
    cps["is_separated"] = person.A_MARITL == 6
    # High school or college/university enrollment status.
    cps["is_full_time_college_student"] = person.A_HSCOL == 2


def add_personal_income_variables(cps: h5py.File, person: DataFrame):
    """
    Add income variables.

    Args:
        cps (h5py.File): The CPS dataset file.
        person (DataFrame): The CPS person table.
    """
    # Get income imputation parameters.
    yamlfilename = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        STORAGE_FOLDER / "input" / "imputation_parameters.yaml",
    )
    with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
        p = yaml.safe_load(yamlfile)
    assert isinstance(p, dict)

    # Assign CPS variables.
    cps["employment_income"] = person.WSAL_VAL

    cps["weekly_hours_worked"] = person.HRSWK * person.WKSWORK / 52

    cps["taxable_interest_income"] = person.INT_VAL * (
        p["taxable_interest_fraction"]
    )
    cps["tax_exempt_interest_income"] = person.INT_VAL * (
        1 - p["taxable_interest_fraction"]
    )
    cps["self_employment_income"] = person.SEMP_VAL
    cps["farm_income"] = person.FRSE_VAL
    cps["qualified_dividend_income"] = person.DIV_VAL * (
        p["qualified_dividend_fraction"]
    )
    cps["non_qualified_dividend_income"] = person.DIV_VAL * (
        1 - p["qualified_dividend_fraction"]
    )
    cps["rental_income"] = person.RNT_VAL
    # Assign Social Security retirement benefits if at least 62.
    MINIMUM_RETIREMENT_AGE = 62
    cps["social_security_retirement"] = np.where(
        person.A_AGE >= MINIMUM_RETIREMENT_AGE, person.SS_VAL, 0
    )
    # Otherwise assign them to Social Security disability benefits.
    cps["social_security_disability"] = (
        person.SS_VAL - cps["social_security_retirement"]
    )
    # Provide placeholders for other Social Security inputs to avoid creating
    # NaNs as they're uprated.
    cps["social_security_dependents"] = np.zeros_like(
        cps["social_security_retirement"]
    )
    cps["social_security_survivors"] = np.zeros_like(
        cps["social_security_retirement"]
    )
    cps["unemployment_compensation"] = person.UC_VAL
    # Add pensions and annuities.
    cps_pensions = person.PNSN_VAL + person.ANN_VAL
    # Assume a constant fraction of pension income is taxable.
    cps["taxable_private_pension_income"] = (
        cps_pensions * p["taxable_pension_fraction"]
    )
    cps["tax_exempt_private_pension_income"] = cps_pensions * (
        1 - p["taxable_pension_fraction"]
    )
    # Retirement account distributions.
    RETIREMENT_CODES = {
        1: "401k",
        2: "403b",
        3: "roth_ira",
        4: "regular_ira",
        5: "keogh",
        6: "sep",  # Simplified Employee Pension plan
        7: "other_type_retirement_account",
    }
    for code, description in RETIREMENT_CODES.items():
        tmp = 0
        # The ASEC splits distributions across four variable pairs.
        for i in ["1", "2", "1_YNG", "2_YNG"]:
            tmp += (person["DST_SC" + i] == code) * person["DST_VAL" + i]
        cps[f"{description}_distributions"] = tmp
    # Allocate retirement distributions by taxability.
    for source_with_taxable_fraction in ["401k", "403b", "sep"]:
        cps[f"taxable_{source_with_taxable_fraction}_distributions"] = (
            cps[f"{source_with_taxable_fraction}_distributions"][...]
            * p[
                f"taxable_{source_with_taxable_fraction}_distribution_fraction"
            ]
        )
        cps[f"tax_exempt_{source_with_taxable_fraction}_distributions"] = cps[
            f"{source_with_taxable_fraction}_distributions"
        ][...] * (
            1
            - p[
                f"taxable_{source_with_taxable_fraction}_distribution_fraction"
            ]
        )
        del cps[f"{source_with_taxable_fraction}_distributions"]

    # Assume all regular IRA distributions are taxable,
    # and all Roth IRA distributions are not.
    cps["taxable_ira_distributions"] = cps["regular_ira_distributions"]
    cps["tax_exempt_ira_distributions"] = cps["roth_ira_distributions"]
    # Other income (OI_VAL) is a catch-all for all other income sources.
    # The code for alimony income is 20.
    cps["alimony_income"] = (person.OI_OFF == 20) * person.OI_VAL
    # The code for strike benefits is 12.
    cps["strike_benefits"] = (person.OI_OFF == 12) * person.OI_VAL
    cps["child_support_received"] = person.CSP_VAL
    # Assume all public assistance / welfare dollars (PAW_VAL) are TANF.
    # They could also include General Assistance.
    cps["tanf_reported"] = person.PAW_VAL
    cps["ssi_reported"] = person.SSI_VAL
    # Assume all retirement contributions are traditional 401(k) for now.
    # Procedure for allocating retirement contributions:
    # 1) If they report any self-employment income, allocate entirely to
    #    self-employed pension contributions.
    # 2) If they report any wage and salary income, allocate in this order:
    #    a) Traditional 401(k) contributions up to to limit
    #    b) Roth 401(k) contributions up to the limit
    #    c) IRA contributions up to the limit, split according
    #       to administrative fractions
    #    d) Other retirement contributions
    # Disregard reported pension contributions from people
    #    who report neither wage and salary nor self-employment income.
    # Assume no 403(b) or 457 contributions for now.
    LIMIT_401K_2022 = 20_500
    LIMIT_401K_CATCH_UP_2022 = 6_500
    LIMIT_IRA_2022 = 6_000
    LIMIT_IRA_CATCH_UP_2022 = 1_000
    CATCH_UP_AGE_2022 = 50
    retirement_contributions = person.RETCB_VAL
    cps["self_employed_pension_contributions"] = np.where(
        person.SEMP_VAL > 0, retirement_contributions, 0
    )
    remaining_retirement_contributions = np.maximum(
        retirement_contributions - cps["self_employed_pension_contributions"],
        0,
    )
    # Compute the 401(k) limit for the person's age.
    catch_up_eligible = person.A_AGE >= CATCH_UP_AGE_2022
    limit_401k = LIMIT_401K_2022 + catch_up_eligible * LIMIT_401K_CATCH_UP_2022
    limit_ira = LIMIT_IRA_2022 + catch_up_eligible * LIMIT_IRA_CATCH_UP_2022
    cps["traditional_401k_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_401k),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions
        - cps["traditional_401k_contributions"],
        0,
    )
    cps["roth_401k_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_401k),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions - cps["roth_401k_contributions"],
        0,
    )
    cps["traditional_ira_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, limit_ira),
        0,
    )
    remaining_retirement_contributions = np.maximum(
        remaining_retirement_contributions
        - cps["traditional_ira_contributions"],
        0,
    )
    roth_ira_limit = limit_ira - cps["traditional_ira_contributions"]
    cps["roth_ira_contributions"] = np.where(
        person.WSAL_VAL > 0,
        np.minimum(remaining_retirement_contributions, roth_ira_limit),
        0,
    )
    # Allocate capital gains into long-term and short-term
    # based on aggregate split.
    cps["long_term_capital_gains"] = person.CAP_VAL * (
        p["long_term_capgain_fraction"]
    )
    cps["short_term_capital_gains"] = person.CAP_VAL * (
        1 - p["long_term_capgain_fraction"]
    )
    cps["receives_wic"] = person.WICYN == 1
    cps["veterans_benefits"] = person.VET_VAL
    cps["workers_compensation"] = person.WC_VAL
    # Disability income has multiple sources and values split across two pairs
    # of variables. Include everything except for worker's compensation
    # (code 1), which is defined as WC_VAL.
    WORKERS_COMP_DISABILITY_CODE = 1
    disability_benefits_1 = person.DIS_VAL1 * (
        person.DIS_SC1 != WORKERS_COMP_DISABILITY_CODE
    )
    disability_benefits_2 = person.DIS_VAL2 * (
        person.DIS_SC2 != WORKERS_COMP_DISABILITY_CODE
    )
    cps["disability_benefits"] = disability_benefits_1 + disability_benefits_2
    # Expenses.
    # "What is the annual amount of child support paid?"
    cps["child_support_expense"] = person.CHSP_VAL
    cps["health_insurance_premiums"] = person.PHIP_VAL
    cps["medical_out_of_pocket_expenses"] = person.MOOP


def add_spm_variables(cps: h5py.File, spm_unit: DataFrame) -> None:
    SPM_RENAMES = {
        "spm_unit_total_income_reported": "SPM_TOTVAL",
        "snap_reported": "SPM_SNAPSUB",
        "spm_unit_capped_housing_subsidy_reported": "SPM_CAPHOUSESUB",
        "free_school_meals_reported": "SPM_SCHLUNCH",
        "spm_unit_energy_subsidy_reported": "SPM_ENGVAL",
        "spm_unit_wic_reported": "SPM_WICVAL",
        "spm_unit_broadband_subsidy_reported": "SPM_BBSUBVAL",
        "spm_unit_payroll_tax_reported": "SPM_FICA",
        "spm_unit_federal_tax_reported": "SPM_FEDTAX",
        # state tax includes refundable credits
        "spm_unit_state_tax_reported": "SPM_STTAX",
        "spm_unit_capped_work_childcare_expenses": "SPM_CAPWKCCXPNS",
        "spm_unit_medical_expenses": "SPM_MEDXPNS",
        "spm_unit_spm_threshold": "SPM_POVTHRESHOLD",
        "spm_unit_net_income_reported": "SPM_RESOURCES",
        "spm_unit_pre_subsidy_childcare_expenses": "SPM_CHILDCAREXPNS",
    }
    for openfisca_variable, asec_variable in SPM_RENAMES.items():
        if asec_variable in spm_unit.columns:
            cps[openfisca_variable] = spm_unit[asec_variable]
    cps["reduced_price_school_meals_reported"] = (
        cps["free_school_meals_reported"][...] * 0
    )


def add_household_variables(cps: h5py.File, household: DataFrame) -> None:
    cps["state_fips"] = household.GESTFIPS
    cps["county_fips"] = household.GTCO
    state_county_fips = cps["state_fips"][...] * 1e3 + cps["county_fips"][...]
    # Assign is_nyc here instead of as a variable formula so that it shows up
    # as toggleable in the webapp.
    # List county FIPS codes for each NYC county/borough.
    NYC_COUNTY_FIPS = [
        5,  # Bronx
        47,  # Kings (Brooklyn)
        61,  # New York (Manhattan)
        81,  # Queens
        85,  # Richmond (Staten Island)
    ]
    # Compute NYC by concatenating NY state FIPS with county FIPS.
    # For example, 36061 is Manhattan.
    NYS_FIPS = 36
    nyc_full_county_fips = [
        NYS_FIPS * 1e3 + county_fips for county_fips in NYC_COUNTY_FIPS
    ]
    cps["in_nyc"] = np.isin(state_county_fips, nyc_full_county_fips)


def add_previous_year_income(self, cps: h5py.File) -> None:
    if self.previous_year_raw_cps is None:
        msg = "Skipping CPS previous year income imputation given lack of data"
        print(f"{msg}...")
        return

    cps_current_year_data = self.raw_cps(require=True).load()
    cps_previous_year_data = self.previous_year_raw_cps(require=True).load()
    cps_previous_year = cps_previous_year_data.person.set_index(
        cps_previous_year_data.person.PERIDNUM
    )
    cps_current_year = cps_current_year_data.person.set_index(
        cps_current_year_data.person.PERIDNUM
    )

    previous_year_data = cps_previous_year[
        ["WSAL_VAL", "SEMP_VAL", "I_ERNVAL", "I_SEVAL"]
    ].rename(
        {
            "WSAL_VAL": "employment_income_last_year",
            "SEMP_VAL": "self_employment_income_last_year",
        },
        axis=1,
    )

    previous_year_data = previous_year_data[
        (previous_year_data.I_ERNVAL == 0) & (previous_year_data.I_SEVAL == 0)
    ]

    previous_year_data.drop(["I_ERNVAL", "I_SEVAL"], axis=1, inplace=True)

    joined_data = cps_current_year.join(previous_year_data)[
        [
            "employment_income_last_year",
            "self_employment_income_last_year",
            "I_ERNVAL",
            "I_SEVAL",
        ]
    ]
    joined_data["previous_year_income_available"] = (
        ~joined_data.employment_income_last_year.isna()
        & ~joined_data.self_employment_income_last_year.isna()
        & (joined_data.I_ERNVAL == 0)
        & (joined_data.I_SEVAL == 0)
    )
    joined_data = joined_data.fillna(-1).drop(["I_ERNVAL", "I_SEVAL"], axis=1)

    # CPS already ordered by PERIDNUM, so the join wouldn't change the order.
    cps["employment_income_last_year"] = joined_data[
        "employment_income_last_year"
    ].values
    cps["self_employment_income_last_year"] = joined_data[
        "self_employment_income_last_year"
    ].values
    cps["previous_year_income_available"] = joined_data[
        "previous_year_income_available"
    ].values


class CPS_2021(CPS):
    name = "cps_2021"
    label = "CPS 2021"
    raw_cps = RawCPS_2021
    file_path = STORAGE_FOLDER / "output" / "cps_2021.h5"
    time_period = 2021


def create_cps_2021():
    CPS_2021().generate()


TC_CPS_AGED_RNG = np.random.default_rng(seed=374651932)

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
    print(f"Downloading raw CPS ASEC for {taxyear}...")
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


def _load_raw_person(taxyear: int) -> pd.DataFrame:
    """Load the person table from the cached raw CPS HDF5 file."""
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
        TC_CPS_AGED_RNG.integers(
            low=80, high=85, endpoint=False, size=len(person)
        ),
        person.A_AGE.values,
    )


def create_tc_cps(taxyear: int) -> pd.DataFrame:
    """
    Create a Tax-Calculator-compatible CPS DataFrame for the given taxyear
    without using PolicyEngine Dataset or hierarchical data files.
    """
    person = _load_raw_person(taxyear)
    print(f"Creating tc CPS dataset for year {taxyear}...")

    # load imputation parameters
    yamlfilename = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        STORAGE_FOLDER / "input" / "imputation_parameters.yaml",
    )
    with open(yamlfilename, "r", encoding="utf-8") as yamlfile:
        p = yaml.safe_load(yamlfile)
    assert isinstance(p, dict)

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
    var["E00300"] = sum_nondep(
        interest_income * p["taxable_interest_fraction"]
    )
    var["E00400"] = sum_nondep(
        interest_income * (1 - p["taxable_interest_fraction"])
    )

    # ... dividend income
    var["E00650"] = sum_nondep(
        dividend_income * p["qualified_dividend_fraction"]
    )
    non_qual_div = sum_nondep(
        dividend_income * (1 - p["qualified_dividend_fraction"])
    )
    var["E00600"] = non_qual_div + var["E00650"]

    # ... rental income (included in e02000)
    var_rental = sum_nondep(rental_income)

    # ... social security
    var["E02400"] = sum_nondep(ss_val)

    # ... unemployment compensation
    var["E02300"] = sum_nondep(uc_val)

    # ... pensions and annuities
    taxable_pension = sum_nondep(pension_val * p["taxable_pension_fraction"])
    tax_exempt_pension = sum_nondep(
        pension_val * (1 - p["taxable_pension_fraction"])
    )
    var["E01700"] = taxable_pension
    var["E01500"] = taxable_pension + tax_exempt_pension

    # ... capital gains
    var["P23250"] = sum_nondep(cap_val * p["long_term_capgain_fraction"])
    var["P22250"] = sum_nondep(cap_val * (1 - p["long_term_capgain_fraction"]))

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

    # ... IRA contributions (traditional_ira_contributions from RETCB_VAL)
    #     (use same allocation logic as in add_personal_income_variables)
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
    zero_tc_names = [
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
    for tcname in zero_tc_names:
        if tcname not in var:
            var[tcname] = zeros

    # ... create DataFrame
    tcdf = pd.DataFrame(var)

    # correct variable name casing for Tax-Calculator
    with open(
        STORAGE_FOLDER / "input" / "tc_variable_metadata.yaml",
        "r",
        encoding="utf-8",
    ) as yfile:
        tc_variable_metadata = yaml.safe_load(yfile)
    renames = {}
    for variable in tcdf.columns:
        if variable.upper() in tc_variable_metadata["read"]:
            renames[variable] = variable.upper()
        elif variable.lower() in tc_variable_metadata["read"]:
            renames[variable] = variable.lower()
    tcdf.rename(columns=renames, inplace=True)

    # drop tax units with zero weight
    nonzero_wt = tcdf["s006"] > 0
    tcdf = tcdf[nonzero_wt].reset_index(drop=True)

    # identify nonfilers using 2022 IRS filing thresholds
    # (use 2022 rules because 2021 had large COVID-related anomalies)
    gross_income = (
        tcdf["e00200"].abs()
        + tcdf["e00300"].abs()
        + tcdf["e00600"].abs()
        + tcdf["e00800"].abs()
        + tcdf["e00900"].abs()
        + tcdf["e01400"].abs()
        + tcdf["e01500"].abs()
        + tcdf["e02100"].abs()
        + tcdf["e02300"].abs()
        + tcdf["e02400"].abs()
        + tcdf["p22250"].abs()
        + tcdf["p23250"].abs()
    )
    head_aged = tcdf["age_head"] >= 65
    spouse_aged = tcdf["age_spouse"] >= 65
    mars = tcdf["MARS"]
    # 2022 IRS filing thresholds by MARS and age
    threshold = pd.Series(np.zeros(len(tcdf)), dtype=float)
    threshold[mars == 1] = np.where(head_aged[mars == 1], 14700, 12950)
    threshold[mars == 2] = np.where(
        head_aged[mars == 2] & spouse_aged[mars == 2],
        28700,
        np.where(head_aged[mars == 2] | spouse_aged[mars == 2], 27300, 25900),
    )
    threshold[mars == 3] = 5
    threshold[mars == 4] = np.where(head_aged[mars == 4], 21150, 19400)
    threshold[mars == 5] = np.where(head_aged[mars == 5], 27300, 25900)
    nonfiler = gross_income < threshold

    # return Tax-Calculator DataFrame and nonfiler boolean Series
    return tcdf, nonfiler


if __name__ == "__main__":
    create_cps_2021()
