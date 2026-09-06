"""
Stage 1 of the growth factors pipeline.

Reads the Census population, CBO baseline, IRS return projection, SOI
estimates, and benefit program source files in tmd/growfactors/data and
writes data/Stage_I_factors.csv, a table of level indexes normalized so
that SYR equals one.  Stage 2 (build_growfactors.py) converts those
indexes into the year-over-year growth factors used by Tax-Calculator.

This logic was migrated from the TaxData repository (puf_stage1/stage1.py).
The code that wrote Stage_II_targets.csv, which was used only by the
TaxData stage2 weighting logic, has been removed; TMD reweights against
tmd/storage/input/soi.csv instead.  The aggregates DataFrame built here
was called Stage_II_targets in TaxData: it is still needed because the
stage 1 factors are computed from it.
"""

import pandas as pd

from tmd.growfactors import DATA_FOLDER

# pylint: disable=invalid-name

SYR = 2011  # calendar year used to normalize factors
BEN_SYR = 2014  # calendar year used just for the benefit start year
EYR = 2036  # last calendar year we have data for
SOI_YR = 2017  # most recently available SOI estimates
IRS_RET_YR = 2022  # most recently available IRS return projections

# define constants for the number refers total population,
# dependent age upper limit, and senior age lower limit
TOTES = 999
DEP = 19
SENIOR = 65

OUTFILE = DATA_FOLDER / "Stage_I_factors.csv"


def read_population():
    """
    Return (TOTAL_POP, POP_DEP, POP_SNR) for years 2008 through EYR by
    combining Census intercensal estimates, postcensal estimates, and the
    2014 national projection.

    Source documentation:
    - Projection from 2014
      <http://www.census.gov/population/projections/data/national/2014/downloadablefiles.html>
    - Historical estimates from 2010 to 2014
      <http://www.census.gov/popest/data/datasets.html>
    - Historical estimates from 2000 to 2010
      <http://www.census.gov/popest/data/intercensal/national/nat2010.html>
    """
    # projection for 2014+
    pop_projection = pd.read_csv(
        DATA_FOLDER / "NP2014_D1.csv", index_col="year"
    )
    pop_projection = pop_projection[
        (pop_projection.sex == 0)
        & (pop_projection.race == 0)
        & (pop_projection.origin == 0)
    ]
    pop_projection = pop_projection.drop(["sex", "race", "origin"], axis=1)
    # drop the rows for years after EYR from the pop_projection DataFrame
    num_drop = EYR + 1 - 2014
    pop_projection = pop_projection.drop(
        pop_projection.index[num_drop:], axis=0
    )
    pop_projection = pop_projection.drop(pop_projection.index[:1], axis=0)

    # data for 2010-2014
    historical1 = pd.read_csv(DATA_FOLDER / "NC-EST2014-AGESEX-RES.csv")
    historical1 = historical1[historical1.SEX == 0]
    historical1 = historical1.drop(
        ["SEX", "CENSUS2010POP", "ESTIMATESBASE2010"], axis=1
    )

    pop_dep1 = historical1[historical1.AGE <= DEP].sum()
    pop_dep1 = pop_dep1.drop(["AGE"], axis=0)

    pop_snr1 = historical1[
        (historical1.AGE >= SENIOR) & (historical1.AGE < TOTES)
    ].sum()
    pop_snr1 = pop_snr1.drop(["AGE"], axis=0)

    total_pop1 = historical1[historical1.AGE == TOTES]
    total_pop1 = total_pop1.drop(["AGE"], axis=1)

    # data for 2008-2009
    historical2 = pd.read_csv(DATA_FOLDER / "US-EST00INT-ALLDATA.csv")
    historical2 = historical2[
        (historical2.MONTH == 7)
        & (historical2.YEAR >= 2008)
        & (historical2.YEAR < 2010)
    ]
    historical2 = historical2.drop(historical2.columns[4:], axis=1)
    historical2 = historical2.drop(historical2.columns[0], axis=1)

    year08under19 = (historical2.YEAR == 2008) & (historical2.AGE <= DEP)
    year09under19 = (historical2.YEAR == 2009) & (historical2.AGE <= DEP)
    pop_dep2 = []
    pop_dep2.append(historical2.TOT_POP[year08under19].sum())
    pop_dep2.append(historical2.TOT_POP[year09under19].sum())

    year08over65 = (
        (historical2.YEAR == 2008)
        & (historical2.AGE >= SENIOR)
        & (historical2.AGE < TOTES)
    )
    year09over65 = (
        (historical2.YEAR == 2009)
        & (historical2.AGE >= SENIOR)
        & (historical2.AGE < TOTES)
    )
    pop_snr2 = []
    pop_snr2.append(historical2.TOT_POP[year08over65].sum())
    pop_snr2.append(historical2.TOT_POP[year09over65].sum())

    year08total = (historical2.YEAR == 2008) & (historical2.AGE == TOTES)
    year09total = (historical2.YEAR == 2009) & (historical2.AGE == TOTES)
    total_pop2 = []
    total_pop2.append(historical2.TOT_POP[year08total].sum())
    total_pop2.append(historical2.TOT_POP[year09total].sum())

    # combine data for 2008-2014 with projection data
    popdf = pd.DataFrame(
        pop_projection[pop_projection.columns[1:21]].sum(axis=1)
    )
    POP_DEP = pd.concat(
        [pd.DataFrame(pop_dep2), pd.DataFrame(pop_dep1), popdf]
    )
    popdf = pd.DataFrame(
        pop_projection[pop_projection.columns[66:]].sum(axis=1)
    )
    POP_SNR = pd.concat(
        [pd.DataFrame(pop_snr2), pd.DataFrame(pop_snr1), popdf]
    )
    TOTAL_POP = pd.concat(
        [
            pd.DataFrame(total_pop2),
            pd.DataFrame(total_pop1.values.transpose()),
            pd.DataFrame(pop_projection.total_pop.values),
        ]
    )
    return TOTAL_POP, POP_DEP, POP_SNR


def benefit_factors(index):
    """
    Return a DataFrame of aggregate benefit level indexes normalized to
    BEN_SYR, extrapolated from the most recent year of benefit program
    cost data through EYR.
    """
    benefit_programs = pd.read_csv(
        DATA_FOLDER / "benefitprograms.csv", index_col="Program"
    )
    benefit_sums = benefit_programs[benefit_programs.columns[2:]].apply(sum)
    # find growth rate between 2020 and 2021 and extrapolate out to EYR
    gr = benefit_sums["2021_cost"] / float(benefit_sums["2020_cost"])
    for year in range(2022, EYR + 1):
        prev_year = year - 1
        prev_value = benefit_sums[f"{prev_year}_cost"]
        benefit_sums[f"{year}_cost"] = prev_value * gr
    ABENEFITS = (benefit_sums / benefit_sums[f"{BEN_SYR}_cost"]).transpose()
    factors = pd.DataFrame()
    for year in index:
        if year <= BEN_SYR:
            factors[year] = [1.0]
        else:
            factors[year] = ABENEFITS[f"{year}_cost"]
    return factors


def create_stage1_factors():
    """
    Create the Stage_I_factors.csv file.
    """
    TOTAL_POP, POP_DEP, POP_SNR = read_population()

    # aggregates holds the levels that the stage 1 factors are derived from
    aggregates = pd.DataFrame(TOTAL_POP)
    aggregates.columns = ["TOTAL_POP"]
    aggregates["POP_DEP"] = POP_DEP.values
    aggregates["POP_SNR"] = POP_SNR.values

    index = list(range(2008, EYR + 1))
    aggregates.index = index

    # calculate Stage_I_factors for population targets
    APOPN = aggregates.TOTAL_POP / aggregates.TOTAL_POP[SYR]
    Stage_I_factors = pd.DataFrame(APOPN, index=index)
    Stage_I_factors.columns = ["APOPN"]
    data = aggregates.POP_DEP / aggregates.POP_DEP[SYR]
    Stage_I_factors["APOPDEP"] = pd.DataFrame(data, index=index)
    data = aggregates.POP_SNR / aggregates.POP_SNR[SYR]
    Stage_I_factors["APOPSNR"] = pd.DataFrame(data, index=index)

    # specify yearly growth rates used to create Stage_I_factors
    pop_growth_rates = pd.DataFrame(aggregates.TOTAL_POP.pct_change() + 1.0)
    pop_growth_rates["POPDEP"] = aggregates.POP_DEP.pct_change() + 1.0
    pop_growth_rates["POPSNR"] = aggregates.POP_SNR.pct_change() + 1.0
    pop_growth_rates = pop_growth_rates.drop(pop_growth_rates.index[0], axis=0)

    # import CBO baseline projection
    cbo_baseline = pd.read_csv(DATA_FOLDER / "CBO_baseline.csv", index_col=0)
    cbobase = cbo_baseline.transpose()
    cbobase.index = index
    cbobase = cbobase.astype(float)
    Stage_I_factors["AGDPN"] = pd.DataFrame(
        cbobase.GDP / cbobase.GDP[SYR], index=index
    )
    Stage_I_factors["ATXPY"] = pd.DataFrame(
        cbobase.TPY / cbobase.TPY[SYR], index=index
    )
    Stage_I_factors["ASCHF"] = pd.DataFrame(
        cbobase.SCHF / cbobase.SCHF[SYR], index=index
    )
    Stage_I_factors["ABOOK"] = pd.DataFrame(
        cbobase.BOOK / cbobase.BOOK[SYR], index=index
    )
    Stage_I_factors["ACPIU"] = pd.DataFrame(
        cbobase.CPIU / cbobase.CPIU[SYR], index=index
    )
    Stage_I_factors["ACPIM"] = pd.DataFrame(
        cbobase.CPIM / cbobase.CPIM[SYR], index=index
    )
    cbo_growth_rates = cbobase.pct_change() + 1.0
    cbo_growth_rates = cbo_growth_rates.drop(cbo_growth_rates.index[0], axis=0)

    # read IRS number-of-returns projection
    irs_returns = pd.read_csv(
        DATA_FOLDER / "IRS_return_projection.csv", index_col=0
    )
    irs_returns = irs_returns.transpose()
    return_growth_rate = irs_returns.pct_change() + 1.0
    vals = []
    indicies = []
    for year in range(IRS_RET_YR + 1, EYR + 1):
        vals.append(return_growth_rate.Returns[str(IRS_RET_YR)])
        indicies.append(str(year))
    ret_growth_vals = pd.DataFrame({"Returns": vals}, index=indicies)
    return_growth_rate = pd.concat([return_growth_rate, ret_growth_vals])
    # NOTE: TaxData set the index via return_growth_rate.Returns.index,
    # which is a no-op under pandas copy-on-write; set it on the frame.
    return_growth_rate.index = index

    # read SOI estimates for 2008+
    soi_estimates = pd.read_csv(DATA_FOLDER / "SOI_estimates.csv", index_col=0)
    soi_estimates = soi_estimates.transpose()
    historical_index = list(range(2008, SOI_YR + 1))
    soi_estimates.index = historical_index

    # use yearly growth rates from Census, CBO, and IRS as blowup factors
    return_projection = soi_estimates
    for i in range(SOI_YR, EYR):  # SOI Estimates loop
        Single = (
            return_projection.Single[i] * return_growth_rate.Returns[i + 1]
        )
        Joint = return_projection.Joint[i] * return_growth_rate.Returns[i + 1]
        HH = return_projection.HH[i] * return_growth_rate.Returns[i + 1]
        SS_return = (
            return_projection.SS_return[i] * pop_growth_rates.POPSNR[i + 1]
        )
        Dep_return = (
            return_projection.Dep_return[i] * pop_growth_rates.POPDEP[i + 1]
        )
        INTS = return_projection.INTS[i] * cbo_growth_rates.INTS[i + 1]
        DIVS = return_projection.DIVS[i] * cbo_growth_rates.DIVS[i + 1]
        SCHCI = return_projection.SCHCI[i] * cbo_growth_rates.SCHC[i + 1]
        SCHCL = return_projection.SCHCL[i] * cbo_growth_rates.SCHC[i + 1]
        CGNS = return_projection.CGNS[i] * cbo_growth_rates.CGNS[i + 1]
        Pension = return_projection.Pension[i] * cbo_growth_rates.TPY[i + 1]
        SCHEI = return_projection.SCHEI[i] * cbo_growth_rates.BOOK[i + 1]
        SCHEL = return_projection.SCHEL[i] * cbo_growth_rates.BOOK[i + 1]
        SS = return_projection.SS[i] * cbo_growth_rates.SOCSEC[i + 1]
        UCOMP = return_projection.UCOMP[i] * cbo_growth_rates.UCOMP[i + 1]
        IPD = return_projection.IPD[i] * cbo_growth_rates.TPY[i + 1]
        wages = [
            return_projection[f"WAGE_{num}"][i] * cbo_growth_rates.Wages[i + 1]
            for num in range(1, 13)
        ]

        current_year = pd.DataFrame(
            [
                Single,
                Joint,
                HH,
                SS_return,
                Dep_return,
                INTS,
                DIVS,
                SCHCI,
                SCHCL,
                CGNS,
                Pension,
                SCHEI,
                SCHEL,
                SS,
                UCOMP,
                IPD,
            ]
            + wages
        )
        current_year = current_year.transpose()

        current_year.columns = return_projection.columns
        current_year.index = [i + 1]
        return_projection = pd.concat([return_projection, current_year])

    # combine historical data with the newly blownup data
    aggregates = pd.concat([aggregates, return_projection], axis=1)

    # create all the rest of the Stage_I_factors
    data = aggregates[aggregates.columns[3:6]].sum(axis=1)
    total_return = pd.DataFrame(data, columns=["ARETS"])

    data = aggregates[aggregates.columns[19:31]].sum(axis=1)
    total_wage = pd.DataFrame(data, columns=["AWAGE"])

    Stage_I_factors["ARETS"] = total_return / total_return.ARETS[SYR]
    Stage_I_factors["AWAGE"] = total_wage / total_wage.AWAGE[SYR]
    Stage_I_factors["ASCHCI"] = aggregates.SCHCI / aggregates.SCHCI[SYR]
    Stage_I_factors["ASCHCL"] = aggregates.SCHCL / aggregates.SCHCL[SYR]
    Stage_I_factors["ASCHEI"] = aggregates.SCHEI / aggregates.SCHEI[SYR]
    Stage_I_factors["ASCHEL"] = aggregates.SCHEL / aggregates.SCHEL[SYR]
    Stage_I_factors["AINTS"] = aggregates.INTS / aggregates.INTS[SYR]
    Stage_I_factors["ADIVS"] = aggregates.DIVS / aggregates.DIVS[SYR]
    Stage_I_factors["ACGNS"] = aggregates.CGNS / aggregates.CGNS[SYR]
    Stage_I_factors["ASOCSEC"] = aggregates.SS / aggregates.SS[SYR]
    Stage_I_factors["AUCOMP"] = aggregates.UCOMP / aggregates.UCOMP[SYR]
    Stage_I_factors["AIPD"] = aggregates.IPD / aggregates.IPD[SYR]

    # add on benefit factors, which are undefined before SYR
    Stage_I_factors["ABENEFITS"] = benefit_factors(
        range(SYR, EYR + 1)
    ).transpose()[0]

    # drop the years before SYR and write the file for build_growfactors.py
    Stage_I_factors.drop([2008, 2009, 2010]).to_csv(
        OUTFILE, float_format="%.4f", index_label="YEAR"
    )


if __name__ == "__main__":
    create_stage1_factors()
