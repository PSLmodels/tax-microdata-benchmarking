"""
Cumulative growth factors used to extrapolate TAXYEAR sampling weights.

Weights for years after TAXYEAR are the TAXYEAR weights scaled by the
projected growth in the number of individual income tax returns, which is
what a sampling weight counts.  Total population growth was used for this
before, but CBO projects returns growing about twice as fast as population
because the filing share of the population keeps rising, so population
growth left aggregate AGI and income tax revenue progressively below CBO's
projections after TAXYEAR.

The return projection ends with the CBO budget window, well short of the
last weight year, so growth reverts to population growth after the last
projected return year, which holds returns per capita fixed at its final
projected value.
"""

import yaml
from tmd.storage import STORAGE_FOLDER
from tmd.imputation_assumptions import POPULATION_FILE, RETURNS_FILE


def _read(filename):
    with open(
        STORAGE_FOLDER / "input" / filename, "r", encoding="utf-8"
    ) as ifile:
        return yaml.safe_load(ifile.read())


def cumulative_growth(
    first_year,
    last_year,
    population_file=POPULATION_FILE,
    returns_file=RETURNS_FILE,
):
    """
    Return dictionary of cumulative weight growth factors, by calendar year,
    relative to first_year, for each year in [first_year, last_year].

    Growth is the year-over-year growth in projected tax returns through the
    last year covered by returns_file, and population growth thereafter.  If
    returns_file is None, population growth is used for every year.
    """
    pop = _read(population_file)
    rets = _read(returns_file) if returns_file else {}
    last_rets_year = max(rets) if rets else first_year
    if rets and first_year not in rets:
        raise ValueError(
            f"{returns_file} has no {first_year} data, "
            "so it cannot anchor weight extrapolation"
        )

    growth = {first_year: 1.0}
    cum = 1.0
    for year in range(first_year + 1, last_year + 1):
        if year <= last_rets_year:
            cum *= rets[year] / rets[year - 1]
        else:
            cum *= pop[year] / pop[year - 1]
        growth[year] = cum
    return growth
