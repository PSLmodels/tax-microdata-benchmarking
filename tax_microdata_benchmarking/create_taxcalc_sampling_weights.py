"""
Construct tmd_weights.csv.gz, a Tax-Calculator-style weights file for 2021+
consistent with initial-year weights in the tmd.csv.gz input variable file.
"""
import yaml
import pandas as pd

FIRST_YEAR = 2021
LAST_YEAR = 2034
VARFILE = "tmd.csv.gz"
POPFILE = "cbo_population_forecast.yaml"
WGTFILE = "tmd_weights.csv.gz"


def create_weights_file():
    """
    Create Tax-Calculator-style weights file for FIRST_YEAR through LAST_YEAR.
    """
    # get population forecast
    with open(POPFILE, "r", encoding="utf-8") as pfile:
        pop = yaml.safe_load(pfile.read())

    # get FIRST_YEAR weights from VARFILE
    vdf = pd.read_csv(VARFILE)
    weights = vdf.s006

    # construct dictionary of weights by year
    wdict = {f"WT{FIRST_YEAR}": weights}
    cum_pop_growth = 1.0
    for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
        annual_pop_growth = pop[year] / pop[year - 1]
        cum_pop_growth *= annual_pop_growth
        wght = weights.copy() * cum_pop_growth
        wdict[f"WT{year}"] = wght

    # write weights to CSV-formatted file
    wdf = pd.DataFrame.from_dict(wdict)
    wdf.to_csv(WGTFILE, index=False, float_format="%.2f", compression="gzip")


if __name__ == "__main__":
    create_weights_file()
