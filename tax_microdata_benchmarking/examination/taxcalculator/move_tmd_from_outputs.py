import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER

df = pd.read_csv(STORAGE_FOLDER / "output" / "tmd_2021.csv")
df.to_csv(
    STORAGE_FOLDER.parent / "examination" / "taxcalculator" / "tmd.csv.zip",
    index=False,
)
