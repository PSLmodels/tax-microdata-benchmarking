import pandas as pd
from pathlib import Path

output = Path(__file__).parent

puf_pe_21 = pd.read_csv(output / "puf_ecps_2021.csv.gz")
pe_21 = pd.read_csv(output / "ecps_2021.csv.gz")
td_23 = pd.read_csv(output / "taxdata_puf_2023.csv.gz")
