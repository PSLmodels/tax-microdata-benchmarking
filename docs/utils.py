import pandas as pd
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
from tax_microdata_benchmarking.utils.cloud import download_gh_release_asset
from tqdm import tqdm

output = STORAGE_FOLDER / "output"

file_paths = [
    output / "puf_ecps_2021.csv.gz",
    output / "ecps_2021.csv.gz",
    output / "taxdata_puf_2023.csv.gz",
]

for file_path in tqdm(file_paths):
    if not file_path.exists():
        df = download_gh_release_asset(
            release_name="latest",
            repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
            asset_name=file_path.name,
        )
        df.to_csv(file_path, index=False)

puf_pe = pd.read_csv(output / "puf_ecps_2021.csv.gz")
pe = pd.read_csv(output / "ecps_2021.csv.gz")
td = pd.read_csv(output / "taxdata_puf_2023.csv.gz")
