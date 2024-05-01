from tax_microdata_benchmarking.cloud import download_gh_release_asset
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import pandas as pd


def load_puf(
    skip_if_exists: bool = True,
) -> pd.DataFrame:
    """
    Download the IRS Public Use File (PUF) from a private GitHub repo.

    Args:
        skip_if_exists (bool): Whether to skip the download if the file already exists.

    Returns:
        pd.DataFrame: The PUF as a DataFrame.
    """
    path = STORAGE_FOLDER / "puf_2015.csv.gz"
    if skip_if_exists and path.exists():
        return pd.read_csv(path)
    puf = download_gh_release_asset(
        repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
        release_name="puf-2015",
        asset_name="puf_2015.csv.gz",
    )
    puf.to_csv(path, index=False)
    return puf


def create_puf():
    puf = load_puf()
