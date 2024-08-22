from tmd.utils.cloud import download_gh_release_asset
from tmd.storage import STORAGE_FOLDER


def download_prerequisites():
    puf_2015 = download_gh_release_asset(
        repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
        release_name="tmd-prerequisites",
        asset_name="puf_2015.csv",
    )

    puf_2015.to_csv(STORAGE_FOLDER / "input" / "puf_2015.csv", index=False)

    demographics_2015 = download_gh_release_asset(
        repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
        release_name="tmd-prerequisites",
        asset_name="demographics_2015.csv",
    )

    demographics_2015.to_csv(
        STORAGE_FOLDER / "input" / "demographics_2015.csv", index=False
    )


if __name__ == "__main__":
    download_prerequisites()
    print("Prerequisites downloaded.")
