from tax_microdata_benchmarking.cloud import download_gh_release_asset

def download_puf():
    download_gh_release_asset(
        repo="nikhilwoodruff/tax-microdata-benchmarking-releases",
        release_name="puf-2015",
        asset_name="puf_2015.csv.gz",
    )