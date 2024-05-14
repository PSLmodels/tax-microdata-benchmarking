import os
import requests
import pandas as pd
from io import BytesIO


def gh_api_call(
    path: str,
    method: str = "GET",
    auth_token: str = os.environ["PSL_TAX_MICRODATA_RELEASE_AUTH_TOKEN"],
    **kwargs,
) -> requests.Response:
    """
    Make a call to the GitHub API

    Args:
        path (str): The path to call
        method (str): The HTTP method to use
        auth_token (str): The GitHub personal access token to use
        **kwargs: Additional arguments to pass to requests

    Returns:
        requests.Response: The response object
    """

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        **kwargs.get("headers", {}),
    }

    kwargs = {k: v for k, v in kwargs.items() if k != "headers"}

    return requests.request(
        method, f"https://api.github.com{path}", headers=headers, **kwargs
    )


def download_gh_release_asset(
    repo: str = "nikhilwoodruff/tax-microdata-benchmarking-releases",
    release_name: str = "puf-2015",
    asset_name: str = "puf_2015.csv.gz",
) -> pd.DataFrame:
    """
    Download a GitHub release asset (a gzipped CSV) from a private GitHub repo

    Args:
        repo (str): The GitHub repo to download the asset from
        release_name (str): The name of the release to download

    Returns:
        pd.DataFrame: The release as a DataFrame
    """

    # Get the release ID
    releases = gh_api_call(
        f"/repos/{repo}/releases/tags/{release_name}"
    ).json()

    assert (
        "id" in releases
    ), f"Release {release_name} not found. Available releases: {releases}"

    release_id = releases["id"]

    # Get the asset ID
    assets = gh_api_call(f"/repos/{repo}/releases/{release_id}")
    assets.raise_for_status()

    asset_id = None
    for asset in assets.json()["assets"]:
        if asset["name"] == asset_name:
            asset_id = asset["id"]
            break

    assert (
        asset_id is not None
    ), f"Asset {asset_name} not found. Available assets: {assets.json()['assets']}"

    # Download the asset
    asset = gh_api_call(f"/repos/{repo}/releases/assets/{asset_id}")
    asset.raise_for_status()

    asset_url = asset.json()["url"].replace("https://api.github.com", "")

    asset = gh_api_call(
        asset_url, headers={"Accept": "application/octet-stream"}
    )
    asset.raise_for_status()

    return pd.read_csv(
        BytesIO(asset.content),
        compression="gzip" if asset_name.endswith(".gz") else None,
    )
