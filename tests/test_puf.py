def test_puf_downloads():
    from tax_microdata_benchmarking.puf import download_puf

    df = download_puf()
    assert (
        len(df.columns) == 223
    ), f"PUF has the wrong number of columns (expected 223, got {len(df.columns)})"
