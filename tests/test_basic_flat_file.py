import os

test_mode = os.environ.get("TEST_MODE", "lite")

def test_flat_file_runs():
    from tax_microdata_benchmarking.create_flat_file import (
        create_stacked_flat_file,
    )

    create_stacked_flat_file(2021, reweight=test_mode == "full")
