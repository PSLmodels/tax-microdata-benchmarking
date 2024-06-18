"""
This module enables generation of all datasets involved in the repo.
"""

from tax_microdata_benchmarking.datasets import *
from tax_microdata_benchmarking.create_taxcalc_growth_factors import (
    create_factors_file,
)
from tax_microdata_benchmarking.create_taxcalc_sampling_weights import (
    create_weights_file,
)
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import time

outputs = STORAGE_FOLDER / "output"

generation_functions = [
    (create_tc_puf_2015, "tc_puf_2015.csv"),
    (create_tc_puf_2021, "tc_puf_2021.csv"),
    (create_tmd_2021, "tmd_2021.csv"),
    (create_uprated_puf_2021, "puf_2021.csv"),
]


def main():
    for generation_function, filename in generation_functions:
        print(f"Generating {filename}...")
        start_time = time.time()
        data = generation_function()
        data.to_csv(outputs / filename, index=False)
        duration = time.time() - start_time
        print(f"   ...completed {filename} in {duration:.2f} seconds")

    create_weights_file()
    create_factors_file()


if __name__ == "__main__":
    main()
