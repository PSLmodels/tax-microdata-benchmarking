"""
This module enables generation of all datasets involved in the repo.
"""

from tax_microdata_benchmarking.datasets.tmd import create_tmd_2021
from tax_microdata_benchmarking.storage import STORAGE_FOLDER
import time

outputs = STORAGE_FOLDER / "output"

generation_functions = [
    (create_tmd_2021, "tmd_2021.csv"),
]


def main():
    for generation_function, filename in generation_functions:
        print(f"Generating {filename}...")
        start_time = time.time()
        data = generation_function()
        data.to_csv(outputs / filename, index=False)
        duration = time.time() - start_time
        print(f"   ...completed {filename} in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
