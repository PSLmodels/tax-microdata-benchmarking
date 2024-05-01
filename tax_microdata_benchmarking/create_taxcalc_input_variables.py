"""
Construct tmd.csv.gz, a Tax-Calculator-style input variable file for 2021.
"""

from tax_microdata_benchmarking.create_flat_file import (
    create_stacked_flat_file,
)
from tax_microdata_benchmarking.adjust_qbi import (
    add_pt_w2_wages,
)


TAXYEAR = 2021


def create_variable_file():
    """
    Create Tax-Calculator-style input variable file for TAXYEAR.
    """
    vdf = create_stacked_flat_file(target_year=TAXYEAR)
    vdf.FLPDYR = TAXYEAR
    vdf = add_pt_w2_wages(vdf, TAXYEAR)
    vdf.to_csv(
        "tmd.csv.gz",
        index=False,
        float_format="%.2f",
        compression="gzip",
    )


if __name__ == "__main__":
    create_variable_file()
