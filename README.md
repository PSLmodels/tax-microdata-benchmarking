# tax-microdata

This repository contains all working files for a project to develop
validated input files for use in
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).

For Tax-Calculator results generated when using these TMD input files,
see [this
folder](https://github.com/PSLmodels/Tax-Calculator/tree/master/taxcalc/cli/input_data_tests).

The **current TMD version is 2.0.0**, which was released on March 29, 2026,
and includes the following significant improvements:

- generate national, state, and Congressional district, input files
for **2022**:
[#470](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/470)
[#471](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/471)
[#472](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/472)
[#473](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/473)
[#474](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/474)
- improve the selection of CPS tax units to represent nonfilers:
[#438](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/438)
- vastly improve the reweighting algorithm:
[#416](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/416)
- impute values for three variables used in new OBBBA deductions:
[#397](https://github.com/PSLmodels/tax-microdata-benchmarking/pull/397)

## Usage instructions

In order to use the code in this repository, you need to license the
2015 PUF from IRS/SOI.  Once you have done that, you will have two
CSV-formatted files from IRS/SOI: `puf_2015.csv` and
`demographics_2015.csv`.

To generate the TMD files from the PUF files, do this:

1. Copy the two 2015 PUF files to the `tmd/storage/input` folder
2. Install the SIPP files described in `tmd/storage/input/SIPP24/README.md`
3. Install the CEX files described in `tmd/storage/input/CEX23/README.md`
4. Run `make clean` in the repository's top-level folder
5. Run `make data` in the repository's top-level folder

The `make data` command creates and tests the three national
`tmd*csv*` data files, which are located in the `tmd/storage/output`
folder.  Read [this
documentation](https://taxcalc.pslmodels.org/usage/data.html#irs-public-use-data-tmd-csv)
on how to use these three files with Tax-Calculator.  Also, you can
look at the tests in this repository to see Python code that uses the
TMD files with Tax-Calculator.

## Sub-national area weights

The repository also produces **per-area weight files** that adapt the national
TMD microdata to a specific state or Congressional district.  The records do not
change; only the weights do, so that weighted sums and targeted distributional
values match state-level (or CD-level) totals from IRS Statistics of Income
(SOI) and other published sources.

See [`tmd/areas/README.md`](tmd/areas/README.md) for how to build
the weights, what files you get, and how to use them — with or
without Tax-Calculator.
