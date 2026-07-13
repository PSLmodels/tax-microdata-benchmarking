# tax-microdata

This repository contains all working files for a project to develop
validated input files for use in
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).

For Tax-Calculator results generated when using these TMD input files,
see [this
folder](https://github.com/PSLmodels/Tax-Calculator/tree/master/taxcalc/cli/input_data_tests).

The **current TMD version is 2.1.4**, which was released on July 13,
2026, and is the same as TMD version 2.1.3 except that a new version
of `scikit-learn` (1.9.0), which fixed a bug in its weighted imputation
algorithm, is used (see #534 for details).  Tax-Calculator version 6.7.1
or higher is required to generate the TMD files.

When using version 2.1.4 to generate **sub-national weights**, there
will be (presumably small) differences from the sub-national weights
generated using earlier versions; however, the sub-national weights
fingerprints have not yet been updated.

The earlier TMD 2.0.0 version introduced the following significant
improvements:
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

If on Windows, it is strongly recommended that you use Microsoft's
free [Windows Subsystem for
Linux](https://learn.microsoft.com/en-us/windows/wsl/) to install the
free Ubuntu Linux operating system, within which you can download the
repository code.  If not on Windows, there is nothing to do at this
stage.

In order to use the code in this repository, you need to license the
2015 PUF from IRS/SOI.  Once you have done that, you will have two
CSV-formatted files from IRS/SOI: `puf_2015.csv` and
`demographics_2015.csv`.

To generate the TMD files from the PUF files, do this:

1. Copy the two 2015 PUF files to the `tmd/storage/input` folder
2. Install the SIPP files described in `tmd/storage/input/SIPP24/README.md`
3. Install the CEX files described in `tmd/storage/input/CEX23/README.md`
4. Run `make data` in the repository's top-level folder

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
