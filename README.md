# tax-microdata

This repository contains all working files for a project to develop a
general-purpose validated microdata file for use in
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) and
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).  The
development will proceed in several phases.

## Usage instructions

In order to use the code in this repository, you need to license the
2015 PUF from IRS/SOI.  Once you have done that, you will have two
CSV-formatted files from IRS/SOI: `puf_2015.csv` and
`demographics_2015.csv`.

To generate the TMD files from the PUF files, do this:

1. Copy the two 2015 PUF files to the `tmd/storage/input` folder
2. Run `make data` in the repository's top-level folder

The `make data` command creates and tests the three `tmd*csv*` data
files, which are located in the `tmd/storage/output` folder.  Read
[this
documentation](https://taxcalc.pslmodels.org/usage/data.html#irs-public-use-data-tmd-csv)
on how to use these three files with Tax-Calculator.  Also, you can
look at the tests in this repository to see Python code that uses the
TMD files with Tax-Calculator.

## Examination results

To assess, review the data examination results that compare federal
agency tax estimates with those generated using the national microdata
files created in each project phase:
* [phase 1 results](./tmd/examination/results1.md)
* [phase 2 results](./tmd/examination/results2.md)
* [phase 3 results](./tmd/examination/results3.md)
* [phase 4+ results](./tmd/examination/results4.md)
