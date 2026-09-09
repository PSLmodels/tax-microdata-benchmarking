# tax-microdata

This repository contains all working files for a project to develop
validated input files for use in
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).

For Tax-Calculator results generated when using these TMD input files,
see [this
folder](https://github.com/PSLmodels/Tax-Calculator/tree/master/taxcalc/cli/input_data_tests).

The **current TMD version is 2.2.0**, which was released on September
9, 2026, and is the same as TMD version 2.1.4 except that post-2022
weights are grown using a CBO returns projection (rather than using a
CBO population projection as in prior versions).  See PR #546 for details.

When using version 2.2.0 to generate national or sub-national weights,
the 2022 weights (and hence, the fingerprints) are the same as for
version 2.1.4.  However, all the post-2022 weights are larger.  Note
that the 118th Congressional district weight fingerprints have not
been updated.

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

### State weights

Here is how to generate the state weights after executing `make data`
to generate the national files.  **First**, generate the state weights
by executing this command in the top-level folder of the repository:

```
make -C tmd/areas states WORKERS=8
```

changing the number of workers from 8 to 4 if spare computing power is
needed while this command executes.  Report any test errors or other
problems by opening a new issue.  **Second**, run the optional state
fingerprint test by executing this command in the top-level folder of
the repository:

```
pytest tests/test_fingerprint.py -v -k states
```

Report any test errors or other problems by opening a new issue.

### 119th Congressional district weights

Here is how to generate the 119th Congressional district weights after
executing `make data` to generate the national files.  **First**,
generate the state weights by executing this command in the top-level
folder of the repository:

```
make -C tmd/areas cds-119 WORKERS=8
```

changing the number of workers from 8 to 4 if spare computing power is
needed while this command executes.  Report any test errors or other
problems by opening a new issue.  **Second**, run the optional district
fingerprint test by executing this command in the top-level folder of
the repository:

```
pytest tests/test_fingerprint.py -v -k cd_119
```

Report any test errors or other problems by opening a new issue.
