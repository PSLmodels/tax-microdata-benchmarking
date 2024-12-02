National Data Examination Methods
=================================

This document describes the methods used to compare model-plus-dataset
estimates of basic federal tax amounts against corresponding federal
agency estimates.

The basic tax amounts that will be compared include:
* **Payroll Tax Liability** (federal employee plus employer share)
* **Individual Income Tax Liability** (federal individual income tax)
* **CTC Tax Expenditure** (from the federal child tax credit)
* **EITC Tax Expenditure** (from the federal earned income credit)
* **SSBEN Tax Expenditure** (from excluding some social security benefits from federal AGI)
* **NIIT Tax Expenditure** (from the 3.8% federal surtax on investment income)
* **CGQD Tax Expenditure** (from taxing long-term capital gains and qualified dividends at lower federal rates)
* **QBID Tax Expenditure** (from the 20% federal qualified business income deduction)
* **SALT Tax Expenditure** (from the itemized deduction of state and local income and property taxes)

Federal Agency Estimates
------------------------

The federal agencies --- CBO, JCT, and TSY --- publish tax amounts for
federal fiscal years.  The model-plus-dataset estimates are for
calendar years.  In order to make the federal agency estimates
comparable to the model-plus-dataset estimates, the estimates for the
two fiscal years overlapping with the calendar year are used in a
simple linear interpolation and linear extrapolation of the two fiscal
year estimates to produce a calendar year estimate.  This linear
adjustment is done by the `fy2cy.awk` script using as input one of
three files containing federal agency estimates for a pair of fiscal years.
For example, the following three files contain estimates for fiscal years 2023
and 2024: `cy23_cbo.csv`, `cy23_jct.csv`, and `cy23_tsy.csv`.  There are
three files for each of several years: `cy23`, and `cy26`.
These `.csv` files, all of which are in the `examination` directory,
contain detailed information about the source of the federal agency
estimates.

Model-plus-Dataset Estimates for Phase 1
----------------------------------------

In the first phase of this project, the
[Tax-Calculator (3.5.0)](https://github.com/PSLmodels/Tax-Calculator)
microsimulation model is used to compute tax estimates that correspond
to the federal agency estimates described above.  Tax-Calculator is
used with two different 2023 input datasets.

The first, which is called the `taxdata dataset`, is the CSV-formatted
dataset created for Tax-Calculator in the [taxdata
repository](https://github.com/PSLmodels/taxdata).  It contains 2011
TSY SOI PUF data, supplemented with CPS data, that is extrapolated to
calendar year 2023.

The second, which is called the `phase 1 dataset`, is a CSV-formatted
version of the hierarchical dataset created for the [Policyengine-US
(0.680.0)](https://github.com/PolicyEngine/policyengine-us)
microsimulation model.  It contains 2022 CPS data, enhanced with 2015
TSY SOI PUF data, that is extrapolated to calendar year 2023.
(Subsequent phases of this project will develop other datasets.)

In both these input dataset cases, the same procedure is used to
estimate the amounts corresponding to the federal agency estimates.
This procedure involves using the Tax-Calculator's
command-line-interface tool,
[`tc`](https://taxcalc.pslmodels.org/guide/cli.html), in the
`examination/taxcalculator` directory.  The payroll and individual
income tax liabilities are estimated using the
[`clp.json`](./taxcalculator/clp.json) null reform to produce
estimates for 2023 baseline tax policy.  Each tax expenditure estimate
is generated using a simple reform that negates that feature of
baseline tax policy.  The several `tc` runs are collected into a
single shell script called `examination/taxcalculator/runs.sh`, which
in turn calls the `examination/taxcalculator/execute.sh` script for
each run.  The simple tax expenditure reforms are included in the
following JSON files:

* **CTC Tax Expenditure**: [`ctc.json`](./taxcalculator/ctc.json)
* **EITC Tax Expenditure**: [`eitc.json`](./taxcalculator/eitc.json)
* **SSBEN Tax Expenditure**: [`ssben.json`](./taxcalculator/ssben.json)
* **NIIT Tax Expenditure**: [`niit.json`](./taxcalculator/niit.json)
* **CGQD Tax Expenditure**: [`cgqd.json`](./taxcalculator/cgqd.json)
* **QBID Tax Expenditure**: [`qbid.json`](./taxcalculator/qbid.json)
* **SALT Tax Expenditure**: [`salt.json`](./taxcalculator/salt.json)

The 2023 results from these two sets of `tc` runs are in the
`examination/taxcalculator/puf-23.res-expect` file for the
`taxdata dataset` and in the
`examination/taxcalculator/pe23-23.res-expect` file for the
`phase 1 dataset`.


Model-plus-Dataset Estimates for Phase 2
----------------------------------------

In the second phase of this project, the
[Tax-Calculator (3.5.3)](https://github.com/PSLmodels/Tax-Calculator)
microsimulation model is used to compute tax estimates that correspond
to the federal agency estimates described above.  Tax-Calculator is
used with two different sets of 2023 and 2026 input datasets.

The first, which is called the `taxdata dataset`, is the CSV-formatted
dataset created for Tax-Calculator in the [taxdata
repository](https://github.com/PSLmodels/taxdata).  It contains 2011
TSY SOI PUF data, supplemented with CPS data, that is extrapolated to
calendar year 2023 and to calendar year 2026.

The second, which is called the `phase 2 dataset`, is a CSV-formatted
dataset developed by this project.  It contains 2015 TSY SOI PUF data
enhanced with 2022 CPS data backcasted to 2015.  The resulting 2015
dataset is then extrapolated to 2021 so that the 2021 dataset
generates aggregate and AGI-group statistics similar to those
published by TSY SOI for that year.  Finally, simpler extrapolation
methods are used to generate datasets for 2023 and 2026 from the 2021
dataset.

In both these input dataset cases, the same procedure is used to
estimate the amounts corresponding to the federal agency estimates.
This procedure involves using the Tax-Calculator's
command-line-interface tool,
[`tc`](https://taxcalc.pslmodels.org/guide/cli.html), in the
`examination/taxcalculator` directory.  The payroll and individual
income tax liabilities are estimated using the
[`clp.json`](./taxcalculator/clp.json) null reform to produce
estimates for 2023 or 2026 baseline tax policy.  Each tax expenditure
estimate is generated using a simple reform that negates that feature
of baseline tax policy.  The several `tc` runs are collected into a
single shell script called `examination/taxcalculator/runs.sh`, which
in turn calls the `examination/taxcalculator/execute.sh` script for
each run.  The simple tax expenditure reforms are included in the
following JSON files:

* **CTC Tax Expenditure**: [`ctc.json`](./taxcalculator/ctc.json)
* **EITC Tax Expenditure**: [`eitc.json`](./taxcalculator/eitc.json)
* **SSBEN Tax Expenditure**: [`ssben.json`](./taxcalculator/ssben.json)
* **NIIT Tax Expenditure**: [`niit.json`](./taxcalculator/niit.json)
* **CGQD Tax Expenditure**: [`cgqd.json`](./taxcalculator/cgqd.json)
* **QBID Tax Expenditure**: [`qbid.json`](./taxcalculator/qbid.json)
* **SALT Tax Expenditure**: [`salt.json`](./taxcalculator/salt.json)

The 2023 results from these two sets of `tc` runs are in the
`examination/taxcalculator/puf-23.res-expect` file for the
`taxdata dataset` and in the
`examination/taxcalculator/xb23-23.res-expect` file for the
`phase 2 dataset`.

The 2026 results from these two sets of `tc` runs are in the
`examination/taxcalculator/puf-26.res-expect` file for the
`taxdata dataset` and in the
`examination/taxcalculator/xb26-26.res-expect` file for the
`phase 2 dataset`.


Model-plus-Dataset Estimates for Phase 3
----------------------------------------

In the third phase of this project, the
[Tax-Calculator (4.2.0)](https://github.com/PSLmodels/Tax-Calculator)
microsimulation model is used to compute tax estimates that correspond
to the federal agency estimates described above.  Tax-Calculator is
used with two different sets of 2023 and 2026 input datasets.

The first, which is called the `taxdata dataset`, is the CSV-formatted
dataset created for Tax-Calculator in the [taxdata
repository](https://github.com/PSLmodels/taxdata).  It contains 2011
TSY SOI PUF data, supplemented with CPS data, that is extrapolated to
calendar year 2023 and to calendar year 2026.

The second, which is called the `phase 3 dataset`, is a CSV-formatted
dataset developed by this project.  It contains 2015 TSY SOI PUF data
enhanced with 2022 CPS data backcasted to 2015.  The resulting 2015
dataset is then extrapolated to 2021 and reweighted so that the 2021
dataset generates aggregate and AGI-group statistics similar to those
published by TSY SOI for that year.

In both these input dataset cases, the same procedure is used to
estimate the amounts corresponding to the federal agency estimates.
This procedure involves using the Tax-Calculator's
command-line-interface tool,
[`tc`](https://taxcalc.pslmodels.org/guide/cli.html), in the
`examination/taxcalculator` directory.  The payroll and individual
income tax liabilities are estimated using the
[`clp.json`](./taxcalculator/clp.json) null reform to produce
estimates for 2023 or 2026 baseline tax policy.  Each tax expenditure
estimate is generated using a simple reform that negates that feature
of baseline tax policy.  The several `tc` runs are collected into a
single shell script called `examination/taxcalculator/runs.sh`, which
in turn calls the `examination/taxcalculator/execute.sh` script for
each run.  The simple tax expenditure reforms are included in the
following JSON files:

* **CTC Tax Expenditure**: [`ctc.json`](./taxcalculator/ctc.json)
* **EITC Tax Expenditure**: [`eitc.json`](./taxcalculator/eitc.json)
* **SSBEN Tax Expenditure**: [`ssben.json`](./taxcalculator/ssben.json)
* **NIIT Tax Expenditure**: [`niit.json`](./taxcalculator/niit.json)
* **CGQD Tax Expenditure**: [`cgqd.json`](./taxcalculator/cgqd.json)
* **QBID Tax Expenditure**: [`qbid.json`](./taxcalculator/qbid.json)
* **SALT Tax Expenditure**: [`salt.json`](./taxcalculator/salt.json)

The 2023 results from these two sets of `tc` runs are in the
`examination/taxcalculator/puf-23.res-expect` file for the
`taxdata dataset` and in the
`examination/taxcalculator/tmd-23.res-expect` file for the
`phase 3 dataset`.

The 2026 results from these two sets of `tc` runs are in the
`examination/taxcalculator/puf-26.res-expect` file for the
`taxdata dataset` and in the
`examination/taxcalculator/tmd-26.res-expect` file for the
`phase 3 dataset`.

Model-plus-Dataset National Estimates for Phase 4+
--------------------------------------------------

Beginning in the fourth phase of this project, the most recent version
of [Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator)
microsimulation model is used to compute tax estimates that correspond
to the federal agency estimates described above.  See the [results4+
document](results4.md) for Tax-Calculator version details.

The 2023 and 2026 tax revenue and tax expenditure estimates using the
most recent version of the `tmd` data filse are generated as part of
the [tax expenditures test](../../tests/test_tax_expenditures.py) and
stored in the `tmd/storage/output/tax_expenditures` file.

Here are the details of the tax expenditure reforms:

* **CTC Tax Expenditure**: [`ctc.json`](./taxcalculator/ctc.json)
* **EITC Tax Expenditure**: [`eitc.json`](./taxcalculator/eitc.json)
* **SSBEN Tax Expenditure**: [`ssben.json`](./taxcalculator/ssben.json)
* **NIIT Tax Expenditure**: [`niit.json`](./taxcalculator/niit.json)
* **CGQD Tax Expenditure**: [`cgqd.json`](./taxcalculator/cgqd.json)
* **QBID Tax Expenditure**: [`qbid.json`](./taxcalculator/qbid.json)
* **SALT Tax Expenditure**: [`salt.json`](./taxcalculator/salt.json)
