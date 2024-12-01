# tax-microdata

This repository contains all working files for a project to develop a
general-purpose validated microdata file for use in
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) and
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).  The
development will proceed in several phases.

## Usage instructions

To generate the microdata files:

1. Run `export POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN=***`
2. Run `export PSL_TAX_MICRODATA_RELEASE_AUTH_TOKEN=***`
3. Run `make data`

The two tokens can be obtained from [Don Boyd](mailto:donboyd5@gmail.com).

## Examination results

To assess, review the data examination results that compare federal
agency tax estimates with those generated using the national microdata
files created in each project phase:
[phase 1 results](./tmd/examination/results1.md) and
[phase 2 results](./tmd/examination/results2.md) and
[phase 3 results](./tmd/examination/results3.md) and
[phase 4+ results](./tmd/examination/results4.md).
