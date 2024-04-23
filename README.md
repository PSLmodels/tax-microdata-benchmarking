# tax-microdata

This repository contains all working files for a project to develop a
general-purpose validated microdata file for use in
[PolicyEngine-US](https://github.com/PolicyEngine/policyengine-us) and
[Tax-Calculator](https://github.com/PSLmodels/Tax-Calculator).  The
development will proceed in several phases.

To install, clone the repository and run `pip install -e .` from the
root directory.  To check that the installation was successful, run
`make test` or `pytest .` from the root directory. This needs the POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN environment variable to be set- contact [Nikhil Woodruff](mailto:nikhil@policyengine.org) for the code if you have access to the IRS PUF.

To assess, review the data examination results that compare federal
agency tax estimates with those generated using the microdata file
created in each project phase: [phase 1
results](./tax_microdata_benchmarking/examination/results1.md) and
[phase 2
results](./tax_microdata_benchmarking/examination/results2.md).
