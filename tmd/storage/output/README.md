# Output files

Three national files suitable for input to Tax-Calculator:
- tmd.csv.gz
- tmd_weights.csv.gz
- tmd_growfactors.csv

## Warning about `tmd_2021.csv` file

There is a special-purpose `tmd_2021.csv` file that includes 2021
Tax-Calculator output variables and the pre-optimization weight,
`s006_original`.  The weights and input variables in this file are not
rounded for Tax-Calculator input (as they are in the `tmd.csv` file),
and therefore, there has always been minor differences between the
content of `tmd_2021.csv` and `tmd.csv` files.  As a result, using the
`tmd_2021.csv` file is not recommended.  There are plans to remove the
`tmd_2021.csv` file in the future.

