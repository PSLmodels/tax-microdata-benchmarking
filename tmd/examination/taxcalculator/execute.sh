#!/bin/zsh
# Executes reform calculations using Tax-Calculator's tc command-line tool.
# USAGE:
#   ./execute.sh  DATA_CSV_FILENAME  TAX_YEAR  REFORM_FILENAME

tc $1 $2 --reform $3.json --exact --tables --dump --dvars $3.dvars #--sqldb

exit 0
