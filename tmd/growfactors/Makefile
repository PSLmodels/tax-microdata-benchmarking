# Growth factors pipeline: Census/CBO/IRS/SOI source data -> growfactors CSV
#
# This pipeline is pre-processing that runs *before* `make data`.  It
# produces committed artifacts (data/Stage_I_factors.csv and
# tmd/storage/input/taxdata<VINTAGE>_growfactors.csv) that `make data`
# then consumes via tmd/create_raw_growth_factors.py.  Most users never
# need to run it; re-run it only when refreshing the source data for a
# new vintage.
#
# Usage from repo root:
#  make -C tmd/growfactors stage1  # stage 1: source data -> Stage_I_factors.csv
#  make -C tmd/growfactors factors # stages 1+2: -> storage/input/*_growfactors.csv
#  make -C tmd/growfactors test    # pipeline tests only
#  make -C tmd/growfactors all     # all stages + pipeline tests
#
# The update-* targets below FETCH FROM LIVE WEBSITES (CBO, BLS, SSA,
# IRS).  They are never invoked by `all`, by `factors`, or by `make
# data`.  Run them by hand once a year, inspect the resulting diff in
# data/, and commit it as its own pull request.
# See the docs/annual_update.md file for details.

ROOT := $(realpath $(CURDIR)/../..)
INPUT := $(ROOT)/tmd/storage/input

# Vintage of the growfactors file this pipeline currently produces.
VINTAGE ?= 26
GROWFACTORS := $(INPUT)/taxdata$(VINTAGE)_growfactors.csv

STAGE1_DEPS := build_stage1_factors.py \
               data/CBO_baseline.csv \
               data/IRS_return_projection.csv \
               data/NC-EST2014-AGESEX-RES.csv \
               data/NP2014_D1.csv \
               data/SOI_estimates.csv \
               data/US-EST00INT-ALLDATA.csv \
               data/benefitprograms.csv

.PHONY: all stage1 factors test clean update-cbo update-soi

data/Stage_I_factors.csv: $(STAGE1_DEPS)
	python -m tmd.growfactors.build_stage1_factors

stage1: data/Stage_I_factors.csv

$(GROWFACTORS): build_growfactors.py \
                data/Stage_I_factors.csv \
                data/benefit_growth_rates.csv
	python -m tmd.growfactors.build_growfactors --vintage $(VINTAGE)

factors: $(GROWFACTORS)

test:
	python -m pytest $(ROOT)/tests/growfactors_pipeline -v
	python -m pytest $(ROOT)/tests/test_growfactors.py -v

all: factors test

# Removes only regenerable pipeline artifacts, never data/ source files.
clean:
	rm -f data/Stage_I_factors.csv

# ---- annual source-data refresh: NETWORK ACCESS, run by hand only ----
# Requires the optional extras:  pip install -e ".[growfactors-update]"

update-cbo:
	python -m tmd.growfactors.update.update_cbo

update-soi:
	@test -n "$(YEAR)" || { echo "usage: make update-soi YEAR=2019 SOIPATH=/path/to/soi/files"; exit 1; }
	@test -n "$(SOIPATH)" || { echo "usage: make update-soi YEAR=2019 SOIPATH=/path/to/soi/files"; exit 1; }
	python -m tmd.growfactors.update.update_soi $(YEAR) $(SOIPATH)
