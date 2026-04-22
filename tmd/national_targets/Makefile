# National targets pipeline: IRS SOI Excel -> soi.csv
#
# This pipeline is pre-processing that runs *before* `make data`.  It
# produces committed artifacts (data/irs_aggregate_values.csv and
# tmd/storage/input/soi.csv) that `make data` then consumes.  Most users
# do not need to run it; re-run it only when adding a new tax year or
# correcting an extraction error.
#
# Usage from repo root:
#   make -C tmd/national_targets extract    # stage 1: IRS Excel -> per-table CSVs
#   make -C tmd/national_targets build      # stages 1 + 2: -> irs_aggregate_values.csv
#   make -C tmd/national_targets soi        # stages 1 + 2 + 3: -> soi.csv
#   make -C tmd/national_targets all        # all stages + pipeline tests
#   make -C tmd/national_targets test       # pipeline tests only
#
# Stage 1 skips per-table CSVs that already exist.  Pass OVERWRITE=1 to
# force re-extraction:
#   make -C tmd/national_targets extract OVERWRITE=1

ROOT := $(realpath $(CURDIR)/../..)

OVERWRITE ?=
ifeq ($(OVERWRITE),1)
EXTRACT_FLAGS := --overwrite
else
EXTRACT_FLAGS :=
endif

.PHONY: all extract build soi test

extract:
	python -m tmd.national_targets.extract_irs_to_csv $(EXTRACT_FLAGS)

build: extract
	python -m tmd.national_targets.build_targets

soi: build
	python -m tmd.national_targets.potential_targets_to_soi

test:
	python -m pytest $(ROOT)/tests/national_targets_pipeline -v

all: soi test
