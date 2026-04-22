# Force bash with pipefail so that a failure in the first stage of a
# pipeline (e.g., `python ... 2>&1 | tee log`) propagates as a recipe
# failure instead of being masked by tee's exit status.
SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c

.PHONY=install
install:
	pip install -e .

.PHONY=clean
clean:
	rm -f tmd/storage/output/tmd*
	rm -f tmd/storage/output/cached*
	rm -f tmd/storage/output/preimpute_tmd.csv.gz
	rm -f tmd/storage/output/make_data.log

# Main imputation/calibration pipeline.  Output is tee'd to
# make_data.log so that warnings emitted during the run (pandas,
# numpy, taxcalc, etc.) are captured and can be reviewed afterward via
# `make warnings`.  On-screen output during the build is unchanged.
tmd/storage/output/tmd.csv.gz:
	python tmd/create_taxcalc_input_variables.py 2>&1 \
	    | tee tmd/storage/output/make_data.log

tmd/storage/output/tmd_weights.csv.gz:
	python tmd/create_taxcalc_sampling_weights.py

tmd/storage/output/tmd_growfactors.csv:
	python tmd/create_taxcalc_growth_factors.py

tmd/storage/output/cached_files:
	python tmd/create_taxcalc_cached_files.py

tmd/storage/output/preimpute_tmd.csv.gz:
	python tmd/create_taxcalc_imputed_variables.py

.PHONY=tmd_files
tmd_files: tmd/storage/output/tmd.csv.gz \
  tmd/storage/output/tmd_weights.csv.gz \
  tmd/storage/output/tmd_growfactors.csv \
  tmd/storage/output/cached_files \
  tmd/storage/output/preimpute_tmd.csv.gz

.PHONY=test
test: tmd_files
	pytest . -v -n4 \
	    --ignore=tests/national_targets_pipeline \
	    --ignore=tests/test_fingerprint.py \
	    --ignore=tests/test_state_weight_results.py \
	    --ignore=tests/test_cd_crosswalk.py \
	    --ignore=tests/test_prepare_targets.py

.PHONY=data
data: install tmd_files test

.PHONY=warnings
warnings:
	@log=tmd/storage/output/make_data.log; \
	if [ ! -f "$$log" ]; then \
	    echo "No $$log found; run 'make data' first."; \
	else \
	    hits=$$(grep -niE 'warning|deprecat' "$$log" || true); \
	    if [ -z "$$hits" ]; then \
	        echo "No warnings found in $$log."; \
	    else \
	        echo "Warnings in $$log:"; \
	        echo "$$hits"; \
	    fi; \
	fi

.PHONY=format
format:
	black . -l 79

PYLINT_DISABLE = duplicate-code,invalid-name,too-many-instance-attributes,too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-branches,too-many-nested-blocks,too-many-return-statements,broad-exception-caught,missing-function-docstring,missing-module-docstring,missing-class-docstring

PYLINT_OPTIONS = --disable=$(PYLINT_DISABLE) --score=no --jobs=4 \
                 --check-quote-consistency=yes

.PHONY=lint
lint:
	@pycodestyle --ignore=E203,E731,E712,W503 .
	@pylint $(PYLINT_OPTIONS) .
