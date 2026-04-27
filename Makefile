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
	rm -f tmd/storage/output/make_data_*.log

# Each of the five build stages below tees its stdout+stderr to a
# per-stage log file in tmd/storage/output/.  On-screen output during
# `make data` is unchanged; the log files are a byproduct so that any
# warnings emitted during the run (pandas, numpy, taxcalc, scipy,
# etc.) can be reviewed afterward via `make warnings`.  Per-stage
# (rather than shared) log files avoid interleaving when the recipes
# are run with `make -j`.
#
# `python -u` forces unbuffered stdout/stderr.  Without it, piping
# python through `tee` causes python to switch from line buffering
# (its default when stdout is a terminal) to block buffering (its
# default when stdout is a pipe), which makes long-running stages
# appear silent on the terminal until the buffer flushes.  The `-u`
# flag preserves the pre-tee behavior of seeing progress in real time.

tmd/storage/output/tmd.csv.gz:
	python -u tmd/create_taxcalc_input_variables.py 2>&1 \
	    | tee tmd/storage/output/make_data_tmd.log

tmd/storage/output/tmd_weights.csv.gz:
	python -u tmd/create_taxcalc_sampling_weights.py 2>&1 \
	    | tee tmd/storage/output/make_data_weights.log

tmd/storage/output/tmd_growfactors.csv:
	python -u tmd/create_taxcalc_growth_factors.py 2>&1 \
	    | tee tmd/storage/output/make_data_growfactors.log

tmd/storage/output/cached_files:
	python -u tmd/create_taxcalc_cached_files.py 2>&1 \
	    | tee tmd/storage/output/make_data_cached.log

tmd/storage/output/preimpute_tmd.csv.gz:
	python -u tmd/create_taxcalc_imputed_variables.py 2>&1 \
	    | tee tmd/storage/output/make_data_preimpute.log

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
data: install clean format lint tmd_files test warnings

.PHONY=warnings
warnings:
	@logs=$$(ls tmd/storage/output/make_data_*.log 2>/dev/null || true); \
	if [ -z "$$logs" ]; then \
	    echo "No tmd/storage/output/make_data_*.log files found; run 'make data' first."; \
	else \
	    hits=$$(grep -nHE '[Ww]arning|[Dd]eprecat|Traceback|\bERROR\b|Error:' $$logs || true); \
	    if [ -z "$$hits" ]; then \
	        echo "No warnings found in pipeline logs:"; \
	        for f in $$logs; do echo "  $$f"; done; \
	    else \
	        echo "Warnings in pipeline logs:"; \
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
