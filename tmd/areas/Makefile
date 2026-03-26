# Area weighting pipeline: states and congressional districts.
#
# Prerequisites: run `make data` from the repo root first.
#
# Usage from repo root:
#   make -C tmd/areas states
#   make -C tmd/areas cds WORKERS=16
#   make -C tmd/areas state-targets    (shares + targets only)
#   make -C tmd/areas cd-targets       (shares + targets only)
#
# Override worker count: make -C tmd/areas cds WORKERS=16

WORKERS ?= 8
ROOT := $(realpath $(CURDIR)/../..)

# --- State pipeline ---

.PHONY: state-shares state-targets state-weights states

state-shares:
	python -m tmd.areas.prepare_shares --scope states

state-targets: state-shares
	python -m tmd.areas.prepare_targets --scope states

state-weights: state-targets
	python -m tmd.areas.solve_weights --scope states --workers $(WORKERS)

states: state-weights
	python -m pytest $(ROOT)/tests/test_prepare_targets.py -v -k "not CD"
	python -m pytest $(ROOT)/tests/test_state_weight_results.py -v
	python -m tmd.areas.quality_report --scope states

# --- Congressional district pipeline ---

.PHONY: cd-shares cd-targets cd-weights cds

cd-shares:
	python -m tmd.areas.prepare_shares --scope cds

cd-targets: cd-shares
	python -m tmd.areas.prepare_targets --scope cds

cd-weights: cd-targets
	python -m tmd.areas.solve_weights --scope cds --workers $(WORKERS)

cds: cd-weights
	python -m pytest $(ROOT)/tests/test_prepare_targets.py -v -k CD
	python -m tmd.areas.quality_report --scope cds
