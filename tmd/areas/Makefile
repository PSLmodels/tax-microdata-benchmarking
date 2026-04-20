# Area weighting pipeline: states and congressional districts.
#
# Prerequisites: run `make data` from the repo root first.
#
# Usage from repo root:
#   make -C tmd/areas states
#   make -C tmd/areas cds-118 WORKERS=16     (118th Congress boundaries)
#   make -C tmd/areas cds-119 WORKERS=16     (119th Congress boundaries)
#   make -C tmd/areas state-targets          (shares + targets only)
#   make -C tmd/areas cd-targets CONGRESS=118  (shares + targets only)
#
# The CD pipeline requires an explicit Congressional session because
# the 118th and 119th Congresses have different district boundaries
# for AL, GA, LA, NY, and NC.  Use CONGRESS=118 or CONGRESS=119 (or
# the convenience targets cds-118 / cds-119).
#
# Override worker count: make -C tmd/areas cds-118 WORKERS=16

WORKERS ?= 8
CONGRESS ?= 118
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
#
# All CD targets require CONGRESS (118 or 119).  Default is 118.

.PHONY: cd-shares cd-targets cd-weights cds cds-118 cds-119

cd-shares:
	python -m tmd.areas.prepare_shares --scope cds --congress $(CONGRESS)

cd-targets: cd-shares
	python -m tmd.areas.prepare_targets --scope cds --congress $(CONGRESS)

cd-weights: cd-targets
	python -m tmd.areas.solve_weights --scope cds --congress $(CONGRESS) \
	    --workers $(WORKERS)

cds: cd-weights
	python -m pytest $(ROOT)/tests/test_prepare_targets.py -v -k CD
	python -m tmd.areas.quality_report --scope cds --congress $(CONGRESS)

# Convenience aliases that set CONGRESS explicitly.

cds-118:
	$(MAKE) cds CONGRESS=118 WORKERS=$(WORKERS)

cds-119:
	$(MAKE) cds CONGRESS=119 WORKERS=$(WORKERS)
