.PHONY=install
install:
	pip install -e .
	python tmd/download_prerequisites.py

.PHONY=clean
clean:
	rm -f tmd/storage/output/tmd* tmd/storage/output/cached_files

tmd/storage/output/tmd.csv.gz: \
  setup.py \
  tmd/imputation_assumptions.py \
  tmd/datasets/tmd.py \
  tmd/datasets/puf.py \
  tmd/datasets/cps.py \
  tmd/datasets/taxcalc_dataset.py \
  tmd/utils/taxcalc_utils.py \
  tmd/utils/imputation.py \
  tmd/utils/is_tax_filer.py \
  tmd/utils/pension_contributions.py \
  tmd/utils/soi_replication.py \
  tmd/utils/soi_targets.py \
  tmd/utils/reweight.py \
  tmd/utils/trace.py \
  tmd/create_taxcalc_input_variables.py
	python tmd/create_taxcalc_input_variables.py

tmd/storage/output/tmd_growfactors.csv: \
  tmd/storage/input/puf_growfactors.csv \
  tmd/create_taxcalc_growth_factors.py
	python tmd/create_taxcalc_growth_factors.py

tmd/storage/output/tmd_weights.csv.gz: \
  tmd/storage/input/cbo_population_forecast.yaml \
  tmd/storage/output/tmd.csv.gz \
  tmd/create_taxcalc_sampling_weights.py
	python tmd/create_taxcalc_sampling_weights.py

tmd/storage/output/cached_files: \
  tmd/storage/output/tmd.csv.gz \
  tmd/storage/output/tmd_growfactors.csv \
  tmd/storage/output/tmd_weights.csv.gz
	python tmd/create_taxcalc_cached_files.py

.PHONY=tmd_files
tmd_files: tmd/storage/output/tmd.csv.gz \
  tmd/storage/output/tmd_growfactors.csv \
  tmd/storage/output/tmd_weights.csv.gz \
  tmd/storage/output/cached_files

.PHONY=test
test: tmd_files
	pytest . -v -n4

.PHONY=data
data: install tmd_files test

.PHONY=format
format:
	black . -l 79

.PHONY=documentation
documentation:
	jb build docs/book

.PHONY=reweighting-visualisation
reweighting-visualisation:
	tensorboard --logdir=tmd/storage/output/reweighting

.PHONY=tax-expenditures-report
tax-expenditures-report: tmd_files
	-pytest . --disable-warnings -m taxexp
	diff tmd/storage/output/tax_expenditures \
             tmd/examination/tax_expenditures
