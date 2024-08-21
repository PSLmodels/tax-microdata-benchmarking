.PHONY=clean
clean:
	rm -f tax_microdata_benchmarking/storage/output/*

.PHONY=install
install:
	pip install -e .
	python tax_microdata_benchmarking/download_prerequisites.py

tax_microdata_benchmarking/storage/output/tmd.csv.gz: \
  tax_microdata_benchmarking/imputation_assumptions.py \
  tax_microdata_benchmarking/datasets/tmd.py \
  tax_microdata_benchmarking/datasets/puf.py \
  tax_microdata_benchmarking/datasets/cps.py \
  tax_microdata_benchmarking/datasets/taxcalc_dataset.py \
  tax_microdata_benchmarking/utils/taxcalc_utils.py \
  tax_microdata_benchmarking/utils/imputation.py \
  tax_microdata_benchmarking/utils/is_tax_filer.py \
  tax_microdata_benchmarking/utils/pension_contributions.py \
  tax_microdata_benchmarking/utils/soi_replication.py \
  tax_microdata_benchmarking/utils/soi_targets.py \
  tax_microdata_benchmarking/utils/reweight.py \
  tax_microdata_benchmarking/utils/trace.py \
  tax_microdata_benchmarking/create_taxcalc_input_variables.py
	python tax_microdata_benchmarking/create_taxcalc_input_variables.py

tax_microdata_benchmarking/storage/output/tmd_growfactors.csv: \
  tax_microdata_benchmarking/storage/input/puf_growfactors.csv \
  tax_microdata_benchmarking/create_taxcalc_growth_factors.py
	python tax_microdata_benchmarking/create_taxcalc_growth_factors.py

tax_microdata_benchmarking/storage/output/tmd_weights.csv.gz: \
  tax_microdata_benchmarking/storage/input/cbo_population_forecast.yaml \
  tax_microdata_benchmarking/storage/output/tmd.csv.gz \
  tax_microdata_benchmarking/create_taxcalc_sampling_weights.py
	python tax_microdata_benchmarking/create_taxcalc_sampling_weights.py

.PHONY=tmd_files
tmd_files: tax_microdata_benchmarking/storage/output/tmd.csv.gz \
  tax_microdata_benchmarking/storage/output/tmd_growfactors.csv \
  tax_microdata_benchmarking/storage/output/tmd_weights.csv.gz

.PHONY=test
test: tmd_files
	pytest . -v

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
	tensorboard --logdir=tax_microdata_benchmarking/storage/output/reweighting

.PHONY=tax-expenditures-report
tax-expenditures-report: tmd_files
	-pytest . --disable-warnings -m taxexp
	diff tax_microdata_benchmarking/storage/output/tax_expenditures \
             tax_microdata_benchmarking/examination/tax_expenditures
