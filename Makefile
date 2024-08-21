install:
	pip install -e .
	python tax_microdata_benchmarking/download_prerequisites.py

tax_microdata_benchmarking/storage/output/tmd_2021.csv:
	python tax_microdata_benchmarking/create_taxcalc_input_variables.py

tax_microdata_benchmarking/storage/output/tmd_growfactors.csv:
	python tax_microdata_benchmarking/create_taxcalc_growth_factors.py

tax_microdata_benchmarking/storage/output/tmd_weights.csv.gz:
	python tax_microdata_benchmarking/create_taxcalc_sampling_weights.py

tmd: tax_microdata_benchmarking/storage/output/tmd_2021.csv \
     tax_microdata_benchmarking/storage/output/tmd_growfactors.csv \
     tax_microdata_benchmarking/storage/output/tmd_weights.csv.gz

test: tmd
	pytest . -v -n4

data: install tmd test

format:
	black . -l 79

documentation:
	jb build docs/book

reweighting-visualisation:
	tensorboard --logdir=tax_microdata_benchmarking/storage/output/reweighting

tax-expenditures-report: tmd
	-pytest . --disable-warnings -m taxexp
	diff tax_microdata_benchmarking/storage/output/tax_expenditures \
             tax_microdata_benchmarking/examination/tax_expenditures
