install:
	pip install -e .
	python tax_microdata_benchmarking/download_prerequisites.py

test:
	pytest . -v

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_taxcalc_input_variables.py
	python tax_microdata_benchmarking/create_taxcalc_growth_factors.py
	python tax_microdata_benchmarking/create_taxcalc_sampling_weights.py

data: install flat-file test

documentation:
	jb build docs/book

reweighting-visualisation:
	tensorboard --logdir=tax_microdata_benchmarking/storage/output/reweighting

tax_microdata_benchmarking/examination/taxcalculator/tmd.csv.zip:
	python tax_microdata_benchmarking/examination/taxcalculator/move_tmd_from_outputs.py

tax-expenditures-report: tax_microdata_benchmarking/examination/taxcalculator/tmd.csv.zip
	cd tax_microdata_benchmarking/examination/taxcalculator && ./runs.sh tmd 23
