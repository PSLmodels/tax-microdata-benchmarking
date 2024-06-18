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
	jb build docs
