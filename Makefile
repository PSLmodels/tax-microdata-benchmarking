install: prerequisite-data-files
	pip install -e .

test:
	pytest . -v

format:
	black . -l 79

prerequisite-data-files:
	python tax_microdata_benchmarking/download_prerequisites.py

flat-file:
	python tax_microdata_benchmarking/create_all_datasets.py

data: install flat-file test

documentation:
	jb build docs
