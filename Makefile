install:
	pip install -e .

test:
	pytest . -v

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_all_datasets.py

data: install flat-file test

documentation:
	jb build docs
