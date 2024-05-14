install:
	pip install -e .

test:
	TEST_MODE=full pytest . -v

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_all_datasets.py

all: format test

docs:
	jb build docs
