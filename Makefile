install-lite:
	pip install -e .
install:
	pip install -e .[reweight]

test-lite:
	TEST_MODE=lite pytest . -v

test:
	TEST_MODE=full pytest . -v

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_flat_file.py

all: format test