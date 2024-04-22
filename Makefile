install-lite:
	pip install -e .
	pip install git+https://github.com/policyengine/policyengine_us@nikhilwoodruff/issue4410
install:
	pip install -e .[reweight]
	pip install git+https://github.com/policyengine/policyengine_us@nikhilwoodruff/issue4410

test-lite:
	TEST_MODE=lite pytest . -v

test:
	TEST_MODE=full pytest . -v

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_flat_file.py

all: format test