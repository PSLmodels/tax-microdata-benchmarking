install:
	pip install -e .
	pip install git+https://github.com/policyengine/policyengine_us@nikhilwoodruff/issue4410

test:
	pytest .

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_flat_file.py

all: format test