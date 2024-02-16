install:
	pip install uv
	uv pip install -e .

test:
	pytest .

format:
	black . -l 79

flat-file:
	python tax_microdata_benchmarking/create_flat_file.py

all: format test