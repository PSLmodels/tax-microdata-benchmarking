install:
	pip install -e .

test:
	pytest .

format:
	black . -l 79

flat_file:
	python initial_flat_file/create_flat_file.py