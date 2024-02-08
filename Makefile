install:
	pip install -r requirements.txt

test:
	pytest .

format:
	black . -l 79
