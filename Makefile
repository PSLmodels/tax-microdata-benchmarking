.PHONY=install
install:
	pip install -e .

.PHONY=clean
clean:
	rm -f tmd/storage/output/tmd*
	rm -f tmd/storage/output/cached*
	rm -f tmd/storage/output/preimpute_tmd.csv.gz

tmd/storage/output/tmd.csv.gz:
	python tmd/create_taxcalc_input_variables.py

tmd/storage/output/tmd_weights.csv.gz:
	python tmd/create_taxcalc_sampling_weights.py

tmd/storage/output/tmd_growfactors.csv:
	python tmd/create_taxcalc_growth_factors.py

tmd/storage/output/cached_files:
	python tmd/create_taxcalc_cached_files.py

tmd/storage/output/preimpute_tmd.csv.gz:
	python tmd/create_taxcalc_imputed_variables.py

.PHONY=tmd_files
tmd_files: tmd/storage/output/tmd.csv.gz \
  tmd/storage/output/tmd_weights.csv.gz \
  tmd/storage/output/tmd_growfactors.csv \
  tmd/storage/output/cached_files \
  tmd/storage/output/preimpute_tmd.csv.gz

.PHONY=test
test: tmd_files
	pytest . -v -n4

.PHONY=data
data: install tmd_files test

.PHONY=format
format:
	black . -l 79

PYLINT_DISABLE = duplicate-code,invalid-name,too-many-instance-attributes,too-many-locals,too-many-arguments,too-many-positional-arguments,too-many-statements,too-many-branches,too-many-nested-blocks,too-many-return-statements,broad-exception-caught,missing-function-docstring,missing-module-docstring,missing-class-docstring,too-many-lines,wrong-import-position

PYLINT_OPTIONS = --disable=$(PYLINT_DISABLE) --score=no --jobs=4 \
                 --check-quote-consistency=yes

.PHONY=lint
lint:
	@pycodestyle --ignore=E203,E731,E712,W503 .
	@pylint $(PYLINT_OPTIONS) .
