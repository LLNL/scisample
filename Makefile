# This Makefile is slightly modified from the makefile from
# LLNL's internal `codepy` tool
# Author: Brian Daub
# Date: 12/14/19

PYTEST = py.test --cov=scisample --cov-report html tests

sphinx:
	sphinx-apidoc -f -M -o docs/source/ scisample
	cd docs && make html

wheel:
	rm -rf build
	python setup.py bdist_wheel

test:
	$(PYTEST)

unit:
	$(PYTEST) -k "unit and not integration and not system and not acceptance"

integration:
	$(PYTEST) -k "integration and not unit and not system and not acceptance"

system:
	$(PYTEST) -k "system or acceptance"

lint:
	flake8 scisample
