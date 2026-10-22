# Virtual occupancy sensing (MMA3001) -- reproducible workflow.
#
# The data-dependent targets (ingest, experiment, sensitivity, figures, all)
# need the raw MSI exports unzipped under data/raw/ -- see data/README.md.
# `make test`, `make test-report`, `make docs` and `make profile` need no data.

PY      ?= .venv/bin/python
VENV    ?= .venv
PYTHON3 ?= python3

.PHONY: all install test test-report ingest experiment experiment-synthetic \
        sensitivity profile figures docs clean-docs help

help:
	@echo "Targets: install test test-report ingest experiment sensitivity profile figures docs all"

# Create the virtual environment (if absent) and install the package + dev tools.
install:
	test -x $(PY) || $(PYTHON3) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

# HTML test report (self-contained) and JUnit XML for the submission.
test-report:
	$(PY) -m pytest -q --html=reports/test_report.html --self-contained-html \
	    --junitxml=reports/test_results.xml

# Raw CSV/XLSX -> data/processed/*.parquet and reports/results/audit/*.csv
ingest:
	$(PY) scripts/ingest.py

# Model comparison on manufactured rooms (no data needed), then on real rooms.
experiment-synthetic:
	$(PY) scripts/run_experiment.py --synthetic

experiment: experiment-synthetic
	$(PY) scripts/run_experiment.py --data data/processed/rooms.parquet

sensitivity:
	$(PY) scripts/sensitivity_synthetic.py
	$(PY) scripts/sensitivity_real.py

profile:
	$(PY) scripts/profile_performance.py

figures:
	$(PY) scripts/make_figures.py

# API documentation (Google-style docstrings, LaTeX maths) -> docs/index.html
docs:
	$(PY) -m pdoc -d google --math -o docs occupancy

clean-docs:
	rm -rf docs/*

# Full pipeline from raw data to documentation.
all: ingest experiment sensitivity profile figures test-report docs
