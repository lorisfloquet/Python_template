.PHONY: init test test-cov run run-v clean

init:
	pip install -r requirements.txt

test:
	pytest -m "not slow"

test-cov:
	pytest -m "not slow" --cov=src --cov-report=term-missing

run:
	PYTHONPATH=src python3 src/mypkg/main.py

run-v:
	PYTHONPATH=src python3 src/mypkg/main.py -v

clean:
	@find . -name "__pycache__" -type d -exec rm -rf {} +
