.PHONY: init lib test test-cov run run-v clean

init:
	pip install -r requirements.txt

lib:
	pip install -e .

test:
	pytest -m "not slow"

test-cov:
	pytest -m "not slow" --cov=mypkg --cov-report=term-missing

run:
	PYTHONPATH="." python3 mypkg/main.py

run-v:
	PYTHONPATH="." python3 mypkg/main.py -v

clean:
	@find $(CURDIR) -type d -name "__pycache__" -exec rm -rf {} +
	@find $(CURDIR) -type d -name '.pytest_cache' -exec rm -rf {} +
	@find $(CURDIR) -type d -name '*.egg-info' -exec rm -rf {} +
	@find $(CURDIR) -type f -name '.coverage' -delete

archive: clean
	zip -r mypkg.zip . -x "*.git/*" ".vscode/*"
