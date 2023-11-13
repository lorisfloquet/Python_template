# Python-Template

## Configuration of the project

This project uses a [`config.py`](src/config.py) file that contains all the constants and eventual private access tokens. To setup this project correctly, add this file to the [`.gitignore`](.gitignore) by uncommenting it and then complete the file with your private information when necessary.

## Development and Testing

This project uses a [`Makefile`](Makefile) to streamline common development tasks. Here's how you can utilize it:

### Prerequisites

Ensure you have `make` installed on your system. Most UNIX-based systems come with `make` pre-installed. If not, refer to your system's package manager to install it.

### Setting Up the Project

To set up the project's dependencies, use:

```bash
make init
```

This will install all necessary Python packages listed in [`requirements.txt`](requirements.txt) using `pip`.  

Make sure you have `pip` installed. If it is not the case, you can run alternatively: `python3 -m pip install -r requirements.txt`.

### Running the Project

To execute the main script:

```bash
make run
```

This will run [`src/mypkg/main.py`](src/mypkg/main.py) using Python 3.

```bash
make run-v
```

This will run the same file but with a higher verbosity level.

### Running Tests

To run the tests:

```bash
make test
```

This will run all the tests using `pytest`, except the ones marked as slow.

Make sure you have `pytest` installed. If it is not the case, you can run alternatively: `python3 -m pytest -m "not slow"`.

#### Running all the tests

```bash
pytest
```

This will run absolutely all the tests.

#### Running just one test

```bash
pytest -s -k test_name
```

This will run all the tests that have `test_name` in their name. The `-s` flag is for verbosity, remove it if you don't want to see any `print` while running your tests.

### Testing with Coverage

To run the tests and get a coverage report in the terminal:

```bash
make test-cov
```

This will display how much of the [`src`](src/) directory is covered by tests (that are not slow) and list any lines of code not executed during testing.

Note: This omits the files mentionned in the [`.coveragerc`](.coveragerc) file.

Additionally, if you want to get a more detailed and visual representation of the coverage, you can generate an HTML coverage report:

```bash
coverage html
```

Run this command in the project's root directory. After execution, it will generate a directory named `htmlcov`. Open the `index.html` file inside this directory using your browser to view the coverage report in a user-friendly format.

### Cleaning Up

To remove any `__pycache__` directories and compiled Python files:

```bash
make clean
```
