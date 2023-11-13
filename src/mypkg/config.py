#!/usr/bin/env python3

import os
from termcolor import colored


# The root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# The directory containing the source code
SRC_DIR = os.path.join(ROOT_DIR, "src")

# The directory containing the data
DATA_DIR = os.path.join(ROOT_DIR, "data")

HELLO_WORLD_PATH = os.path.join(DATA_DIR, "hello_world.txt")


def check_config_in_gitignore():
    """
    Check if the config.py file is in the .gitignore file. If not, raise an exception.
    You can delete this function and its call if you want when the project is setup.
    """
    gitignore_path = os.path.join(ROOT_DIR, ".gitignore")
    with open(gitignore_path, "r") as f:
        gitignore = f.read()

    if "# src/config.py" in gitignore:
        print()
        raise RuntimeError(
            colored("Please uncomment the line '# config.py' in the .gitignore file to setup the project.", "red"))
    
check_config_in_gitignore()
