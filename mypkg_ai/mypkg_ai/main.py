#!/usr/bin/env python3

from icecream import ic

import mypkg_ai.config as config
from mypkg_ai.utils import set_venv_verbose


def main():
    # Set the VERBOSE environment variable based on the command-line argument
    set_venv_verbose()
    

if __name__ == "__main__":
    main()
