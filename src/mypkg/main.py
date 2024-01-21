#!/usr/bin/env python3

from icecream import ic

import mypkg.config as config
from mypkg.utils import read_file, set_venv_verbose


def main():
    # Set the VERBOSE environment variable based on the command-line argument
    set_venv_verbose()

    hello_world = read_file(config.HELLO_WORLD_PATH)
    ic(hello_world)
    

if __name__ == "__main__":
    main()
