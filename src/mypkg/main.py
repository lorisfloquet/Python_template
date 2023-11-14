#!/usr/bin/env python3

import os
import argparse
from icecream import ic

import mypkg.config as config
from mypkg.utils import read_file


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Set the VERBOSE environment variable based on the command-line argument
    if args.verbose:
        os.environ['VERBOSE'] = '1'
    else:
        os.environ['VERBOSE'] = '0'

    hello_world = read_file(config.HELLO_WORLD_PATH)
    ic(hello_world)
    

if __name__ == "__main__":
    main()
