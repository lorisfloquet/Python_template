#!/usr/bin/env python3

import os
from functools import wraps
from time import perf_counter
from termcolor import colored


def time_it(func):
    """
    Decorator to time a function. Optionally print the elapsed time in color if VERBOSE environment variable is set.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if the VERBOSE environment variable is set and compare its value
        if os.getenv('VERBOSE', '0').lower() in ['1', 'true', 'yes']:
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            elapsed_time = end_time - start_time
            
            # Print the elapsed time in cyan
            print(colored(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.", "cyan"))

            return result

        return func(*args, **kwargs)

    return wrapper


@time_it
def read_file(file_path: str) -> str:
    """
    Read the contents of a file and return it as a string.

    Args:
        file_path (str): The path of the file to read.

    Returns:
        str: The contents of the file.
    """
    with open(file_path, 'r') as f:
        return f.read()
    

def save_file(file_path: str, contents: str) -> None:
    """
    Save the contents to a file.

    Args:
        file_path (str): The path of the file to save to.
        contents (str): The contents to save to the file.
    """
    with open(file_path, 'w') as f:
        f.write(contents)
