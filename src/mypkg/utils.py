#!/usr/bin/env python3

import os
from functools import wraps
from time import perf_counter
from termcolor import colored


def is_verbose_set() -> bool:
    """
    Returns True if the VERBOSE environment variable is set, False otherwise.
    """
    return os.getenv('VERBOSE', '0').lower() in ['1', 'true', 'yes']


def time_it(func):
    """
    Decorator to time a function.
    Optionally print the elapsed time in color if the VERBOSE environment variable is set.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        # Extracting the 'verbose' argument if present, default to None if not
        verbose_arg = kwargs.get('verbose', None)

        # Determine whether to print based on the 'verbose' argument or the environment variable
        should_print = verbose_arg if isinstance(verbose_arg, bool) else is_verbose_set()
        
        if should_print:
            print(colored(f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.", "cyan"))

        return result

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
