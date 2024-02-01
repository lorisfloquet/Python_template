#!/usr/bin/env python3

import pytest

import os
import sys
from io import StringIO
from time import sleep, perf_counter
from typing import Optional
from unittest.mock import patch

from mypkg.utils import (
    set_venv_verbose,
    is_verbose_set,
    time_it,
)


########################################################################################################################
# Test the set_venv_verbose function
########################################################################################################################

def test_set_venv_verbose_enabled():
    with patch('sys.argv', ['script_name', '-v']):
        with patch.dict(os.environ, {}, clear=True):
            set_venv_verbose()
            assert os.environ['VERBOSE'] == '1'

def test_set_venv_verbose_disabled():
    with patch('sys.argv', ['script_name']):
        with patch.dict(os.environ, {}, clear=True):
            set_venv_verbose()
            assert os.environ['VERBOSE'] == '0'


########################################################################################################################
# Test the is_verbose_set function
########################################################################################################################

def test_is_verbose_set_true(monkeypatch):
    monkeypatch.setenv("VERBOSE", "1")
    assert is_verbose_set() == True

def test_is_verbose_set_false(monkeypatch):
    monkeypatch.setenv("VERBOSE", "0")
    assert is_verbose_set() == False

def test_is_verbose_set_true_string(monkeypatch):
    monkeypatch.setenv("VERBOSE", "true")
    assert is_verbose_set() == True

def test_is_verbose_set_false_string(monkeypatch):
    monkeypatch.setenv("VERBOSE", "no")
    assert is_verbose_set() == False

def test_is_verbose_set_non_existent(monkeypatch):
    monkeypatch.delenv("VERBOSE", raising=False)
    assert is_verbose_set() == False


########################################################################################################################
# Test the time_it decorator
########################################################################################################################

@time_it
def dummy_func_to_time(seconds: float, verbose: Optional[bool] = None):
    """A dummy function to use for testing the decorator"""
    sleep(seconds)
    return seconds


def test_time_it_no_errors_without_verbose():
    """Test that the function is correctly decorated and runs without errors without verbose output"""
    # Set VERBOSE environment variable for this test
    os.environ['VERBOSE'] = '0'

    elapsed_time = 1 / 1000  # 1 millisecond

    start_time = perf_counter()
    result = dummy_func_to_time(elapsed_time)
    end_time = perf_counter()

    assert result == elapsed_time
    assert end_time - start_time >= elapsed_time  # At least 1 millisecond due to sleep


def test_time_it_with_verbose():
    """Test that the function prints out the timing with verbose output"""
    # Set VERBOSE environment variable for this test
    os.environ['VERBOSE'] = '1'

    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    elapsed_time = 1 / 1000  # 1 millisecond

    # Run the decorated function
    result = dummy_func_to_time(elapsed_time)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Reset VERBOSE environment variable to not interfere with other tests
    del os.environ['VERBOSE']

    output = captured_output.getvalue().strip()
    assert "dummy_func_to_time" in output
    assert "seconds to execute" in output
    assert result == elapsed_time

    # The time printed is supposed to be before "seconds to execute" and after "took "
    time_printed = float(output.split("seconds to execute")[0].split("took ")[1])
    assert time_printed >= result # At least 1 millisecond due to sleep


def test_time_it_silent_with_verbose_not_being_set():
    """Test that the function does not print out the timing without verbose output"""
    # Ensure VERBOSE environment variable is not set
    if 'VERBOSE' in os.environ:
        del os.environ['VERBOSE']

    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    elapsed_time = 1 / 1000  # 1 millisecond

    # Run the decorated function
    result = dummy_func_to_time(elapsed_time)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue().strip()
    assert output == ""  # No output should be present
    assert result == elapsed_time


def test_time_it_with_explicit_verbose_true():
    """Test that the function prints out the timing when verbose argument is explicitly set to True"""
    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    elapsed_time = 1 / 1000  # 1 millisecond

    # Run the decorated function with verbose=True
    result = dummy_func_to_time(elapsed_time, verbose=True)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue().strip()
    assert "dummy_func_to_time" in output
    assert "seconds to execute" in output
    assert result == elapsed_time

    # Check if the timing is correctly printed
    time_printed = float(output.split("seconds to execute")[0].split("took ")[1])
    assert time_printed >= result


def test_time_it_with_explicit_verbose_false():
    """Test that the function does not print out the timing when verbose argument is explicitly set to False"""
    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    elapsed_time = 1 / 1000  # 1 millisecond

    # Run the decorated function with verbose=False
    result = dummy_func_to_time(elapsed_time, verbose=False)

    # Reset stdout
    sys.stdout = sys.__stdout__

    output = captured_output.getvalue().strip()
    assert output == ""  # No output should be present
    assert result == elapsed_time


def test_time_it_with_verbose_none():
    """Test that the function follows the VERBOSE environment variable when verbose argument is None"""
    # Set VERBOSE environment variable for this test
    os.environ['VERBOSE'] = '1'

    # Redirect stdout to capture print statements
    captured_output = StringIO()
    sys.stdout = captured_output

    elapsed_time = 1 / 1000  # 1 millisecond

    # Run the decorated function with verbose=None
    result = dummy_func_to_time(elapsed_time, verbose=None)

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Reset VERBOSE environment variable to not interfere with other tests
    del os.environ['VERBOSE']

    output = captured_output.getvalue().strip()
    assert "dummy_func_to_time" in output
    assert "seconds to execute" in output
    assert result == elapsed_time

    # Check if the timing is correctly printed
    time_printed = float(output.split("seconds to execute")[0].split("took ")[1])
    assert time_printed >= result
