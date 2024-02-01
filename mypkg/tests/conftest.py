#!/usr/bin/env python3

import pytest

import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for the test case"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Provide the temporary directory to the test case
    yield temp_dir

    # Remove the temporary directory after the test case is done
    shutil.rmtree(temp_dir)
