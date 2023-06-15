#!/usr/bin/env python3
# Copyright (c) 2023, ZDF.
"""
Config / fixtures for all tests.
"""

import pytest

try:
    # import dotenv and read .env file BEFORE any other import
    import dotenv

    # use print because logging is not yet set up correctly
    print("Reading .env file.")
    dotenv.load_dotenv()
except ImportError:
    # dotenv probably not installed
    # use print because logging is not yet set up correctly
    print("Could not import dotenv, not reading .env file.")


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
