#!/usr/bin/env python3
# Copyright (c) 2023, ZDF.
"""
tests for pa_base.data.dataframes module
"""

import pandas as pd
import pytest

from pa_base.data import dataframes as dfs


def test_content_df():
    content = dfs.get_content_df()
    assert isinstance(content, pd.DataFrame)
    assert len(content) > 100


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
