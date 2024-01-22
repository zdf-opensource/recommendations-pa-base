# Copyright (c) 2024, ZDF.
"""
Helper functions for content preprocessing.
"""

import pandas as pd


def extract_brands_from_content(content: pd.DataFrame) -> pd.DataFrame:
    brands: pd.DataFrame = content[content.contenttype == "brand"].copy()
    brands.drop_duplicates(subset="brand_externalid", inplace=True)
    # convert categorical column labels to normal index (otherwise at[] won't work)
    brands.set_index(brands["brand_externalid"].astype(str), drop=False, inplace=True)
    return brands


def extract_topics_from_content(content: pd.DataFrame) -> pd.DataFrame:
    topics: pd.DataFrame = content[content.contenttype == "topic"].copy()
    topics.drop_duplicates(subset="externalid", inplace=True)
    # convert categorical column labels to normal index (otherwise at[] won't work)
    topics.set_index(topics["externalid"].astype(str), drop=False, inplace=True)
    return topics
