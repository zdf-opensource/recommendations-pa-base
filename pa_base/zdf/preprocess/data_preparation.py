# Copyright (c) 2024, ZDF.
"""
Helper functions for content preprocessing.
"""

import pandas as pd


def _extract_contenttype_from_content(
    content: pd.DataFrame,
    contenttype: str,
    index_col: str = "externalid",
) -> pd.DataFrame:
    df: pd.DataFrame = content[content.contenttype == contenttype].copy()
    df.drop_duplicates(subset=index_col, inplace=True)
    # convert categorical column labels to normal index (otherwise at[] won't work)
    df.set_index(df[index_col].astype(str), drop=False, inplace=True)
    return df


def extract_brands_from_content(content: pd.DataFrame) -> pd.DataFrame:
    return _extract_contenttype_from_content(content, "brand")


def extract_topics_from_content(content: pd.DataFrame) -> pd.DataFrame:
    return _extract_contenttype_from_content(content, "topic")
