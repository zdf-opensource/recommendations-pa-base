# Copyright (c) 2023, ZDF.
"""
Functions to conveniently load dataframes.
"""
import logging

logging.warning(
    "pa_base.data.dataframes is deprecated, use pa_base.zdf.dataframes for ZDF-specific dataframes instead. If you're using this in tenant code, please inform ZDF."
)

# import functions for backwards compatibility here
from pa_base.zdf.data.dataframes import (  # noqa: E402, F401
    get_agg_cl_data_df,
    get_agg_cl_short_term_most_viewed_df,
    get_captions_df,
    get_content_df,
    get_denoised_data,
    get_fsdb_matches_df,
    get_interactions_df,
    get_recos_data,
    get_teravolt_metadata_df,
)
