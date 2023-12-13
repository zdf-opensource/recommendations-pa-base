# Copyright (c) 2023, ZDF.
"""
stub for backwards compatibility
"""
import logging

logging.warning(
    "pa_base.preprocess.data_preparation is deprecated, use pa_base.zdf.preprocess.data_preparation for ZDF-specific data preparation instead. If you're using this in tenant code, please inform ZDF."
)

from pa_base.zdf.preprocess.data_preparation import *  # noqa: E402, F401, F403
