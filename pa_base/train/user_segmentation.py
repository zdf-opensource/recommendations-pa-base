# Copyright (c) 2023, ZDF.
"""
stub for backwards compatibility
"""
import logging

logging.warning(
    "pa_base.train.user_segmentation is deprecated, use pa_base.zdf.train.user_segmentation for ZDF-specific segmentation instead. If you're using this in tenant code, please inform ZDF."
)

from pa_base.zdf.train.user_segmentation import *  # noqa: E402, F401, F403
