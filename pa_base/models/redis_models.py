# Copyright (c) 2023, ZDF.
"""
stub for backwards compatibility
"""
import logging

logging.warning(
    "pa_base.models.redis_models is deprecated, use pa_base.zdf.models.redis_models for ZDF-specific RedisModels instead. If you're using this in tenant code, please inform ZDF."
)

from pa_base.zdf.models.redis_models import *  # noqa: E402, F401, F403
