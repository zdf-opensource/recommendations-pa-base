# Copyright (c) 2023, ZDF.
"""
Param objects for recos/automations/clusterlists.
"""

import logging

logging.warning(
    "pa_base.datastructures.params is deprecated, use pa_base.zdf.datastructures.params for ZDF-specific params instead. If you're using this in tenant code, please inform ZDF."
)

from pa_base.zdf.datastructures.params import (  # noqa: E402, F401
    AutomationParams,
    BaseParams,
    ClusterlistParams,
    CMABRecommendationParams,
    ContentClusterlistParams,
    RecommendationBaseParams,
    RecommendationParams,
)
