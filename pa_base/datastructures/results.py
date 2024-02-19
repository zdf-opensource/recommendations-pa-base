# Copyright (c) 2024, ZDF.
"""
This module defines dataclasses which represent recommendations, recommendation lists, clusters and cluster lists.
"""

from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Tuple, Union

from attrs import define


@define
class Scores:
    scores: List[Tuple[str, float]]
    valid_items_count: int
    recoexplain: Union[None, DefaultDict[str, Dict[str, str]]] = None

    error: Optional[str] = None


@define
class Recommendation:
    externalid: str
    score: float

    recoexplain: Optional[List[Any]] = None
    engine: Optional[str] = None

    # optional values set in _insert_promotions_into_result_set
    id: Optional[str] = None
    path: Optional[str] = None
    target: Optional[str] = None
    video: Optional[str] = None
    video_type: Optional[str] = None


@define
class RecommendationSet:
    recommendations: List[Recommendation]
    error: Optional[str] = None
    warning: Optional[str] = None
    label: Optional[str] = None
    recommendableAssetsCount: Optional[str] = None
    userSegmentId: Optional[str] = None
    reference: Optional[Dict[str, Any]] = None
    refDocId: Optional[str] = None


@define
class Cluster:
    label: str
    score: float
    recommendations: RecommendationSet
    explanation: Optional[Mapping[str, Any]] = None
    cluster_id: str = ""


@define
class ClusterSet:
    clusters: List[Cluster]
    error: Optional[str] = None
    warning: Optional[str] = None
