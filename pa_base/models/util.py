# Copyright (c) 2023, ZDF.
"""
Util functions for recommendation models.
"""
import logging
from collections import defaultdict
from operator import itemgetter
from typing import Iterable, List, Mapping, Tuple

import numpy as np


def decay_func(time: int, quantity: int = 1, decay_constant: float = 0.25) -> float:
    """
    exponential decay:

    ``N(t) = N_0 * exp(-\\lambda * t)``

    :param time: the time step (in 1..n)
    :param quantity: the initial quantity N_0
    :param decay_constant: controls how fast scores are decayed (larger -> faster)
    """
    return quantity * np.exp(-decay_constant * time)


def combine_decayed(*multiple_scores: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    combine multiple lists of scores into one list of scores using exponential decay, i.e., decaying scores farther
    to the back of scores

    :param multiple_scores: list of [lists of (externalid, score) tuples]
    :return: a list of (externalid, score) tuples
    """
    result: Mapping[str, float] = defaultdict(float)
    idx: int
    scores: Iterable[Tuple[str, float]]
    for idx, scores in enumerate(multiple_scores):
        decay = decay_func(idx)
        externalid: str
        score: float
        for externalid, score in scores:
            result[externalid] += decay * score
    return sorted(result.items(), key=itemgetter(1), reverse=True)


def get_model_target(*args, **kwargs):
    """
    stub for backwards compatibility
    """
    logging.warning(
        "pa_base.models.util.get_model_target is deprecated, use pa_base.zdf.models.util.get_model_target for ZDF-specific utils instead. If you're using this in tenant code, please inform ZDF."
    )
    from pa_base.zdf.models.util import get_model_target as zdf_get_model_target

    return zdf_get_model_target(*args, **kwargs)


def SPECIAL_UA():
    """
    stub for backwards compatibility
    """
    logging.warning(
        "pa_base.models.util.SPECIAL_UA is deprecated, use pa_base.zdf.models.util.SPECIAL_UA for ZDF-specific utils instead. If you're using this in tenant code, please inform ZDF."
    )
    from pa_base.zdf.models.util import SPECIAL_UA as ZDF_SPECIAL_UA

    return ZDF_SPECIAL_UA
