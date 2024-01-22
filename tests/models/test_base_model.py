#!/usr/bin/env python3
# Copyright (c) 2024, ZDF.
"""
Unit tests for models/base_model.py

(c) ZDF
"""

from itertools import islice
from typing import Any, List, Tuple

import pytest

from pa_base.models.base_model import normalize_scores


def test_base_model_normalize_scores_none():
    assert normalize_scores(None) == []  # noqa


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([], id="empty"),
        pytest.param([("a", 200.0)], id="single"),
        pytest.param([("a", 200.0), ("b", 150.0)], id="two"),
        pytest.param(list(zip(range(100), range(100, 1, -1))), id="many"),
    ],
)
def test_base_model_normalize_scores_list(scores: List[Tuple[Any, float]]):
    normalized = normalize_scores(scores)
    assert len(scores) == len(normalized)
    for s, n in zip(scores, normalized):
        assert s[0] == n[0]  # id should not change
        assert 0.0 <= n[1] <= 1.0  # score should be normalized


@pytest.mark.parametrize(
    "scores",
    [
        pytest.param([], id="empty"),
        pytest.param([("a", 200.0)], id="single"),
        pytest.param([("a", 200.0), ("b", 150.0)], id="two"),
        pytest.param(list(zip(range(100), range(100, 1, -1))), id="many"),
    ],
)
def test_base_model_normalize_scores_islice(scores: List[Tuple[Any, float]]):
    normalized = normalize_scores(islice(scores, None))
    assert len(scores) == len(normalized)
    for s, n in zip(scores, normalized):
        assert s[0] == n[0]  # id should not change
        assert 0.0 <= n[1] <= 1.0  # score should be normalized


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
