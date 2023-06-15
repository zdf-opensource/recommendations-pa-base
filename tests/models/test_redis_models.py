#!/usr/bin/env python3
# Copyright (c) 2023, ZDF.
"""
Unit tests for redis_models.py
"""

import pytest
from pytest_mock import MockFixture


@pytest.fixture
def redis_model(mocker: MockFixture):
    mocker.patch("pa_base.configuration.config.REDISMODEL_HOST", "something")
    mocker.patch("pa_base.configuration.config.REDISMODEL_PORT", "something")
    # import after mocking away config
    from pa_base.models.redis_models import ModelSuffix, RedisModel

    rm = RedisModel(suffix=ModelSuffix.cbm_pctablet, r=mocker.MagicMock())
    mocker.patch.object(
        rm,
        "_get_scores_for_item",
        lambda itemid, k: {
            "a": [("b", 1.0), ("c", 0.1)],
            "b": [("a", 0.9), ("c", 0.2), ("d", 0.7)],
            "c": [],
        }.get(itemid),
    )
    return rm


def test_reco_predict(redis_model):
    scores = redis_model.predict(("a", "b"), select=("c", "d"))
    assert any(scores)
    assert isinstance(scores, list)
    assert isinstance(scores[0], tuple)
    assert isinstance(scores[0][0], str)
    assert isinstance(scores[0][1], float)
    scores_decayed = redis_model._predict_decayed(("a", "b"), select=("c", "d"))
    assert scores == scores_decayed
    scores_chained = redis_model._predict_chained(("a", "b"), select=("c", "d"))
    assert scores != scores_chained


def test_reco_predict_decayed(redis_model):
    scores = redis_model.predict(("a", "b"), select=("c", "d"))
    # only c and d left in scores
    assert len(scores) == 2
    assert scores[0][0] == "d"
    assert scores[1][0] == "c"
    # check again with different select
    scores = redis_model.predict(("a", "b"), select=("c", "e"))
    # only c and d left in scores
    assert len(scores) == 1
    assert scores[0][0] == "c"


def test_reco_predict_chained(redis_model):
    scores = redis_model.predict(("a", "b"), select=("c", "d"))
    # only c and d left in scores
    assert len(scores) == 2
    assert scores[0][0] == "d"
    assert scores[1][0] == "c"
    # check again with different select
    scores = redis_model.predict(("a", "b"), select=("c", "e"))
    # only c and d left in scores
    assert len(scores) == 1
    assert scores[0][0] == "c"


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
