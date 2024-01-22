#!/usr/bin/env python3
# Copyright (c) 2024, ZDF.
"""
Unit tests for redis_models.py
"""

import pytest
from pytest_mock import MockFixture
from utils import redis_delete_all_keys

from pa_base.zdf.models.redis_models import (
    SEGMENTS_SCORES_REDIS,
    EventsModelSuffix,
    RedisEventRecommendationsModel,
    RedisStageRecommendationsModel,
    StageModelSuffix,
)
from pa_base.zdf.train.user_segmentation import GenreSegmentsEnum


@pytest.fixture
def redis_model(mocker: MockFixture):
    mocker.patch("pa_base.configuration.config.REDISMODEL_HOST", "something")
    mocker.patch("pa_base.configuration.config.REDISMODEL_PORT", "something")
    # import after mocking away config
    from pa_base.zdf.models.redis_models import ModelSuffix, RedisModel

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


class TestRedisStageRecommendationsModel:
    def test_init(self):
        # GIVEN: a redis stage suffix
        suffix = StageModelSuffix.stage_cmab
        # WHEN: we instantiate the stage redis model
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            # THEN: a redis connection should exist
            assert stage_cmab.r.ping() is True

    def test_init_fail(self):
        # GIVEN: a redis event suffix
        suffix = EventsModelSuffix.event_cmab
        # WHEN: we instantiate the stage redis model
        with pytest.raises(TypeError):
            # THEN: We expect a TypeError to be raised, since the
            # redis event model suffix is incompatible with the redis stage model
            RedisStageRecommendationsModel(suffix=suffix)

    @pytest.mark.parametrize(
        "reco_scores_for_segment, expected_scores",
        [
            ("0", [("item_id_0", 0.7), ("item_id_1", 0.0)]),
            ("1", [("item_id_2", 0.7), ("item_id_3", 0.7)]),
            ("2", [("item_id_4", 0.13), ("item_id_5", 0.0)]),
            ("3", [("item_id_6", 0.8), ("item_id_7", 0.1)]),
            ("4", [("item_id_8", 0.5), ("item_id_9", 0.4)]),
            ("5", [("item_id_10", 0.2), ("item_id_11", 0.2)]),
            ("6", [("item_id_12", 0.9), ("item_id_13", 0.6)]),
        ],
        indirect=["reco_scores_for_segment"],
    )
    def test_add_knn_for_item(self, redis_stage_setup, reco_scores_for_segment, expected_scores):
        # GIVEN: reco scores for a segment
        # AND: the redis suffix
        # AND: the segment
        # AND: the expected score to be found in redis
        segment_idx, reco_scores = reco_scores_for_segment
        suffix = redis_stage_setup
        segment = list(GenreSegmentsEnum)[int(segment_idx)]
        # WHEN: we write the scores for a segment to redis
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.add_knn_for_item(str(segment.value), reco_scores)
        # THEN: we expect the scores retrieved from redis to be equal to the expected scores
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            scores = stage_cmab.get_scores_for_segment(segment)
            assert scores == expected_scores

    def test_add_knn_for_all_items(self, redis_stage_setup, reco_scores_per_segment):
        # GIVEN: Reco scores for all segments
        # AND: the redis suffix
        # AND: the expected scores for each segment to be found in redis
        expected_scores_per_segment = [
            [("item_id_0", 0.7), ("item_id_1", 0.0)],
            [("item_id_2", 0.7), ("item_id_3", 0.7)],
            [("item_id_4", 0.13), ("item_id_5", 0.0)],
            [("item_id_6", 0.8), ("item_id_7", 0.1)],
            [("item_id_8", 0.5), ("item_id_9", 0.4)],
            [("item_id_10", 0.2), ("item_id_11", 0.2)],
            [("item_id_12", 0.9), ("item_id_13", 0.6)],
        ]
        suffix = redis_stage_setup
        # WHEN: we write the scores for all segments to redis
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.add_knn_for_all_items(reco_scores_per_segment)
        # THEN: we expect the scores retrieved from redis to be equal to the expected scores
        for segment, expected_scores in zip(GenreSegmentsEnum, expected_scores_per_segment):
            with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
                scores = stage_cmab.get_scores_for_segment(segment)
                assert scores == expected_scores

    @pytest.mark.parametrize(
        "select, n, expected_scores",
        [
            ((), None, [("item_id_0", 0.7), ("item_id_1", 0.0)]),
            (("item_id_0"), None, [("item_id_0", 0.7)]),
            ((), 1, [("item_id_0", 0.7), ("item_id_1", 0.0)]),
            ((), 0, [("item_id_0", 0.7)]),
            (("item_id_1"), 1, [("item_id_1", 0.0)]),
        ],
    )
    def test_get_scores(self, redis_stage_setup, reco_scores_per_segment, select, n, expected_scores):
        # GIVEN: Reco scores for all segments
        # AND: the redis suffix
        # AND: the segment (i.e segment_0)
        # AND: the items to select
        # AND: the number of items to retrieve from redis
        # AND: the expected scores
        suffix = redis_stage_setup
        segment = list(GenreSegmentsEnum)[0]
        # WHEN: we write the scores for segment_0 to redis
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.add_knn_for_all_items(reco_scores_per_segment)
        # THEN: we expect the scores retrieved from redis for segment_0 to be equal to the expected scores for segment_0
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            scores = stage_cmab.get_scores_for_segment(segment, select=select, n=n)
            assert scores == expected_scores

    def test_remove_items(self, reco_scores_per_segment):
        # GIVEN: Reco scores for all segments
        # AND: the redis suffix
        suffix = StageModelSuffix.stage_cmab
        # WHEN: we write the scores for all segment to redis
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.add_knn_for_all_items(reco_scores_per_segment)
        # AND: we remove the keys/score from redis for all segments
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.remove_items()
        # THEN: we expect the scores for each segment to be empty when retrieving them from redis
        for segment in GenreSegmentsEnum:
            with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
                scores = stage_cmab.get_scores_for_segment(segment)
                assert scores == []

    def test_remove_keep_items(self, reco_scores_per_segment):
        # GIVEN: Reco scores for all segments
        # AND: the redis suffix
        # AND: the segment items to keep
        # AND: the expected scores to be found in redis
        suffix = StageModelSuffix.stage_cmab
        keep_items = ["0", "6"]
        expected_kept_scores_per_segment = {
            "0": [("item_id_0", 0.7), ("item_id_1", 0.0)],
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [("item_id_12", 0.9), ("item_id_13", 0.6)],
        }
        # WHEN: we write the scores for all segment to redis
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.add_knn_for_all_items(reco_scores_per_segment)
        # AND: we remove the keys/scores from redis for thos items we do not want to keep
        with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
            stage_cmab.remove_items(keep_items=keep_items)
        # THEN: we expect the "kept" segment scores to be available in redis
        scores_per_segment = {}
        for segment in GenreSegmentsEnum:
            with RedisStageRecommendationsModel(suffix=suffix) as stage_cmab:
                scores = stage_cmab.get_scores_for_segment(segment)
                scores_per_segment[str(segment.value)] = scores
        assert scores_per_segment == expected_kept_scores_per_segment

        # Delete keys in redis.
        redis_delete_all_keys(SEGMENTS_SCORES_REDIS, suffix.value)


class TestRedisEventRecommendationsModel:
    def test_init(self):
        # GIVEN: a redis event suffix
        suffix = EventsModelSuffix.event_cmab
        # WHEN: we instantiate the event redis model
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            # THEN: a redis connection should exist
            assert event_cmab.r.ping() is True

    def test_init_fail(self):
        # GIVEN: a redis stage suffix
        suffix = StageModelSuffix.stage_cmab
        # WHEN: we instantiate the event redis model
        with pytest.raises(TypeError):
            # THEN: We expect a TypeError to be raised, since the
            # redis event model suffix is incompatible with the redis stage model
            RedisEventRecommendationsModel(suffix=suffix)

    def test_add_scores(self, redis_event_setup, reco_scores):
        # GIVEN: a redis event suffix
        # AND: the expected scores
        expected_scores = [
            ("item_id_0", 0.7),
            ("item_id_1", 0.42),
            ("item_id_2", 0.7),
            ("item_id_3", 0.66),
            ("item_id_4", 0.14),
            ("item_id_5", 0.0),
        ]
        suffix = redis_event_setup
        # WHEN: we write the scores for to redis
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            event_cmab.add_scores(reco_scores)
        # THEN: we expect the scores retrieved from redis to be equal to the expected scores
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            scores = event_cmab.get_scores()
            assert scores == expected_scores

    @pytest.mark.parametrize(
        "select, n, expected_scores",
        [
            (
                (),
                None,
                [
                    ("item_id_0", 0.7),
                    ("item_id_1", 0.42),
                    ("item_id_2", 0.7),
                    ("item_id_3", 0.66),
                    ("item_id_4", 0.14),
                    ("item_id_5", 0.0),
                ],
            ),
            (
                ("item_id_0", "item_id_5"),
                None,
                [
                    ("item_id_0", 0.7),
                    ("item_id_5", 0.0),
                ],
            ),
            (
                (),
                3,
                [
                    ("item_id_0", 0.7),
                    ("item_id_1", 0.42),
                    ("item_id_2", 0.7),
                    ("item_id_3", 0.66),
                ],
            ),
            (
                ("item_id_3"),
                3,
                [
                    ("item_id_3", 0.66),
                ],
            ),
        ],
    )
    def test_get_scores(self, redis_event_setup, reco_scores, select, n, expected_scores):
        # GIVEN: Reco scores
        # AND: the redis suffix
        # AND: the items to select
        # AND: the number of items to retrieve from redis
        # AND: the expected scores
        suffix = redis_event_setup
        # WHEN: we write the scores for to redis
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            event_cmab.add_scores(reco_scores)
        # THEN: we expect the scores retrieved from redis for to be equal to the expected scores
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            scores = event_cmab.get_scores(select=select, n=n)
            assert scores == expected_scores

    def test_remove_items(self, reco_scores):
        # GIVEN: Reco scores
        # AND: the redis suffix
        suffix = EventsModelSuffix.event_cmab
        # WHEN: we write the scores to redis
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            event_cmab.add_scores(reco_scores)
        # AND: we remove the keys/score from redis
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            event_cmab.remove_items()
        # THEN: we expect the scores to be empty when retrieving them from redis
        with RedisEventRecommendationsModel(suffix=suffix) as event_cmab:
            scores = event_cmab.get_scores()
            assert scores == []


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
