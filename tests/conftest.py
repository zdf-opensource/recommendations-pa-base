#!/usr/bin/env python3
# Copyright (c) 2024, ZDF.
"""
Config / fixtures for all tests.
"""

import pytest
import redis
from utils import redis_delete_all_keys

from pa_base.configuration.config import (
    REDISMODEL_HOST,
    REDISMODEL_PORT,
    WITHOUT_REDIS_MODELS,
)
from pa_base.zdf.models.redis_models import (
    EVENTS_SCORES_REDIS,
    SEGMENTS_SCORES_REDIS,
    EventsModelSuffix,
    StageModelSuffix,
)

try:
    # import dotenv and read .env file BEFORE any other import
    import dotenv

    # use print because logging is not yet set up correctly
    print("Reading .env file.")
    dotenv.load_dotenv()
except ImportError:
    # dotenv probably not installed
    # use print because logging is not yet set up correctly
    print("Could not import dotenv, not reading .env file.")


class TestSetupError(Exception):
    pass


def pytest_sessionstart():
    print("Starting test session")
    if WITHOUT_REDIS_MODELS:
        print("Running without redis models")
    elif not REDISMODEL_HOST or not REDISMODEL_PORT:
        raise TestSetupError("REDISMODEL_HOST or REDISMODEL_PORT not set.")
    else:
        try:
            r = redis.Redis(host=REDISMODEL_HOST, port=REDISMODEL_PORT)
            assert r.ping() is True
        except redis.exceptions.ConnectionError:
            raise TestSetupError(f"No Redis server running on 'host={REDISMODEL_HOST}, port={REDISMODEL_PORT}'.")


@pytest.fixture
def redis_stage_setup() -> StageModelSuffix:
    r = SEGMENTS_SCORES_REDIS
    suffix = StageModelSuffix.stage_cmab
    yield suffix
    # Delete keys in redis for stage scores.
    redis_delete_all_keys(r, suffix.value)


@pytest.fixture
def redis_event_setup() -> EventsModelSuffix:
    r = EVENTS_SCORES_REDIS
    suffix = EventsModelSuffix.event_cmab
    yield suffix
    # Delete keys in redis for stage scores.
    redis_delete_all_keys(r, suffix.value)


@pytest.fixture
def reco_scores_per_segment() -> dict:
    return {
        "0": {
            "item_id_0": 0.7018513679504395,
            "item_id_1": 0.0018633523723110557,
        },
        "1": {
            "item_id_2": 0.7018513679504395,
            "item_id_3": 0.7018626928329468,
        },
        "2": {
            "item_id_4": 0.13018513679504395,
            "item_id_5": 0.0018626928329468,
        },
        "3": {
            "item_id_6": 0.8018513679504395,
            "item_id_7": 0.1018626928329468,
        },
        "4": {
            "item_id_8": 0.5018513679504395,
            "item_id_9": 0.4018626928329468,
        },
        "5": {
            "item_id_10": 0.2018513679504395,
            "item_id_11": 0.2018626928329468,
        },
        "6": {
            "item_id_12": 0.9018513679504395,
            "item_id_13": 0.6018626928329468,
        },
    }


@pytest.fixture
def reco_scores_for_segment(reco_scores_per_segment, request):
    segment_idx = request.param
    return segment_idx, reco_scores_per_segment[segment_idx]


@pytest.fixture
def reco_scores():
    return {
        "item_id_0": 0.7018513679504395,
        "item_id_1": 0.4218633523723110557,
        "item_id_2": 0.7018513679504395,
        "item_id_3": 0.6618633523723110557,
        "item_id_4": 0.1418513679504395,
        "item_id_5": 0.0018633523723110557,
    }


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
