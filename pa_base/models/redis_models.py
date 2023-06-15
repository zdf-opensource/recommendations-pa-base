# Copyright (c) 2023, ZDF.
"""
This module encapsulates redis accesses for storing and retrieving pre-calculated model scores in/from redis.
"""

from abc import abstractmethod
from contextlib import AbstractContextManager
from datetime import timedelta
from enum import Enum, unique
from itertools import chain, islice
from typing import Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

import redis

from pa_base.configuration.config import (
    REDISMODEL_HOST,
    REDISMODEL_PORT,
    REDISMODEL_TOKEN,
)
from pa_base.models.base_model import (
    KnowsItemMixin,
    PredictMixin,
    SimilarItemsMixin,
    normalize_scores,
)
from pa_base.train.user_segmentation import GenreSegmentsEnum, default_segment
from pa_base.util import combine_decayed

if not REDISMODEL_HOST or not REDISMODEL_PORT:
    raise ValueError("REDISMODEL_HOST and REDISMODEL_PORT need to be defined")


# CONFIG
# TTL (timedelta or seconds -> convert to seconds so it can be used as default arg and spares conversion on each redis EXPIRE call)
# use None to persist instead of expire
RECOMMENDATION_TTL: Optional[int] = int(timedelta(days=2).total_seconds())
AUTOMATION_TTL: Optional[int] = None
USER_SEGMENTS_TTL: Optional[int] = None
# TTL for default stage model
STAGE_RECOMMENDATION_TTL: Optional[int] = None
# TTL for stage alternative/secondary A/B testing models
STAGE_AB_RECOMMENDATION_TTL: Optional[int] = int(timedelta(days=1).total_seconds())

# number of similar items to store for each item (for each model)
KNN_IN_REDIS: int = 1000


# CONNECTIONS
_pool_args = dict(host=REDISMODEL_HOST, port=REDISMODEL_PORT, decode_responses=True)
if REDISMODEL_TOKEN:
    _pool_args["password"] = REDISMODEL_TOKEN
_POOL_DB0 = redis.ConnectionPool(db=0, **_pool_args)
_POOL_DB1 = redis.ConnectionPool(db=1, **_pool_args)
_POOL_DB2 = redis.ConnectionPool(db=2, **_pool_args)
_POOL_DB3 = redis.ConnectionPool(db=3, **_pool_args)

# redis-based pre-computed recommendation scores (DB 0)
SCORES_REDIS: redis.Redis = redis.Redis(connection_pool=_POOL_DB0)
# redis-based pre-computed automation rankings (DB 1)
AUTOMATIONS_REDIS: redis.Redis = redis.Redis(connection_pool=_POOL_DB1)

# redis-based pre-computed mapping userid->segment (DB 2)
USERS_SEGMENTS_REDIS: redis.Redis = redis.Redis(connection_pool=_POOL_DB2)

# redis-based pre-computed mapping segment->scores (DB 3)
# use ModelSuffix (stage_cmab, stage_sbm) to distinguish the keys
SEGMENTS_SCORES_REDIS: redis.Redis = redis.Redis(connection_pool=_POOL_DB3)


# HELPERS
@unique
class ModelSuffix(str, Enum):
    cbm_pctablet = ":cbm_p"
    """collaborative model"""

    cbm_mobile = ":cbm_m"
    """collaborative model"""

    cbm_tv = ":cbm_t"
    """collaborative model"""

    cbm_pctablet_mobile = ":cbm_pm"
    """collaborative model trained on 2 frontends' interactions"""

    text = ":tbm"
    """text model"""

    text_all = ":tbm_a"
    """text model variant using fsdb + txtwerk entities"""

    text_txtwerk = ":tbm_t"
    """text model variant using only txtwerk entities"""

    text_fsdb = ":tbm_f"
    """text model variant using only fsdb tags"""

    sbm_pctablet = ":sbm_p"
    """sequential model"""

    sbm_mobile = ":sbm_m"
    """sequential model"""

    sbm_tv = ":sbm_t"
    """sequential model"""

    sbm_pctablet_mobile = ":sbm_pm"
    """sequential model trained on 2 frontends' interactions"""


@unique
class AutomationKey(str, Enum):

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    #     last-chance, preview, most-viewed, top-5, trending      #
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

    lcm = "lcm"
    """last-chance model"""

    pvm = "pvm"
    """preview model"""

    mvm_pctablet = "mvm_p"
    """most-viewed model"""

    mvm_mobile = "mvm_m"
    """most-viewed model"""

    mvm_tv = "mvm_t"
    """most-viewed model"""

    t5m_pctablet = "t5m_p"
    """top-5 model"""

    t5m_mobile = "t5m_m"
    """top-5 model"""

    t5m_tv = "t5m_t"
    """top-5 model"""

    tdm_pctablet = "tdm_p"
    """trending model"""

    tdm_mobile = "tdm_m"
    """trending model"""

    tdm_tv = "tdm_t"
    """trending model"""

    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    #                   ZDFinfo topic clusters                    #
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #

    zdfinfo_history = "zdfinfo_history"
    """zdfinfo topic cluster for history & current events"""

    zdfinfo_politics = "zdfinfo_politics"
    """zdfinfo topic cluster for politics, society & foreign affairs"""

    zdfinfo_science = "zdfinfo_science"
    """zdfinfo topic cluster for science & technology"""

    zdfinfo_service = "zdfinfo_service"
    """zdfinfo topic cluster for service & consumers"""

    zdfinfo_true_crime = "zdfinfo_true_crime"
    """zdfinfo topic cluster for true crime"""


@unique
class StageModelSuffix(str, Enum):
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    #     model types used for stage recommendations              #
    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #
    # stage_sbm = ":stage_sbm"
    # """sequential model for stage recommendations"""
    # TODO changes on stage_cmabX should reflect in cluster_configs.yaml schema and in P/A service ContentPool
    stage_cmab = ":stage_cmab"
    """cmab for stage recommendations"""

    stage_cmab2 = ":stage_cmab2"
    """cmab for stage recommendations"""

    stage_cmab3 = ":stage_cmab3"
    """cmab for stage recommendations"""


_TRedisBaseModel = TypeVar("_TRedisBaseModel", bound="_RedisBaseModel")


class _RedisBaseModel(AbstractContextManager, KnowsItemMixin):
    """
    an abstract context manager for redis connections & keys for pre-computed model scores in redis

    opens a pipeline (non-transactional) in ``__enter__`` which is executed and closed in ``__exit__``
    """

    def __init__(self, *, r: redis.Redis = ...):
        # input validation
        assert r is not ..., "A redis instance needs to be provided to RedisModel!"
        # instance attributes
        self.r = r
        self.pipeline: Optional[redis.client.Pipeline] = None
        # just to fit in with other models (during prediction / service runtime)
        self.itemids = []
        self.description = self.__class__.__name__

    def __enter__(self: _TRedisBaseModel) -> _TRedisBaseModel:
        if self.pipeline is not None:
            raise RuntimeError("Already started a pipeline!")
        # do not use a transaction since this would block the whole single-threaded redis instance
        self.pipeline = self.r.pipeline(transaction=False)
        self.pipeline.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        self.pipeline.execute()
        self.pipeline.__exit__(exc_type, exc_val, exc_tb)
        self.pipeline = None

    def knows_item(self, externalid: str):
        """check whether this model 'knows' an item"""
        # EXISTS returns the number of existing keys -> 0 or 1 here
        return bool(self.r.exists(externalid))

    def _purge_redis(self, prefix="", suffix=""):
        """purge redis keys with optional prefix and suffix"""
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        for key in self.pipeline.scan_iter(f"{prefix}*{suffix}"):
            # UNLINK is similar to DEL, but works async in redis
            self.pipeline.unlink(key)

    def _list_redis_keys(self, prefix="", suffix=""):
        """NOT EXECUTED IN PIPELINE - list redis keys with optional prefix and suffix"""
        # if self.pipeline is None:
        #     raise RuntimeError("No pipeline has been started!")
        keys = []
        # do not use a pipeline here because scan_iter is not compatible with it
        for key in self.r.scan_iter(f"{prefix}*{suffix}"):
            keys.append(key)
        return keys

    @abstractmethod
    def remove_items(self, keep_items=()) -> None:
        """
        remove all items matching ``self.suffix`` which are not in keep_items

        :param keep_items: externalids of items that should be kept
        """
        raise NotImplementedError(
            "Subclasses should overwrite this if it fits their use case"
        )


class RedisModel(_RedisBaseModel, PredictMixin, SimilarItemsMixin):
    """
    a context manager for redis connections & keys for pre-computed model scores in redis

    opens a pipeline (non-transactional) in ``__enter__`` which is executed and closed in ``__exit__``

    >>> with RedisModel(suffix=ModelSuffix.text) as r:
    ...     l = r._list_redis_keys(suffix=r.suffix.value)  # just list all keys in this example
    ... type(l)
    list

    """

    def __init__(self, *, suffix: ModelSuffix, r: redis.Redis = ...):
        # input validation
        # assert suffix in {":cbm", ":tbm", ":sbm"}
        assert isinstance(suffix, ModelSuffix)
        if r is ...:
            r = SCORES_REDIS
        # instantiate parent
        _RedisBaseModel.__init__(self, r=r)
        # instance attributes
        self.suffix = suffix
        # just to fit in with other models (during prediction / service runtime)
        self.description = f"{self.__class__.__name__}: {suffix.value}"

    def remove_items(self, keep_items=()) -> None:
        """
        remove all items matching ``self.suffix`` which are not in keep_items

        :param keep_items: externalids of items that should be kept
        """
        if keep_items is None or not any(keep_items):
            self._purge_redis(suffix=self.suffix)
        else:
            if self.pipeline is None:
                raise RuntimeError("No pipeline has been started!")
            redis_keys: List[str] = self._list_redis_keys(suffix=self.suffix)
            del_keys = set(redis_keys) - {
                itemid + self.suffix.value for itemid in keep_items
            }
            if any(del_keys):
                # UNLINK is similar to DEL, but works async in redis
                self.pipeline.unlink(*del_keys)

    def add_knn_for_item(self, itemid: str, scores: Mapping[str, float]):
        """
        add nearest neighbors (model prediction) for a single item in redis

        :param itemid: externalid of this item
        :param scores: nearest neighbors for this item {neighbor_id->score, ...}
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # clear scores set before writing new ones
        self.pipeline.delete(itemid + self.suffix.value)
        if not scores:
            return
        # store only 2 decimals of each score - redis only knows byte strings, so the datatype doesn't matter
        self.pipeline.rpush(
            itemid + self.suffix.value, *[f"{k}:{v:.2f}" for k, v in scores.items()]
        )
        ttl: Union[int, timedelta] = RECOMMENDATION_TTL
        if ttl is None:
            # persist the key, i.e., remove previously set ttl
            self.pipeline.persist(itemid + self.suffix.value)
        else:
            self.pipeline.expire(itemid + self.suffix.value, ttl)

    def add_knn_for_all_items(
        self, item_score_mapping: Mapping[str, Mapping[str, float]]
    ):
        """
        add nearest neighbors (model prediction) for all items in redis

        :param item_score_mapping: a mapping of nearest neighbors for each item {itemid->{neighbor_id->score, ...}, ...}
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        self.remove_items(keep_items=item_score_mapping.keys())
        for itemid, scores in item_score_mapping.items():
            self.add_knn_for_item(itemid, scores)

    def _get_scores_for_item(self, itemid: str, k: Optional[int]):
        """NOT EXECUTED IN PIPELINE - gets ``k`` scores (predictions / nearest neighbors) for a single item"""
        return [
            (lambda x: (x[0], float(x[1])))(v.rsplit(":", maxsplit=1))
            for v in self.r.lrange(
                itemid + self.suffix.value,
                0,
                -1 if k is None else k,
            )
        ]

    def _predict_chained(
        self,
        sequence: Iterable[str],
        *,
        select: Iterable[str] = (),
        n: int = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """get predictions for sequence of interactions, where all scores are chained (without duplicates)

        :param sequence: user history from newest to oldest
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params
        :return iterable of (externalid, score) tuples
        """
        if not any(sequence):
            return []
        threshold: float
        if "threshold" in kwargs:
            threshold = kwargs["threshold"]
        else:
            threshold = 0.0
        n_offset: int = kwargs["n_offset"] if "n_offset" in kwargs else 0
        # scores is kept lazy most of the time to improve performance
        scores: Iterable[Tuple[str, float]] = iter(())
        seen_extids = set()
        for item in sequence:
            # scores.extend(self._get_scores_for_item(itemid=item, k=n - len(scores)))
            # explicitly fetch ALL scores (k=None), so we have enough after filtering against ``select``
            _scores = self._get_scores_for_item(itemid=item, k=None)
            _scores = (item for item in _scores if item[0] not in seen_extids)
            if any(select):
                _scores = (item for item in _scores if item[0] in select)
            if threshold:
                _scores = (item for item in _scores if item[1] >= threshold)
            # unroll because we need it twice -> generator would be exhausted after one iteration
            _scores = list(_scores)
            if _scores:
                scores = chain(scores, _scores)
                seen_extids.update(list(zip(*_scores))[0])
                if n is not None and len(seen_extids) >= n:
                    scores = islice(scores, n + n_offset)
                    break
        scores = islice(scores, n_offset, None)
        if scale:
            scores = normalize_scores(scores)
        return list(scores)

    def _predict_decayed(
        self,
        sequence: Iterable[str],
        *,
        select: Iterable[str] = (),
        n: int = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """get predictions for sequence of interactions, where later scores are exponentially decayed

        :param sequence: user history from newest to oldest
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params
        :return iterable of (externalid, score) tuples
        """
        if not any(sequence):
            return []
        threshold: float
        if "threshold" in kwargs:
            threshold = kwargs["threshold"]
        else:
            threshold = 0.0
        n_offset: int = kwargs["n_offset"] if "n_offset" in kwargs else 0
        # scores is kept lazy most of the time to improve performance
        sequence_scores: List[Iterable[Tuple[str, float]]] = []
        for item in sequence:
            # scores.extend(self._get_scores_for_item(itemid=item, k=n - len(scores)))
            # explicitly fetch ALL scores (k=None), so we have enough after filtering against ``select``
            _scores = self._get_scores_for_item(itemid=item, k=None)
            if any(select):
                _scores = (item for item in _scores if item[0] in select)
            if threshold:
                _scores = (item for item in _scores if item[1] >= threshold)
            # unroll because we need it twice -> generator would be exhausted after one iteration
            _scores = list(_scores)
            if _scores:
                sequence_scores.append(_scores)
        scores: Iterable[Tuple[str, float]] = combine_decayed(*sequence_scores)
        scores = islice(scores, n_offset, n + n_offset if n is not None else None)
        if scale:
            scores = normalize_scores(scores)
        return list(scores)

    def predict(
        self,
        sequence: Iterable[str],
        *,
        select: Iterable[str] = (),
        n: int = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """get predictions for sequence of interactions

        :param sequence: user history from newest to oldest
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params
        :return iterable of (externalid, score) tuples
        """
        return self._predict_decayed(
            sequence,
            select=select,
            n=n,
            scale=scale,
            **kwargs,
        )

    def similar_items(
        self,
        item: str,
        *,
        select: Iterable[str] = (),
        n: int = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """get similar items for item

        :param item: externalid of reference item
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params and ``threshold`` for scores
        :return iterable of (externalid, score) tuples
        """
        if not item:
            return []
        # explicitly fetch ALL scores (k=None), so we have enough after filtering against ``select``
        scores = self._get_scores_for_item(itemid=item, k=None)
        if any(select):
            scores = (item for item in scores if item[0] in select)
        if "threshold" in kwargs:
            scores = (item for item in scores if item[1] >= kwargs["threshold"])
        n_offset: int = kwargs["n_offset"] if "n_offset" in kwargs else 0
        scores = islice(scores, n_offset, n + n_offset if n is not None else None)
        if scale:
            scores = normalize_scores(scores)
        return list(scores)


class RedisAutomationModel(_RedisBaseModel, PredictMixin):
    """
    a context manager for redis connections & keys for pre-computed model scores in redis

    opens a pipeline (non-transactional) in ``__enter__`` which is executed and closed in ``__exit__``

    >>> with RedisAutomationModel(key=AutomationKey.tdm_pctablet) as r:
    ...     l = r._list_redis_keys(key=r.key.value)  # just list all keys in this example
    ... type(l)
    list

    """

    def __init__(self, *, key: AutomationKey, r: redis.Redis = ...):
        # input validation
        # assert key in {"lcm", "pvm", "lvm", "t5m", "tdm"}
        assert isinstance(key, AutomationKey)
        if r is ...:
            r = AUTOMATIONS_REDIS
        # instantiate parent
        _RedisBaseModel.__init__(self, r=r)
        # instance attributes
        self.key = key
        # just to fit in with other models (during prediction / service runtime)
        self.description = f"{self.__class__.__name__}: {key.value}"

    def remove_items(self, keep_items=()) -> None:
        """
        remove all items matching ``self.suffix`` which are not in keep_items

        :param keep_items: ignored for automations
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # check if the key exists in redis
        # redis_keys: List[str] = self._list_redis_keys(prefix=self.key, suffix="")
        # if self.key.value in redis_keys:
        if self.r.exists(self.key.value):
            # UNLINK is similar to DEL, but works async in redis
            self.pipeline.unlink(self.key.value)

    def add_scores(self, scores: Mapping[str, float]):
        """
        add model prediction to redis

        :param scores: scores {external_id->score, ...}
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # clear scores set before writing new ones
        self.pipeline.delete(self.key.value)
        if not scores:
            return
        # store only 2 decimals of each score - redis only knows byte strings, so the datatype doesn't matter
        self.pipeline.rpush(
            self.key.value, *[f"{k}:{v:.2f}" for k, v in scores.items()]
        )
        ttl: Union[int, timedelta] = AUTOMATION_TTL
        if ttl is None:
            # persist the key, i.e., remove previously set ttl
            self.pipeline.persist(self.key.value)
        else:
            self.pipeline.expire(self.key.value, ttl)

    def _get_scores(self, k: Optional[int]):
        """NOT EXECUTED IN PIPELINE - gets ``k`` scores"""
        return [
            (lambda x: (x[0], float(x[1])))(v.rsplit(":", maxsplit=1))
            for v in self.r.lrange(self.key.value, 0, -1 if k is None else k)
        ]

    def predict(
        self,
        sequence: Iterable[str] = (),
        *,
        select: Iterable[str] = (),
        n: int = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """get predictions for sequence of interactions

        :param sequence: user history from newest to oldest
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params and ``threshold`` for scores
        :return iterable of (externalid, score) tuples
        """
        # explicitly fetch ALL scores (k=None), so we have enough after filtering against ``select``
        scores = self._get_scores(k=None)
        if any(select):
            scores = (item for item in scores if item[0] in select)
        if "threshold" in kwargs:
            scores = (item for item in scores if item[1] >= kwargs["threshold"])
        n_offset: int = kwargs["n_offset"] if "n_offset" in kwargs else 0
        scores = islice(scores, n_offset, n + n_offset if n is not None else None)
        if scale:
            scores = normalize_scores(scores)
        return list(scores)


class RedisUserSegmentModel(_RedisBaseModel):
    """
    a context manager for redis connections & keys for pre-computed userid-segment mapping in redis

    opens a pipeline (non-transactional) in ``__enter__`` which is executed and closed in ``__exit__``

    >>> with RedisUserSegmentModel() as r:
    ...     l = r._list_redis_keys()  # just list all keys in this example
    ... type(l)
    list

    """

    def __init__(self, *, r: redis.Redis = ...):
        # input validation
        if r is ...:
            r = USERS_SEGMENTS_REDIS
        # instantiate parent
        _RedisBaseModel.__init__(self, r=r)
        # just to fit in with other models (during prediction / service runtime)
        self.description = self.__class__.__name__

    def remove_items(self, keep_items=()) -> None:
        """
        remove all items matching ``self.suffix`` which are not in keep_items

        :param keep_items: externalids of items that should be kept
        """
        if keep_items is None or not any(keep_items):
            self._purge_redis()
        else:
            if self.pipeline is None:
                raise RuntimeError("No pipeline has been started!")
            redis_keys: List[str] = self._list_redis_keys()
            del_keys = set(redis_keys) - {itemid for itemid in keep_items}
            if any(del_keys):
                # UNLINK is similar to DEL, but works async in redis
                self.pipeline.unlink(*del_keys)

    def add_segment_for_userid_hash(self, userid_hash: str, segment: GenreSegmentsEnum):
        """
        add userid to segment mapping to redis (currently saves the segment's name, e.g., 'my_uuid'->'serien')

        :param userid_hash: hash of a userid
        :param segment: segment for this user
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # clear scores set before writing new ones
        self.pipeline.delete(userid_hash)
        # if segment is None:
        #     return
        ttl: Union[int, timedelta] = USER_SEGMENTS_TTL
        if ttl is None:
            # persist the key, i.e., remove previously set ttl
            self.pipeline.set(userid_hash, segment.name, ex=None, keepttl=False)
        else:
            self.pipeline.setex(userid_hash, ttl, segment.name)

    def add_segments_for_all_userid_hashes(
        self, user_segment_mapping: Mapping[str, Union[str, GenreSegmentsEnum]]
    ):
        """
        add userid to segment mapping to redis (currently saves the segment's name, e.g., 'my_uuid'->'serien')

        :param user_segment_mapping: mapping of hash of a userid to user's segment
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # clear scores set before writing new ones
        self.remove_items(keep_items=user_segment_mapping.keys())
        # transform segments from enum to int
        user_segment_mapping: Mapping[str, str] = {
            userid_hash: (
                segment.name
                if isinstance(segment, GenreSegmentsEnum)
                else (
                    segment
                    if segment in GenreSegmentsEnum.names
                    else default_segment.name
                )
            )
            for userid_hash, segment in user_segment_mapping.items()
        }
        ttl: Union[int, timedelta] = USER_SEGMENTS_TTL
        if ttl is None:
            # persist the key, i.e., remove previously set ttl
            # mset sets multiple keys/values at once atomically
            self.pipeline.mset(user_segment_mapping)
        else:
            # mset sets multiple keys/values at once atomically but without ttl
            self.pipeline.mset(user_segment_mapping)
            for userid_hash in user_segment_mapping.keys():
                self.pipeline.expire(userid_hash, ttl)

    def get_segment_for_userid_hash(self, userid_hash: str) -> GenreSegmentsEnum:
        """
        NOT EXECUTED IN PIPELINE - gets a segment for a hashed userid

        :param userid_hash: blake2b hash with 20 bytes digest size of a userid,
            c.f., :func:`~reco.service.identity_service.get_hashed_user_id`
        """
        segment: str = self.r.get(userid_hash)
        try:
            return GenreSegmentsEnum[segment]
        except KeyError:
            # this segment does not exist in the enum
            return default_segment


class RedisStageRecommendationsModel(_RedisBaseModel):
    """
    #TODO CAREFUL: this part is duplicated / required in the cmab code
    a context manager for redis connections & keys for pre-computed model scores per segment in redis

    opens a pipeline (non-transactional) in ``__enter__`` which is executed and closed in ``__exit__``

    >>> with RedisStageRecommendationsModel(suffix=StageModelSuffix.stage_cmab) as r:
    ...     l = r._list_redis_keys(suffix=r.suffix.value)  # just list all keys in this example
    ... type(l)
    list

    """

    def __init__(self, *, suffix: StageModelSuffix, r: redis.Redis = ...):
        # input validation
        assert isinstance(suffix, StageModelSuffix)
        if r is ...:
            r = SEGMENTS_SCORES_REDIS
        # instantiate parent
        _RedisBaseModel.__init__(self, r=r)
        # instance attributes
        self.suffix = suffix
        # just to fit in with other models (during prediction / service runtime)
        self.description = f"{self.__class__.__name__}: {suffix.value}"

    def remove_items(self, keep_items=()) -> None:
        """
        remove all items matching ``self.suffix`` which are not in keep_items

        :param keep_items: externalids of items that should be kept
        """
        if keep_items is None or not any(keep_items):
            self._purge_redis(suffix=self.suffix)
        else:
            if self.pipeline is None:
                raise RuntimeError("No pipeline has been started!")
            redis_keys: List[str] = self._list_redis_keys(suffix=self.suffix)
            del_keys = set(redis_keys) - {
                itemid + self.suffix.value for itemid in keep_items
            }
            if any(del_keys):
                # UNLINK is similar to DEL, but works async in redis
                self.pipeline.unlink(*del_keys)

    def add_knn_for_item(self, itemid: str, scores: Mapping[str, float]):
        """
        add nearest neighbors (model prediction) for a single item in redis

        :param itemid: externalid of this item
        :param scores: nearest neighbors for this item {neighbor_id->score, ...}
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        # clear scores set before writing new ones
        self.pipeline.delete(itemid + self.suffix.value)
        if not scores:
            return
        # store only 2 decimals of each score - redis only knows byte strings, so the datatype doesn't matter
        self.pipeline.rpush(
            itemid + self.suffix.value, *[f"{k}:{v:.2f}" for k, v in scores.items()]
        )
        ttl: Union[int, timedelta] = (
            # use default stage model TTL for default model stage_cmab
            STAGE_RECOMMENDATION_TTL
            if self.suffix == StageModelSuffix.stage_cmab
            # use alternative stage model TTL for other stage models
            else STAGE_AB_RECOMMENDATION_TTL
        )
        if ttl is None:
            # persist the key, i.e., remove previously set ttl
            self.pipeline.persist(itemid + self.suffix.value)
        else:
            self.pipeline.expire(itemid + self.suffix.value, ttl)

    def add_knn_for_all_items(
        self, item_score_mapping: Mapping[str, Mapping[str, float]]
    ):
        """
        add nearest neighbors (model prediction) for all items in redis

        :param item_score_mapping: a mapping of nearest neighbors for each item {itemid->{neighbor_id->score, ...}, ...}
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline has been started!")
        self.remove_items(keep_items=item_score_mapping.keys())
        for itemid, scores in item_score_mapping.items():
            self.add_knn_for_item(itemid, scores)

    def get_scores_for_segment(
        self,
        segment: GenreSegmentsEnum,
        *,
        select: Iterable[str] = (),
        n: Optional[int],
        scale: bool = False,
    ):
        """NOT EXECUTED IN PIPELINE - gets ``k`` scores (predictions) for a segment"""
        scores = [
            (lambda x: (x[0], float(x[1])))(v.rsplit(":", maxsplit=1))
            for v in self.r.lrange(
                str(segment.value) + self.suffix.value,
                0,
                -1 if n is None else n,
            )
        ]
        if any(select):
            scores = (item for item in scores if item[0] in select)
        if scale:
            scores = normalize_scores(scores)
        return list(scores)
