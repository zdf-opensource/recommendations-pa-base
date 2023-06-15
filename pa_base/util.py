# Copyright (c) 2023, ZDF.
"""
Util functions for recommendations and web service.
"""

import logging
import re
from collections import defaultdict
from functools import reduce
from hashlib import blake2b  # TODO replace by blake3 (pip install blake3)?
from operator import itemgetter
from typing import (
    Any,
    ByteString,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from pa_base.configuration.dynamic_configs import CONTENT_CONFIGS
from pa_base.datastructures.params import BaseParams, RecommendationBaseParams
from pa_base.datastructures.trie import Trie

try:
    from device_detector import DeviceDetector
except ImportError as exc:
    logging.error("Could not import device-detector", exc_info=exc)


# UA and model target handling
SPECIAL_UA: Dict[str, Dict[str, str]] = {
    "python-requests/2.22.0": {"debug": "true"},
    "smarttvbackend-MIT/v3.1": {"debug": "false", "device_type": "tv"},
    "GuzzleHttp/6.3.3 curl/7.58.0 PHP/7.2.17-0ubuntu0.18.04.1": {
        "debug": "false",
        "device_type": "smartphone",
    },
}

DEFAULT_MODEL_TARGET: str = CONTENT_CONFIGS.get("DEFAULT_MODEL_TARGET")


def decay_func(time: int, quantity: int = 1, decay_constant: float = 0.25) -> float:
    """
    exponential decay:

    ``N(t) = N_0 * exp(-\\lambda * t)``

    :param time: the time step (in 1..n)
    :param quantity: the initial quantity N_0
    :param decay_constant: controls how fast scores are decayed (larger -> faster)
    """
    return quantity * np.exp(-decay_constant * time)


def combine_decayed(
    *multiple_scores: Iterable[Tuple[str, float]]
) -> List[Tuple[str, float]]:
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


def freeze(d: Any):
    """
    'Freeze' any value to prevent errors such as `TypeError: unhashable type: 'dict'`

    :param d: value that should be frozen (in order to make it hashable)
    """
    if isinstance(d, dict):
        return frozenset((key, freeze(value)) for key, value in d.items())
    elif isinstance(d, list):
        # use tuple which keeps order and duplicates of list
        return tuple(freeze(value) for value in d)
    elif isinstance(d, set):
        return frozenset(freeze(value) for value in d)
    return d


def get_model_target(appid: str, user_agent: str) -> str:
    """get the model frontend target for given appid and user agent"""
    if not appid and not user_agent:
        return DEFAULT_MODEL_TARGET
    appid = str(appid).lower()
    user_agent = str(user_agent)

    # evaluate appid
    if appid:
        if appid.startswith(
            (
                "cellular-zdf-androidtv",
                "cellular-zdf-tvos",
                "cellular-zdf-firetv",
                "mitxperts-zdf",
                "teravolt-zdf-startleiste",
            )
        ):
            # MUST be before "mobile" case, because android(tv) prefixes match partially
            return "tv"  # app_class = "native_tv" / "hbbtv"
        elif appid.startswith(
            (
                "exozet-zdf-pd",
                "exozet-tivi-pd",
                "exozet-3sat-pd",
                "cellular-zdf-pwa",
                "moccu-recopoc",  # previously "moccu-recopoc-zdf-recommendations",
            )
        ):
            return "pctablet"  # app_class = "rws" / "pwa"
        elif appid.startswith(
            (
                "cellular-zdf-android",
                "cellular-zdf-ios",
                "exozet-3sat-android",
                "exozet-3sat-ios",
                "endava-tivi3-android",
                "endava-tivi3-ios",
            )
        ):
            return "mobile"  # app_class = "native"
        else:
            logging.info(f"Unknown appId '{appid}'")

    if user_agent:
        # evaluate user_agent if app_id is ambiguous or unavailable
        # check if its a known SPECIAL_UA
        if user_agent in SPECIAL_UA:
            device_type = SPECIAL_UA[user_agent].get("device_type", "")
        else:
            try:
                device_type = DeviceDetector(user_agent).parse().device_type()
            except Exception:  # noqa
                device_type = ""
    else:
        device_type = ""

    # we have three model targets: tv, pc (including tablet) and mobile
    if device_type == "tv":  # or app_class in {"native_tv", "hbbtv"}:
        return "tv"
    elif device_type == "smartphone":  # or app_class == "native":
        return "mobile"
    elif device_type in {"tablet", "desktop"}:  # or app_class in {"rws", "pwa"}:
        return "pctablet"
    return DEFAULT_MODEL_TARGET


@overload
def hash_user_id(user_id: Union[str, ByteString]) -> str:
    ...


@overload
def hash_user_id(user_id: None) -> None:
    ...


def hash_user_id(user_id: Union[None, str, ByteString]) -> Optional[str]:
    # actually user_id: Union[None, bytes, bytearray, memoryview, array, mmap]
    """
    hash a user id for data regulation compliant usage outside this module
    :param user_id: a user id in clear text
    :return: a 20-byte hexdigest representation of the hash
    """
    if not user_id:
        return None
    if isinstance(user_id, str):
        user_id: bytes = user_id.encode("utf-8")
    return blake2b(user_id, digest_size=20).hexdigest()


@cached(
    cache=TTLCache(maxsize=128, ttl=300),  # cache for 5 minutes
    # cache key: discard non-hashable content & trie, freeze all other params to make them hashable
    key=lambda content, content_path_trie, **kwargs: hashkey(
        **{k: freeze(v) for k, v in kwargs.items()}
    ),
)
def _get_valid_items(
    *,
    content: pd.DataFrame,
    content_path_trie: Optional[Trie],
    tvservices: Collection[str],
    min_duration: int,
    accessibility_enabled: bool,
    accessibility_options: Set[str],
    tags: Collection[str],
    tags_exact_matching: bool,
    exclude_tags: Collection[str],
    exclude_tags_exact_matching: bool,
    brand_tags: Collection[str],
    brand_tags_exact_matching: bool,
    brand_exclude_tags: Collection[str],
    brand_exclude_tags_exact_matching: bool,
    types: Collection[str],
    paths: Collection[str],
    exclude_paths: Collection[str],
    genres: Collection[str],
    exclude_genres: Collection[str],
    sntw_categories: Collection[str],
    exclude_sntw_categories: Collection[str],
    exclude_docs: Collection[str],
    brands: Collection[str],
    exclude_brands: Collection[str],
    tivi_age_group: str,
    teaser_type: str,
    content_owners: Collection[str],
    exclude_content_owners: Collection[str],
    only_documentaries: bool,
) -> pd.Series:
    valid_items: pd.DataFrame = content
    # aggregate all filters in a list to be applied at the bottom
    conditions = [
        valid_items.is_visible,
    ]
    if tvservices:
        conditions.append(valid_items.tvservice.isin(tvservices))
    if min_duration:
        conditions.append(valid_items.video_duration >= min_duration)
    if accessibility_enabled and accessibility_options:
        if "ut" in accessibility_options and "has_ut" in valid_items.columns:
            conditions.append(valid_items.has_ut)
        if "ad" in accessibility_options and "has_ad" in valid_items.columns:
            conditions.append(valid_items.has_ad)
        if "dgs" in accessibility_options and "has_dgs" in valid_items.columns:
            conditions.append(valid_items.has_dgs)
    if tags and tags_exact_matching:
        conditions.append(
            ~valid_items.editorial_tags.str.split(",", expand=False).map(
                lambda item_tags: set(item_tags).isdisjoint(tags)
            )
        )
    elif tags:
        conditions.append(
            valid_items.editorial_tags.str.contains(
                "|".join(map(re.escape, tags)),
                regex=True,
                flags=re.IGNORECASE,
                na=False,
            )
        )
    if exclude_tags and exclude_tags_exact_matching:
        conditions.append(
            valid_items.editorial_tags.str.split(",", expand=False).map(
                lambda item_tags: set(item_tags).isdisjoint(exclude_tags)
            )
        )
    elif exclude_tags:
        conditions.append(
            ~valid_items.editorial_tags.str.contains(
                "|".join(map(re.escape, exclude_tags)),
                regex=True,
                flags=re.IGNORECASE,
                na=False,
            )
        )
    if brand_tags and brand_tags_exact_matching:
        valid_brands = valid_items.loc[
            (valid_items.contenttype == "brand")
            & ~valid_items.editorial_tags.str.split(",", expand=False).map(
                lambda item_tags: set(item_tags).isdisjoint(brand_tags)
            ),
            "externalid",
        ]
        conditions.append(valid_items.brand_externalid.isin(valid_brands))
    elif brand_tags:
        valid_brands = valid_items.loc[
            (valid_items.contenttype == "brand")
            & valid_items.editorial_tags.str.contains(
                "|".join(map(re.escape, brand_tags)),
                regex=True,
                flags=re.IGNORECASE,
                na=False,
            ),
            "externalid",
        ]
        conditions.append(valid_items.brand_externalid.isin(valid_brands))
    if brand_exclude_tags and brand_exclude_tags_exact_matching:
        exclude_brands = valid_items.loc[
            (valid_items.contenttype == "brand")
            & valid_items.editorial_tags.str.split(",", expand=False).map(
                lambda item_tags: set(item_tags).isdisjoint(brand_exclude_tags)
            ),
            "externalid",
        ]
        conditions.append(valid_items.brand_externalid.isin(exclude_brands))
    elif brand_exclude_tags:
        exclude_brands = valid_items.loc[
            (valid_items.contenttype == "brand")
            & ~valid_items.editorial_tags.str.contains(
                "|".join(map(re.escape, brand_exclude_tags)),
                regex=True,
                flags=re.IGNORECASE,
                na=False,
            ),
            "externalid",
        ]
        conditions.append(valid_items.brand_externalid.isin(exclude_brands))
    if types:
        conditions.append(valid_items.contenttype.isin(types))
        # if "trailer" not in params.types:
        #     conditions.append(valid_items.current_videotype != "trailer")
    if paths:
        if content_path_trie is None:
            conditions.append(valid_items.path.str.startswith(tuple(paths)))
        else:
            # trie is faster, but may not be passed, e.g., for test/mig content
            conditions.append(
                valid_items.externalid.isin(
                    [
                        extid
                        for path in paths
                        for extid in content_path_trie.startswith(path)
                    ]
                )
            )
    if exclude_paths:
        if content_path_trie is None:
            conditions.append(~valid_items.path.str.startswith(tuple(exclude_paths)))
        else:
            # trie is faster, but may not be passed, e.g., for test/mig content
            conditions.append(
                ~valid_items.externalid.isin(
                    [
                        extid
                        for path in exclude_paths
                        for extid in content_path_trie.startswith(path)
                    ]
                )
            )
    if genres:
        conditions.append(valid_items.path_level_1.isin(list(genres)))
    if exclude_genres:
        conditions.append(~valid_items.path_level_1.isin(list(exclude_genres)))
    if sntw_categories:
        conditions.append(valid_items.sntw_category.isin(list(sntw_categories)))
    if exclude_sntw_categories:
        conditions.append(
            ~valid_items.sntw_category.isin(list(exclude_sntw_categories))
        )
    # NOT CACHEABLE --> moved to get_valid_items()
    # if (
    #     isinstance(params, RecommendationBaseParams)
    #     and (not params.keep_history_docs)
    #     and params.history_doc_ids
    # ):
    #     conditions.append(~valid_items.externalid.isin(params.history_doc_ids))
    if exclude_docs:
        conditions.append(~valid_items.externalid.isin(exclude_docs))
    if brands:
        conditions.append(valid_items.brand_externalid.isin(brands))
    if exclude_brands:
        conditions.append(~valid_items.brand_externalid.isin(exclude_brands))
    if tivi_age_group:
        conditions.append(valid_items.tivi_age_group == tivi_age_group)
    if teaser_type and teaser_type != "default":
        if teaser_type == "group-stage":
            # PERAUT-1130 stageRecommendation and autoStageTeaser combined in is_stage_allowed
            conditions.append(valid_items.is_stage_allowed)
        elif teaser_type == "group-cluster-poster":
            # PERAUT-1130 flag in is_poster_allowed
            conditions.append(valid_items.is_poster_allowed)
        else:
            raise ValueError(f"Unknown teaser_type '{teaser_type}'.")
    if content_owners:
        conditions.append(valid_items.content_owner.isin(content_owners))
    if exclude_content_owners:
        conditions.append(~valid_items.content_owner.isin(exclude_content_owners))
    if only_documentaries and "is_doku" in valid_items.columns:
        # TODO remove col check once content_owner is available in all environments
        conditions.append(valid_items.is_doku)
    # combine all conditions with logical_and `&` and filter valid_items
    return valid_items.loc[reduce(np.logical_and, conditions), "externalid"]


def get_valid_items(
    *,
    params: BaseParams = None,
    content: pd.DataFrame,
    content_path_trie: Optional[Trie] = None,
    cached: bool = True,
) -> pd.Series:
    """
    get valid item selection according to any :class:`BaseParams` on content items

    :param params: any (subtype of) BaseParams
    :param content: content to draw valid items from
    :param cached: True to use cached version, False to disable caching
    :return: pd.Series of externalids (str) that may be included in the output
    """
    if params is None:
        params = BaseParams()
    if content is None:
        raise ValueError("_get_valid_items(): content must not be None")

    # get the __wrapped__ function to disable caching iff cached==False
    helper = _get_valid_items if cached else _get_valid_items.__wrapped__

    valid_items: pd.Series = helper(
        content=content,
        content_path_trie=content_path_trie,
        tvservices=params.tvservices,
        min_duration=params.min_duration,
        accessibility_enabled=params.accessibility_enabled,
        accessibility_options=params.accessibility_options,
        tags=params.tags,
        tags_exact_matching=params.tags_exact_matching,
        exclude_tags=params.exclude_tags,
        exclude_tags_exact_matching=params.exclude_tags_exact_matching,
        brand_tags=params.brand_tags,
        brand_tags_exact_matching=params.brand_tags_exact_matching,
        brand_exclude_tags=params.brand_exclude_tags,
        brand_exclude_tags_exact_matching=params.brand_exclude_tags_exact_matching,
        types=params.types,
        paths=params.paths,
        exclude_paths=params.exclude_paths,
        genres=params.genres,
        exclude_genres=params.exclude_genres,
        sntw_categories=params.sntw_categories,
        exclude_sntw_categories=params.exclude_sntw_categories,
        exclude_docs=params.exclude_docs,
        brands=params.brands,
        exclude_brands=params.exclude_brands,
        tivi_age_group=params.tivi_age_group,
        teaser_type=params.teaser_type,
        content_owners=params.content_owners,
        exclude_content_owners=params.exclude_content_owners,
        only_documentaries=params.only_documentaries,
    )

    # remove history docs here because this part would break caching on _get_valid_items()
    if (
        isinstance(params, RecommendationBaseParams)
        and (not params.keep_history_docs)
        and params.history_doc_ids
    ):
        valid_items.drop(
            index=params.history_doc_ids,  # noqa
            inplace=True,
            errors="ignore",
        )

    return valid_items
