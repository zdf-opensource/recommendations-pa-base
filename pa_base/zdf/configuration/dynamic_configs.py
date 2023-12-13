# Copyright (c) 2023, ZDF.
"""
ZDF-specific dynamic configuration to be loaded at runtime.
"""

import logging
import re
from typing import Any, Container, Dict, List, Mapping, Pattern, Set, Tuple, Union

from pa_base.configuration.config import CONFIG_BUCKET
from pa_base.configuration.util import (
    async_exec,
    download_config,
    download_if_exists,
    download_json_if_exists,
)

# download dynamic configs from S3
CLUSTER_CONFIGS: Dict[str, Any]
CLUSTERLIST_CONFIGS: Dict[str, Any]
CONTENT_CONFIGS: Dict[str, Any]
API_TOKEN_CONFIGS: Dict[str, Any]
PUBSUB_SA: Dict[str, Any]
try:
    # if os.environ.get("NO_DYNAMIC_CONFIG").lower() != "true":
    (
        CLUSTER_CONFIGS,
        CLUSTERLIST_CONFIGS,
        CONTENT_CONFIGS,
        MODEL_CONFIGS,
        API_TOKEN_CONFIGS,
        PUBSUB_SA,
    ) = async_exec(
        (download_config, "cluster_configs.yml"),
        (download_config, "clusterlist_configs.yml"),
        (download_config, "content_configs.yml"),
        (download_config, "model_configs.yml"),
        (download_if_exists, "api_token_st.yml"),
        (download_json_if_exists, "pa-pubsub-publisher-sa.json"),
    )
    # else:
    # logging.warning("Not loading dynamic configs since NO_DYNAMIC_CONFIG=='true'.")
except Exception as exc:
    logging.error(
        f"Could not load dynamic configs from '{CONFIG_BUCKET}'.",
        exc_info=exc,
    )
    if CONFIG_BUCKET.startswith("de.zdf"):
        # dynamic configs are absolutely required for ZDF services, crash here if not provided
        raise exc
    logging.info("Could not load dynamic configs, continuing without.")
    CLUSTER_CONFIGS = {}
    CLUSTERLIST_CONFIGS = {}
    CONTENT_CONFIGS = {}
    MODEL_CONFIGS = {}
    API_TOKEN_CONFIGS = {}
    PUBSUB_SA = {}


# text model variants are needed in modelfactory and service
TEXT_MODEL_VARIANTS: Dict[str, List[str]] = CONTENT_CONFIGS.get("TEXT_MODEL_VARIANTS", {})

SENTENCE_TRANSFORMER_VARIANTS: List[Dict[str, str]] = CONTENT_CONFIGS.get("SENTENCE_TRANSFORMER_VARIANTS", [])

# collab model variants are needed in modelfactory and service
COLLAB_MODEL_VARIANTS: List[str] = CONTENT_CONFIGS.get("COLLAB_MODEL_VARIANTS", [])

# ZDFinfo topic clusters
ZDFINFO_TOPIC_CLUSTER_VARIANTS: Dict[str, Dict[str, List[str]]] = CONTENT_CONFIGS.get(
    "ZDFINFO_TOPIC_CLUSTER_VARIANTS",
    {},
)


# ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# #####                 Service config                  #####
# ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# reload content every X seconds (-1 disables reloading)
CONTENT_RELOAD_INTERVAL_SECONDS: int = CONTENT_CONFIGS.get("CONTENT_RELOAD_INTERVAL_SECONDS", -1)
# quicken / delay above interval by x seconds to spread load of multiple instances over time
CONTENT_RELOAD_JITTER_SECONDS: int = CONTENT_CONFIGS.get("CONTENT_RELOAD_JITTER_SECONDS", 0)

# FNV: activate or deactivate fast next-video completely
FAST_NEXT_VIDEO_ACTIVATED: bool = CONTENT_CONFIGS.get("FAST_NEXT_VIDEO_ACTIVATED", False)

# REDISMODEL: activate or deactivate usage of pre-computed scores from redis instead of loading models
REDISMODEL_AUTOMATIONS_ACTIVATED: bool = CONTENT_CONFIGS.get("REDISMODEL_AUTOMATIONS_ACTIVATED", False)
REDISMODEL_NEXT_VIDEO_ACTIVATED: bool = CONTENT_CONFIGS.get("REDISMODEL_NEXT_VIDEO_ACTIVATED", False)
REDISMODEL_SIMILAR_ITEMS_ACTIVATED: bool = CONTENT_CONFIGS.get("REDISMODEL_SIMILAR_ITEMS_ACTIVATED", False)
# list of mappings of { model: (sbm|cbm|text), variants: (pctablet|mobile|tv|text,text_fsdb|...), suffix: ModelSuffix }
REDISMODEL_RECO_MODELS: List[Dict[str, Union[str, List[str], bool]]] = CONTENT_CONFIGS.get("REDISMODEL_RECO_MODELS", [])

# ratio (percentage) of recos published to PubSub (clipped to [0.0, 1.0])
PUBSUB_RATIO: int = min(
    max(
        CONTENT_CONFIGS.get("PUBSUB_RATIO", 0.0),
        0.0,
    ),
    1.0,
)

# allowed / known A/B groups
ABGROUPS: List[str] = CONTENT_CONFIGS.get("ABGROUPS")

# index pages (dict of externalid -> human-readable name / id)
INDEX_PAGES: Dict[str, str] = CONTENT_CONFIGS.get("INDEX_PAGES")

# mix short-term API token into MTR3_ENV of content configs
MTR3_ENV: Dict[str, Any] = CONTENT_CONFIGS.get("MTR3_ENV")
try:
    for zdf_env_name, api_config in (API_TOKEN_CONFIGS or {}).get("MTR3_ENV", {}).items():
        MTR3_ENV[zdf_env_name].update(api_config)
except AttributeError as exc:
    # failed loading API_TOKEN_CONFIGS
    logging.error(
        f"Could not load API_TOKEN_CONFIGS from '{CONFIG_BUCKET}'.",
        exc_info=exc,
    )

DEFAULT_COVERAGE: float = CONTENT_CONFIGS.get("DEFAULT_COVERAGE")

# recos / clusters query params / payload
QUERY_PLAYS_MIN_PROGRESS_SECONDS: int = CONTENT_CONFIGS.get("QUERY_PLAYS_MIN_PROGRESS_SECONDS", 0)

# myprogram
MAX_PLAYBACK_PROGRESS_THRESHOLD: float = CONTENT_CONFIGS.get("MAX_PLAYBACK_PROGRESS_THRESHOLD")
MIN_PLAYBACK_THRESHOLD: float = CONTENT_CONFIGS.get("MIN_PLAYBACK_THRESHOLD")

# score threshold for item/brand similarity metrics
ITEM_SCORE_THRESHOLD: float = CONTENT_CONFIGS.get("ITEM_SCORE_THRESHOLD")

# genres for item-brand mapping
ITEM_BRAND_MAPPING_GENRES: Set[str] = set(CONTENT_CONFIGS.get("ITEM_BRAND_MAPPING_GENRES", []))
ITEM_TOPIC_MAPPING_BRANDS: Set[str] = set(CONTENT_CONFIGS.get("ITEM_TOPIC_MAPPING_BRANDS", []))

# genres for post_series_next
SERIES_NEWEST_GENRES: Set[str] = set(CONTENT_CONFIGS.get("SERIES_NEWEST_GENRES", []))
SERIES_NEXT_GENRES: Set[str] = set(CONTENT_CONFIGS.get("SERIES_NEXT_GENRES", []))

# brands for post_topic_next
TOPIC_NEWEST_BRANDS: Set[str] = set(CONTENT_CONFIGS.get("TOPIC_NEWEST_BRANDS", []))
TOPIC_NEXT_BRANDS: Set[str] = set(CONTENT_CONFIGS.get("TOPIC_NEXT_BRANDS", []))

# genres for next-episode from brand in next-video
NEXT_VIDEO_SERIES_NEWEST_GENRES: Set[str] = set(CONTENT_CONFIGS.get("NEXT_VIDEO_SERIES_NEWEST_GENRES", []))
NEXT_VIDEO_SERIES_NEXT_GENRES: Set[str] = set(CONTENT_CONFIGS.get("NEXT_VIDEO_SERIES_NEXT_GENRES", []))

# brands for next-episode from topic in next-video
NEXT_VIDEO_TOPIC_NEWEST_BRANDS: Set[str] = set(CONTENT_CONFIGS.get("NEXT_VIDEO_TOPIC_NEWEST_BRANDS", []))
NEXT_VIDEO_TOPIC_NEXT_BRANDS: Set[str] = set(CONTENT_CONFIGS.get("NEXT_VIDEO_TOPIC_NEXT_BRANDS", []))

HISTORY_PICKS_REFERENCE_ITEM_EXCLUDE_PATHS: Set[str] = set(
    CONTENT_CONFIGS.get("HISTORY_PICKS_REFERENCE_ITEM_EXCLUDE_PATHS", [])
)
SNTW_MORE_EPISODES_BLACKLIST: Set[str] = set(CONTENT_CONFIGS.get("SNTW_MORE_EPISODES_BLACKLIST", []))

# number of candidate items
N_CAND_ITEMS: int = CONTENT_CONFIGS.get("N_CAND_ITEMS")

# max number of items from sequence used in collaborative models
MAX_SEQUENCE_LEN_COLLABORATIVE: int = CONTENT_CONFIGS.get("MAX_SEQUENCE_LEN_COLLABORATIVE")
# max number of items from sequence used in sequential models
MAX_SEQUENCE_LEN_SEQUENTIAL: int = CONTENT_CONFIGS.get("MX_SEQUENCE_LEN_SEQUENTIAL")

# default max score for trending()
DEFAULT_TRENDING_MAX_SCORE: float = CONTENT_CONFIGS.get("DEFAULT_TRENDING_MAX_SCORE")

# exozet profiles
PROFILE_MAPPING: Dict[str, str] = CONTENT_CONFIGS.get("PROFILE_MAPPING")

# interest (MeinZDF, by title) <-> path level 1 mapping
INTEREST_PATH_LEVEL_1_MAPPING: Dict[str, Union[str, List[str]]] = CONTENT_CONFIGS.get("INTEREST_PATH_LEVEL_1_MAPPING")

INTEREST_EXTID_LUT: Dict[str, Union[str, List[str]]] = CONTENT_CONFIGS.get("INTEREST_EXTID_LUT")


# ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# #####                   ETL config                    #####
# ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

DEFAULT_MODEL_TARGET: str = CONTENT_CONFIGS.get("DEFAULT_MODEL_TARGET")
# mapping from target to appid prefix, e.g. "tv": ["appId1", "appId2", ...]
TARGET_TO_APPID_PREFIXES_MAPPING: Dict[str, Tuple[str]] = {
    target: tuple(appids) for target, appids in CONTENT_CONFIGS.get("TARGET_TO_APPID_PREFIXES_MAPPING", {}).items()
}

# TV service selection (include)
TV_SERVICES: List[str] = CONTENT_CONFIGS.get("TV_SERVICES")

# overlap threshold for FSDB matching
OVERLAP_THRESHOLD: float = CONTENT_CONFIGS.get("OVERLAP_THRESHOLD")

# play coverage threshold (skip everything below)
PLAY_COVERAGE_THRESHOLD: float = CONTENT_CONFIGS.get("PLAY_COVERAGE_THRESHOLD")

# play event max age (skip everything below)
PLAY_MAX_AGE_DAYS: int = CONTENT_CONFIGS.get("PLAY_MAX_AGE_DAYS")

# tracking play events <> seconds
TRACKING_PLAY_INTERVAL: float = CONTENT_CONFIGS.get("TRACKING_PLAY_INTERVAL")

# max age (days) per genre, brand, ...
MAX_AGE: Mapping[str, Mapping[str, int]] = CONTENT_CONFIGS.get("MAX_AGE")
MAX_AGE_EXCLUDE: Mapping[str, Mapping[str, int]] = CONTENT_CONFIGS.get("MAX_AGE_EXCLUDE")

# brands for which an 'episode' is assumed to be a multipart content
MULTIPART_BRANDS: List[str] = CONTENT_CONFIGS.get("MULTIPART_BRANDS")

# tags that are considered "synthetic"
SYNTHETIC_TAGS: List[str] = CONTENT_CONFIGS.get("SYNTHETIC_TAGS")

# genres that are considered "fictional"
FICTION_GENRES: List[str] = CONTENT_CONFIGS.get("FICTION_GENRES")

"""re.Pattern that matches named groups (config)_(target)_(group) if available in a configuration"""
_configuration_pattern: Pattern[str] = re.compile(
    r"""
    ^
    (?P<config>
        (?:
            (?#match any word character not followed by _{target} or _{ab_group})
            [\w\-](?!pctablet|mobile|tv|gruppe-[a-e])
        )+
    )
    (?#match _{target})
    (?:_(?P<target>pctablet|mobile|tv))?
    (?#match _{ab_group})
    (?:_(?P<ab_group>gruppe-[a-e]))?
    $
    """,
    re.VERBOSE,
)


def _normalize_configuration(configuration: str) -> str:
    """
    normalizes configuration names to {config-with-hyphens}[_{target}][_{gruppe-(abcde)}], i.e., the config part no
    longers contains any ``_`` after normalization, making it safe to split at ``_`` to extract the different parts
    and allowing to specify all Cluster-(A|P|list) configs with hyphens only

    >>> _normalize_configuration("doku1")
    'doku1'
    >>> _normalize_configuration("dkdi-require-history")
    'dkdi-require-history'
    >>> _normalize_configuration("dkdi_require_history")
    'dkdi-require-history'
    >>> _normalize_configuration("dkdi_require_history_gruppe-a")
    'dkdi-require-history_gruppe-a'
    >>> _normalize_configuration("dkdi_require_history_pctablet")
    'dkdi-require-history_pctablet'
    >>> _normalize_configuration("dkdi_require_history_tv_gruppe-e")
    'dkdi-require-history_tv_gruppe-e'
    >>> _normalize_configuration("dkdi-require-history_tv_gruppe-e")
    'dkdi-require-history_tv_gruppe-e'

    :param configuration: the "raw" configuration from request params
    :return: a normalized config, where ``_`` only splits the parts (config, target, group)
    """
    if configuration is None:
        return ""
    return "_".join(s.replace("_", "-") for s in _configuration_pattern.match(configuration).groups() if s is not None)


def get_group_from_configuration_name(
    configuration: str,
):
    return _configuration_pattern.match(configuration).group("ab_group")


def get_target_from_configuration_name(
    configuration: str,
):
    return _configuration_pattern.match(configuration).group("target")


def get_final_configuration(
    *,
    configuration: str,
    target: str,
    ab_group: str,
    allowed_configurations: Container[str],
) -> str:
    configuration = _normalize_configuration(configuration)
    # combine abGroup information if possible
    if f"{configuration}_{target}_{ab_group}" in allowed_configurations:
        return f"{configuration}_{target}_{ab_group}"
    elif f"{configuration}_{target}" in allowed_configurations:
        return f"{configuration}_{target}"
    elif f"{configuration}_{ab_group}" in allowed_configurations:
        return f"{configuration}_{ab_group}"
    elif configuration in allowed_configurations:
        return configuration
    else:
        logging.error(f"Unknown configuration '{configuration}'.")
        return configuration
