# Copyright (c) 2023, ZDF.
"""
Param objects for recos/automations/clusterlists.
"""

import copy
from typing import Any, Collection, Dict, List, Mapping, Optional, Set

import attr


@attr.dataclass(kw_only=True)
# on python 3.7 this could be solver with dataclasses.dataclass and deep-copying in __post_init_()
class BaseParams:
    """
    params for valid item selection

    **Fields**
        | ``n_items`` -- number of results to be generated
        | ``min_teasers`` -- lower bound of cluster length, all clusters with fewer teasers will be removed,\
              0 allows all non-empty clusters
        | ``options`` -- dictionary of options
        | ``paths`` -- paths allowed in result set ("/zdf", "/zdf/dokumentation", ...)
        | ``exclude_paths`` -- paths NOT allowed in result set ("/zdf", "/zdf/dokumentation", ...)
        | ``genres`` -- genres (path_level_1) allowed in result set ("dokumentation", ...)
        | ``exclude_genres`` -- genres (path_level_1) NOT allowed in result set ("dokumentation", ...)
        | ``tags`` -- editorial tags to be included in recommendations
        | ``tags_exact_matching`` -- whether {tags} should match editorial tags exact (true) or partial (false)
        | ``exclude_tags`` -- editorial tags to be excluded from recommendations
        | ``exclude_tags_exact_matching`` -- whether {exclude_tags} should match editorial tags exact (true) or
        |     partially (false)
        | ``brand_tags`` -- editorial tags on the brand of an item to be included in recommendations
        | ``brand_tags_exact_matching`` -- whether {brand_tags} should match editorial tags exact (true) or
              partially (false)
        | ``brand_exclude_tags`` -- editorial tags on the brand of an item to be excluded from recommendations
        | ``brand_exclude_tags_exact_matching`` -- whether {brand_exclude_tags} should match tags exact (true) or
              partially (false)
        | ``types`` -- types allowed in result set (clip, episode, ...)
        | ``tvservices`` -- tvservices allowed in result set (ZDF, ZDFinfo, ...), None / empty means all are allowed
        | ``min_duration`` -- in seconds, 0 to allow all durations
        | ``exclude_docs`` -- arbitrary items to be filtered out
        | ``brands`` -- arbitrary brands' externalids to be included (all others filtered out)
        | ``exclude_brands`` -- arbitrary brands' externalids to be filtered out
        | ``recoexplain`` -- whether output should contain contain explanations for each item
        | ``accessibility_enabled`` -- whether to adhere to accessibility_options (true) or ignore/discard them (false)
        | ``accessibility_options`` --
              empty set by default to allow everything, may include "ut" (subtitles),
              "ad" (audio descriptions) and "dgs" (sign language) to keep only matching items
        | ``tivi_age_group`` -- differentiate between ZDFchen ("2-5") and other tivi content ("")
        | ``teaser_type`` -- allow only items with a specific type of teaser, e.g., group-cluster-poster for stage
              automation
    """

    n_items: int = 26
    min_teasers: int = 0
    options: Optional[Mapping[str, Any]] = attr.ib(
        factory=dict,
        # prevent AttributeErrors if options is None
        converter=(lambda x: {} if x is None else copy.deepcopy(x)),
    )
    # deepcopy to prevent downstream changes of original params
    # for reference: https://stackoverflow.com/a/5105554/3410474
    # deepcopy is capable of copying even nested dictionaries - but not thread-safe
    # thread-safe alternative: self.options = json.loads(json.dumps(options))
    paths: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    exclude_paths: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    genres: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    exclude_genres: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    sntw_categories: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    exclude_sntw_categories: Collection[str] = attr.ib(
        default=(), converter=copy.deepcopy
    )
    tags: Collection[str] = attr.ib(factory=tuple, converter=copy.deepcopy)
    tags_exact_matching: bool = False
    exclude_tags: Collection[str] = attr.ib(factory=tuple, converter=copy.deepcopy)
    exclude_tags_exact_matching: bool = False
    brand_tags: Collection[str] = attr.ib(factory=tuple, converter=copy.deepcopy)
    brand_tags_exact_matching: bool = False
    brand_exclude_tags: Collection[str] = attr.ib(
        factory=tuple, converter=copy.deepcopy
    )
    brand_exclude_tags_exact_matching: bool = False
    types: Collection[str] = attr.ib(
        default=("clip", "episode"), converter=copy.deepcopy
    )
    tvservices: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    min_duration: int = 0
    exclude_docs: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    brands: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    exclude_brands: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    recoexplain: bool = False
    # enable accessibility_options or discard them
    accessibility_enabled: bool = False
    # accessibility params: any combination of {"ut", "ad", "dgs"} or an empty set to allow all
    accessibility_options: Set[str] = attr.ib(
        factory=frozenset,
        converter=(lambda x: frozenset() if x is None else frozenset(copy.deepcopy(x))),
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.in_(options={"ut", "ad", "dgs"})
        ),
    )
    tivi_age_group: str = attr.ib(
        default="",
        converter=attr.converters.default_if_none(""),
        validator=attr.validators.in_(options={"", "2-5"}),
    )
    teaser_type: str = attr.ib(
        default="",
        converter=attr.converters.default_if_none(""),
    )
    content_owners: Collection[str] = attr.ib(default=(), converter=copy.deepcopy)
    exclude_content_owners: Collection[str] = attr.ib(
        # PERAUT-1244 ARD should be excluded by default if not explicitly allowed
        default=("ARD",),
        converter=copy.deepcopy,
    )
    only_documentaries: bool = False

    # other params, that have to be present for postprocessing but
    # cannot be set in BaseParams because it makes no sense semantically
    target: str = attr.ib(default="pctablet", init=False)
    padding: str = attr.ib(
        default="none", init=False, converter=attr.converters.default_if_none("none")
    )
    promotion_assets: List[Dict[str, Any]] = attr.ib(
        factory=list,
        init=False,
        converter=lambda x: [] if x is None else copy.deepcopy(x),
    )

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        if "limit" in conf_dict:
            self.n_items = conf_dict.get("limit")
        if "min_teasers" in conf_dict:
            self.min_teasers = conf_dict.get("min_teasers")
        if ("options" in conf_dict) and (conf_dict["options"] is not None):
            self.options = {**self.options, **conf_dict.get("options")}
        if "paths" in conf_dict:
            self.paths = conf_dict.get("paths")
        if "exclude_paths" in conf_dict:
            self.exclude_paths = conf_dict.get("exclude_paths")
        if "genres" in conf_dict:
            self.paths = conf_dict.get("genres")
        if "exclude_genres" in conf_dict:
            self.exclude_paths = conf_dict.get("exclude_genres")
        if "sntw_categories" in conf_dict:
            self.sntw_categories = conf_dict.get("sntw_categories")
        if "exclude_sntw_categories" in conf_dict:
            self.exclude_sntw_categories = conf_dict.get("exclude_sntw_categories")
        if "tags" in conf_dict:
            self.tags = conf_dict.get("tags")
        if "tags_exact_matching" in conf_dict:
            self.tags_exact_matching = conf_dict.get("tags_exact_matching")
        if "exclude_tags" in conf_dict:
            self.exclude_tags = conf_dict.get("exclude_tags")
        if "exclude_tags_exact_matching" in conf_dict:
            self.exclude_tags_exact_matching = conf_dict.get(
                "exclude_tags_exact_matching"
            )
        if "brand_tags" in conf_dict:
            self.brand_tags = conf_dict.get("brand_tags")
        if "brand_tags_exact_matching" in conf_dict:
            self.brand_tags_exact_matching = conf_dict.get("brand_tags_exact_matching")
        if "brand_exclude_tags" in conf_dict:
            self.brand_exclude_tags = conf_dict.get("brand_exclude_tags")
        if "brand_exclude_tags_exact_matching" in conf_dict:
            self.brand_exclude_tags_exact_matching = conf_dict.get(
                "brand_exclude_tags_exact_matching"
            )
        if "types" in conf_dict:
            self.types = conf_dict.get("types")
        if "tvservices" in conf_dict:
            self.tvservices = conf_dict.get("tvservices")
        if "min_duration" in conf_dict:
            self.min_duration = conf_dict.get("min_duration")
        if "exclude_docs" in conf_dict:
            self.exclude_docs = conf_dict.get("exclude_docs")
        if "brands" in conf_dict:
            self.brands = conf_dict.get("brands")
        if "exclude_brands" in conf_dict:
            self.exclude_brands = conf_dict.get("exclude_brands")
        if "accessibility_enabled" in conf_dict:
            self.accessibility_enabled = conf_dict.get("accessibility_enabled")
        if "accessibility_options" in conf_dict:
            self.accessibility_options = conf_dict.get("accessibility_options")
        if "tivi_age_group" in conf_dict:
            self.tivi_age_group = conf_dict.get("tivi_age_group") or ""
        if "teaser_type" in conf_dict:
            self.teaser_type = conf_dict.get("teaser_type") or ""
        if "content_owners" in conf_dict:
            self.content_owners = conf_dict.get("content_owners")
        if "exclude_content_owners" in conf_dict:
            # PERAUT-1244 ARD should be excluded by default if not explicitly allowed
            self.exclude_content_owners = conf_dict.get(
                "exclude_content_owners", ("ARD",)
            )
        if "only_documentaries" in conf_dict:
            self.only_documentaries = conf_dict.get("only_documentaries")
        if "target" in conf_dict:
            self.target = conf_dict.get("target")
        if "padding" in conf_dict:
            # replace None by "none" to be explicit and type-safe
            self.padding = conf_dict.get("padding") or "none"
        if "promotion_assets" in conf_dict:
            self.promotion_assets = conf_dict.get("promotion_assets") or []


@attr.dataclass(kw_only=True)
class AutomationParams(BaseParams):
    """
    params for automations

    **Fields** from :class:`BaseParams` plus the following:
        | ``target`` -- target platform such as 'pctablet'
        | ``promotion_assets`` -- list of promotions (reco-like dict format)

    the format of ``promotion_assets`` is::

        [{
            'externalid': 'SCMS_44561b43-784c-497e-8348-d50fb459f5b0',
            'id': 'countdown-zum-kriegsende-die-letzten-100-tage-durchbruch-im-osten-102',
            'path': '/zdf/dokumentation/zdfinfo-doku',
            'target': 'https://www.zdf.de/dokumentation/zdfinfo-doku/countdown-zum-kriegsende-die-letzten-100-tage-durchbruch-im-osten-102.html',
            'score': 1.,
            'video_type': 'episode',
            'position': 0,
        }]
    """

    target: str = "pctablet"
    promotion_assets: List[Dict[str, Any]] = attr.ib(
        factory=list, converter=lambda x: [] if x is None else copy.deepcopy(x)
    )

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )
        if "target" in conf_dict:
            self.target = conf_dict.get("target")
        if "promotion_assets" in conf_dict:
            self.promotion_assets = conf_dict.get("promotion_assets") or []


@attr.dataclass(kw_only=True)
class RecommendationBaseParams(AutomationParams):
    """
    base params for all recommendation-like things, e.g., recommendations, clusterlists, ...

    **Fields** from :class:`AutomationParams` plus the following:
        | ``history_doc_ids`` -- already visited/watched items to be filtered out
        | ``keep_history_docs`` -- set True to not filter out history_doc_ids (but still use them for recos),
              default False
        | ``padding`` -- optional padding, may be None/False or any fallback (WiP, currently only supports trending)
    """

    history_doc_ids: Collection[str] = attr.ib(
        # default=(),
        # converter=copy.deepcopy,
        factory=list,
        # prevent AttributeErrors if history_doc_ids is None
        converter=(lambda x: [] if x is None else list(copy.deepcopy(x))),
    )
    keep_history_docs: bool = False
    padding: str = attr.ib(
        default="none", converter=attr.converters.default_if_none("none")
    )

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )
        if "keep_history_items" in conf_dict:
            self.keep_history_docs = conf_dict.get("keep_history_items")
        if "padding" in conf_dict:
            # replace None by "none" to be explicit and type-safe
            self.padding = conf_dict.get("padding") or "none"


@attr.dataclass(kw_only=True)
class RecommendationParams(RecommendationBaseParams):
    """
    params for recommendations

    **Fields** from :class:`RecommendationBaseParams`
    """

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )


@attr.dataclass(kw_only=True)
class ClusterlistParams(RecommendationBaseParams):
    """
    params for clusterlists

    **Fields** from :class:`RecommendationBaseParams` plus the following:
        | ``n_clusters`` -- maximum number of clusters to be generated
        | ``list_type`` -- the type of clusterlist to be generated, e.g., favourite_genres, history_picks,
              trending_picks. Keep empty for fallback.
        | ``no_dups`` -- whether duplicate items in subsequent clusters should be filtered
        | ``cluster_list`` -- list of cluster-p configs or nodeId/path tuples that specify the content of
              list_type=cluster_list
    """

    n_clusters: int = 3
    list_type: str = "empty"
    no_dups: bool = False
    cluster_list: Optional[List[Dict[str, str]]] = attr.ib(
        factory=list,
        converter=(lambda x: [] if x is None else list(copy.deepcopy(x))),
    )

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )
        if "n_clusters" in conf_dict:
            self.n_clusters = conf_dict.get("n_clusters")
        if "list_type" in conf_dict:
            self.list_type = conf_dict.get("list_type")
        if "no_dups" in conf_dict:
            self.no_dups = conf_dict.get("no_dups")
        if "cluster_list" in conf_dict and isinstance(
            conf_dict.get("cluster_list"), list
        ):
            self.cluster_list = conf_dict.get("cluster_list")


@attr.dataclass(kw_only=True)
class CMABRecommendationParams(RecommendationBaseParams):
    """
    params for CMAB cluster
        config: CMAB config
    """

    config: str = ""
    history_doc_ids = []
    abGroup: str = ""

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )
        if "config" in conf_dict:
            self.config = conf_dict.get("config")
        if "abGroup" in conf_dict:
            self.abGroup = conf_dict.get("abGroup")


@attr.dataclass(kw_only=True)
class ContentClusterlistParams(ClusterlistParams):
    """
    params for contentClusterlists

    **Fields** from :class:`ClusterlistParams` plus the following:
        | ``genre`` -- genre to create clusters of
        | ``cluster_offset`` -- pagination parameter
        | ``configuration`` -- clustering configuration
    """

    min_teasers = 5
    cluster_offset: int = 0
    genre: str = "dokumentation"
    configuration: str = "default"

    def update_with_configuration_dict(
        self, conf_dict: Mapping[str, Any], copy_conf_dict: bool = True
    ):
        if copy_conf_dict:
            # prevent accidental modifications of configurations
            conf_dict = copy.deepcopy(conf_dict)
        super().update_with_configuration_dict(
            conf_dict=conf_dict, copy_conf_dict=False
        )
        if "genre" in conf_dict:
            self.genre = conf_dict.get("genre")
        if "clusterOffset" in conf_dict:
            self.cluster_offset = conf_dict.get("clusterOffset")
        if "configuration" in conf_dict:
            self.configuration = conf_dict.get("configuration")
