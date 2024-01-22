# Copyright (c) 2024, ZDF.
"""
ZDF-specific utility functions for recommendations and web service.
"""

import re
from functools import reduce
from typing import Collection, Optional, Set

import numpy as np
import pandas as pd
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from pa_base.datastructures.trie import Trie
from pa_base.util import freeze
from pa_base.zdf.datastructures.params import BaseParams, RecommendationBaseParams


@cached(
    cache=TTLCache(maxsize=128, ttl=300),  # cache for 5 minutes
    # cache key: discard non-hashable content & trie, freeze all other params to make them hashable
    key=lambda content, content_path_trie, **kwargs: hashkey(**{k: freeze(v) for k, v in kwargs.items()}),
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
    exclude_tivi_age_groups: Collection[str],
    teaser_type: str,
    content_owners: Collection[str],
    exclude_content_owners: Collection[str],
    only_documentaries: bool,
    video_fsks: Collection[str],
    exclude_video_fsks: Collection[str],
    exclude_sensitive_content: bool,
) -> pd.Series:
    valid_items: pd.DataFrame = content
    # aggregate all filters in a list to be applied at the bottom
    conditions = [
        valid_items.is_visible,
    ]
    if exclude_sensitive_content:
        conditions.append(valid_items.load_recommendations)
    if tvservices:
        conditions.append(valid_items.tvservice.isin(tvservices))
    if min_duration:
        conditions.append(valid_items.video_duration >= min_duration)
    if accessibility_enabled and accessibility_options:
        if "ut" in accessibility_options:
            conditions.append(valid_items.has_ut)
        if "ad" in accessibility_options:
            conditions.append(valid_items.has_ad)
        if "dgs" in accessibility_options:
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
                valid_items.externalid.isin([extid for path in paths for extid in content_path_trie.startswith(path)])
            )
    if exclude_paths:
        if content_path_trie is None:
            conditions.append(~valid_items.path.str.startswith(tuple(exclude_paths)))
        else:
            # trie is faster, but may not be passed, e.g., for test/mig content
            conditions.append(
                ~valid_items.externalid.isin(
                    [extid for path in exclude_paths for extid in content_path_trie.startswith(path)]
                )
            )
    if genres:
        conditions.append(valid_items.path_level_1.isin(list(genres)))
    if exclude_genres:
        conditions.append(~valid_items.path_level_1.isin(list(exclude_genres)))
    if sntw_categories:
        sntw_category_conditions = [
            valid_items[category_column]
            for category_column in ["sntw_" + c for c in sntw_categories]
            if category_column in valid_items.columns
        ]
        conditions.append(reduce(np.logical_or, sntw_category_conditions))
    if exclude_sntw_categories:
        for category_column in ["sntw_" + c for c in exclude_sntw_categories]:
            if category_column in valid_items.columns:
                conditions.append(~valid_items[category_column])

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
    if exclude_tivi_age_groups:
        conditions.append(~valid_items.tivi_age_group.isin(exclude_tivi_age_groups))
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
    if only_documentaries:
        # TODO remove col check once content_owner is available in all environments
        conditions.append(valid_items.is_doku)
    if video_fsks:
        conditions.append(valid_items.video_fsk.isin(list(video_fsks)))
    if exclude_video_fsks:
        conditions.append(~valid_items.video_fsk.isin(list(exclude_video_fsks)))
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
        exclude_tivi_age_groups=params.exclude_tivi_age_groups,
        teaser_type=params.teaser_type,
        content_owners=params.content_owners,
        exclude_content_owners=params.exclude_content_owners,
        only_documentaries=params.only_documentaries,
        video_fsks=params.video_fsks,
        exclude_video_fsks=params.exclude_video_fsks,
        exclude_sensitive_content=params.exclude_sensitive_content,
    )

    # remove history docs here because this part would break caching on _get_valid_items()
    if isinstance(params, RecommendationBaseParams) and (not params.keep_history_docs) and params.history_doc_ids:
        valid_items.drop(
            index=params.history_doc_ids,  # noqa
            inplace=True,
            errors="ignore",
        )

    return valid_items
