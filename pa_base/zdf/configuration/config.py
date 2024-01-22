# Copyright (c) 2024, ZDF.
"""
ZDF-specific configuration.
"""
import typing as t

ZDF_CONTENT_DUMP_COLS: t.List[str] = [
    "externalid",
    "id",
    "contenttype",
    "tvservice",
    "title",
    "brand",
    "brand_id",
    "brand_externalid",
    "leadparagraph",
    "path",
    "hasvideo",
    "enddate",
    "video_duration",
    "publication_date",
    "visiblefrom",
    "visibleto",
    "tivi_age_group",
    "current_videotype",
    "editorial_date",
    "editorial_tags",
    "teaserimage",
    "search_service_tagging_results",
    "prodNr",
    "has_dgs",
    "has_ut",
    "has_ad",
    "extracted_initial_publication_date",
    "zdfinfo_metadata",
    "is_stage_allowed",
    "is_poster_allowed",
    "is_partnercontent",
    "content_owner",
    "is_doku",
    "sntw_category",
    "sntw_categories",
    "airtimebegin",
    "airtimeend",
    "actor_details",
    "crew_details",
    "episode_number",
    "season_number",
    "pharos_ids",
    "airtimebegins",
    "country",
    "text",
    "path_level_1",
    "path_level_2",
    "is_multipart",
    "is_visible",
    "is_visible_reason",
    "is_fiction",
    "series_uuid",
    "series_title",
    "series_index_page_id",
    "series_index_page_externalid",
    "video_fsk",
    "duplicates",
    "original_externalId",
    "load_recommendations",
]
ZDF_CONTENT_DUMP_DATECOLS: t.List[str] = [
    "enddate",
    "publication_date",
    "visiblefrom",
    "visibleto",
    "editorial_date",
    "extracted_initial_publication_date",
    "airtimebegin",
    "airtimeend",
]
ZDF_CONTENT_DUMP_CATCOLS: t.List[str] = [
    "tvservice",
    "brand",
    "brand_id",
    "brand_externalid",
    "current_videotype",
    "country",
    "contenttype",
    "path_level_1",
    "path_level_2",  # TODO remove path_level_2
    "tivi_age_group",
    "series_uuid",
    "series_title",
    "series_index_page_id",  # TODO remove series_index_page_id
    "series_index_page_externalid",
    "content_owner",
    "sntw_category",
    "video_fsk",
]
ZDF_CONTENT_DUMP_INTCOLS: t.List[str] = ["episode_number", "season_number"]
ZDF_CONTENT_DUMP_STRCOLS: t.List[str] = ["editorial_tags", "pharos_ids"]
ZDF_CONTENT_DUMP_BOOLCOLS: t.List[str] = ["load_recommendations"]
