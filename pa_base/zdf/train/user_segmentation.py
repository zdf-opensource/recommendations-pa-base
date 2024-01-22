# Copyright (c) 2024, ZDF.
"""
Module containing pre-defined user segments and functionality to segment users based on history.

Currently, this offers only a very simplistic segmentation based only on
most-consumed genre for testing user segmentation and segment-based recommendations.
"""

from enum import IntEnum
from typing import Collection

import pandas as pd

# mapping of segment numbers to segment names (genres)
_SEGMENTS_TO_GENRES = {
    0: ["default"],
    1: ["serien", "filme"],
    2: ["comedy", "show"],
    3: ["dokumentation", "wissen"],
    4: ["politik", "gesellschaft", "kultur", "verbraucher"],
    5: [
        "nachrichten",
    ],
    6: [
        "sport",
    ],
}
# reverse mapping of segment names (genres, unique) to segment numbers (non-unique)
_GENRES_TO_SEGMENTS = {w: k for k, v in _SEGMENTS_TO_GENRES.items() for w in v}
# enum from above reverse mapping for type-safe passing through the system
GenreSegmentsEnum = IntEnum("GenreSegmentsEnum", _GENRES_TO_SEGMENTS)
GenreSegmentsEnum.names = [s for e in _SEGMENTS_TO_GENRES.values() for s in e]
GenreSegmentsEnum.values = [e.value for e in GenreSegmentsEnum]
default_segment = GenreSegmentsEnum["default"]


def get_segment_for_history(history_doc_ids: Collection[str], content: pd.DataFrame) -> GenreSegmentsEnum:
    """
    get user segment either by hashed userid (from precomputed mapping) or extract from history

    :param history_doc_ids: externalids of items already seen by the user
    :param content: content to draw valid items from
    :return: a user segment for stage recommendations
    """
    # get user_segment from the most common genre in their history
    most_common_genres = content["path_level_1"].reindex(history_doc_ids).dropna()
    most_common_genres = most_common_genres[most_common_genres.isin(GenreSegmentsEnum.names)]
    segment = most_common_genres.mode()
    try:
        return GenreSegmentsEnum[segment[0]]
    except (KeyError, IndexError):
        # this segment does not exist in the enum
        return default_segment
