# Copyright (c) 2023, ZDF.
"""
Util functions for recommendations and web service.
"""

from hashlib import blake2b  # TODO replace by blake3 (pip install blake3)?
from typing import Any, ByteString, Optional, Union, overload


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
