#!/usr/bin/env python3
# Copyright (c) 2024, ZDF.
"""
Unit tests for util.py
"""

from itertools import tee

import pytest

from pa_base.models.util import combine_decayed, decay_func


def pairwise(iterable):
    """
    from itertools recipes (included only in Python 3.10+)

    source: https://docs.python.org/3.6/library/itertools.html#itertools-recipes

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def test_decay():
    times = range(100)
    for i, j in pairwise(times):
        assert decay_func(i) > decay_func(j)


def test_combine_decayed():
    scores1 = list(zip("abcdef", range(6, 0, -1)))
    scores2 = list(zip("defghi", range(6, 0, -1)))
    scores = combine_decayed(scores1, scores2)
    extids, scores = list(zip(*scores))
    # check sorting
    for i, j in pairwise(scores):
        assert i >= j
    # each key should be included only once
    assert len(set(extids)) == len(extids)


if __name__ == "__main__":
    pytest.main(args=["-p", "no:cacheprovider"])
