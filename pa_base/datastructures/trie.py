# Copyright (c) 2023, ZDF.
"""
Trie implementation for efficient prefix matching, e.g., for path prefix matching during filtering of allowed items.
"""

from typing import Sequence, TypeVar

import pandas as pd

_TrieSequence = TypeVar("_TrieSequence", Sequence, str)


class Trie:
    """tree structure for efficient prefix search"""

    def __init__(self):
        self.data = {}
        self.values = []
        self.isLeaf = False

    @classmethod
    def from_series(cls, series: pd.Series) -> "Trie":
        trie = Trie()
        for idx, val in series.items():
            trie.insert(val, idx)
        return trie

    def insert(self, word: _TrieSequence, externalid: str) -> None:
        """
        insert a new 'word' in the tree and tag it with externalid

        :param word: a new value for the prefix search. Can be any str (prefix searchable on char level) or any
            sequence. Using a sequence may be much more performant if the structure allows for larger chunks than using
            each char as a node in the tree.
        :param externalid: each node is tagged with all externalids for faster lookup of matched prefixes
        """
        cur = self
        for i in word:
            if i not in cur.data:
                cur.data[i] = Trie()
            cur = cur.data[i]
            cur.values.append(externalid)
        cur.isLeaf = True

    def startswith(self, prefix: _TrieSequence) -> [str]:
        """
        find all externalids for which the "word" matches the prefix

        works like str.startswith or pd.DataFrame.str.startswith

        :param prefix: any prefix that should be matched, must be of same type as inserted words
        """
        cur = self
        for i in prefix:
            if i not in cur.data:
                return []
            cur = cur.data[i]
        return cur.values
