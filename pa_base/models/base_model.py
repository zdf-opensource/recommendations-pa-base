# Copyright (c) 2024, ZDF.
"""
This module implements a simple model capable of generating scored predictions.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import BinaryIO, Iterable, List, Tuple, TypeVar

import numpy as np

from pa_base.data.s3_util import s3_file


class PredictMixin(ABC):
    @abstractmethod
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
        raise NotImplementedError("Subclasses should overwrite this if it fits their use case")


class SimilarItemsMixin(ABC):
    @abstractmethod
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
        :param kwargs: model-specific params
        :return iterable of (externalid, score) tuples
        """
        raise NotImplementedError("Subclasses should overwrite this if it fits their use case")


class KnowsItemMixin(ABC):
    @abstractmethod
    def knows_item(self, externalid: str):
        """check whether this model 'knows' an item"""
        raise NotImplementedError("Subclasses should overwrite this if it fits their use case")


class BaseModel(KnowsItemMixin):
    """Simple base model class

    A BaseModel can generate scored predictions of future user
    interactions for a given sequence of interactions.

    Models are fetched from cloud storage using simple name suffix
    matching.

    Attributes:
        description: A string providing a description of the model generation.
    """

    def __init__(self, s3_bucket, meta_suffix="meta.pkl", model_suffix=".pkl", tmpdir="/tmp"):
        """load model from model_source and initialize"""
        logging.info(f"Trying to load model '{model_suffix}' / meta '{meta_suffix}' from bucket '{s3_bucket}'.")
        if not s3_bucket:
            err = f"{model_suffix}: Init failed. No model S3 source given: 's3_bucket={s3_bucket}'."
            logging.error(err)
            raise ValueError(err)
        # find model meta data for latest model in bucket
        with s3_file(
            s3_bucket=s3_bucket,
            s3_key_prefix="",
            s3_key_suffix=meta_suffix,
            filetypes=("",),  # already included in suffix
            output_dir=tmpdir,
        ) as file:
            logging.info(f"{meta_suffix}: Using model meta '{file}'.")
            self._load_meta(file)
        # find matching model
        with s3_file(
            s3_bucket=s3_bucket,
            s3_key_prefix="",
            s3_key_suffix=model_suffix,
            filetypes=("",),  # already included in suffix
            output_dir=tmpdir,
        ) as file:
            logging.info(f"{model_suffix}: Using base model '{file}'.")
            self._load_model(file)
        logging.info(f"Model '{model_suffix}' initialized: '{self.description}'.")

    def _load_meta(self, meta_file: BinaryIO):
        """load meta file, set description and initialize item-id-mapping"""
        metadata = pickle.load(meta_file)
        mand_keys = ["description"]
        assert all([key in list(metadata.keys()) for key in mand_keys])
        if "externalid_for_itemid" in metadata:
            self._extid_for_itemid = metadata["externalid_for_itemid"]
            self._itemid_for_extid = {v: k for k, v in self._extid_for_itemid.items()}
        self.description = metadata["description"]

    def _load_model(self, model_file: BinaryIO):
        """load model file"""
        model = pickle.load(model_file)
        self._model = model

    @property
    def itemids(self):
        """get externalids known to model"""
        return list(self._itemid_for_extid.keys())

    def knows_item(self, externalid: str):
        """check whether this model 'knows' an item"""
        return externalid in self.itemids


T = TypeVar("T")


def normalize_scores(scores: Iterable[Tuple[T, float]]) -> List[Tuple[T, float]]:
    """
    normalize scores given as (idx, score) tuples

    :param scores: list of (externalid: str, score: float) tuples
    :returns: list of (externalid: str, score: float) tuples in input order with [0,1] normalized scores
    :raises: TypeError if scores are not passed as float
    """
    if not scores:
        return []
    # transform list of (id,score)-tuples into two lists (ids and scores)
    # splat/unpacking works for containers (list, tuple, ...) and iterators/generators (slice, islice, ...)
    extids_values = list(zip(*scores))
    if not len(extids_values):
        # check if any values are present before unpacking
        return []
    extids, values = extids_values
    if not len(values):
        # check after unrolling (by splatting) because empty iterators/generators cannot be otherwise detected
        return []
    values: np.array = np.array(values)
    # NORMALIZE
    # solution using sklearn --> heavy import
    #   from sklearn.preprocessing import maxabs_scale
    #   values = maxabs_scale(values)
    # pure numpy: ptp == peak-to-peak, giving the distance between min and max (negative if sign of them differs)
    ptp = np.ptp(values)
    if ptp != 0.0:
        values = (values - np.min(values)) / ptp
    else:
        values = np.ones_like(values)
    # transform two lists (ids and scores) back into one list of (id,score)-tuples
    return list(zip(extids, values.tolist()))
