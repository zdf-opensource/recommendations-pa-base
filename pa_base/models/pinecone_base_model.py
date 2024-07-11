#  Copyright (c) 2022, ZDF.
"""
Pinecone base model for generating predictions by invoking a Sagemaker Endpoint

This module implements a simple model capable of invoking a Sagemaker Endpoint to retrieve predictions.

(c) 2022 ZDF
"""

import contextlib
import logging
import sys
import urllib.parse
from collections.abc import Collection, Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

from pa_base.configuration.config import PINECONE_API_KEY, PINECONE_TIMEOUT_SECONDS
from pa_base.models.base_model import KnowsItemMixin, SimilarItemsMixin

if sys.version_info < (3, 8):
    # pinecone-client 3 requires Python 3.8+
    raise ImportError("Pinecone requires Python 3.8+")
else:
    from pinecone import Pinecone

if not PINECONE_API_KEY:
    # try to load from AWS Secrets Manager if PINECONE_API_KEY is not set
    logging.info("PINECONE_API_KEY not set. Trying to load from AWS Secrets Manager.")
    try:
        import os

        import boto3

        from pa_base.configuration.config import DEFAULT_REGION

        pinecone_secret_name: Optional[str] = os.environ.get("PINECONE_SECRET_NAME")
        if pinecone_secret_name is not None:
            logging.info("Retrieving pinecone API key from secret name.")
            boto_session = boto3.Session()
            secrets_client = boto_session.client(service_name="secretsmanager", region_name=DEFAULT_REGION)
            pinecone_api_key = secrets_client.get_secret_value(SecretId=pinecone_secret_name)["SecretString"]
            PINECONE_API_KEY = pinecone_api_key
    except Exception as exc:
        logging.warning(
            f"Could not retrieve Pinecone API Key. Exception: {exc}. "
            "Resuming without the possibility to use pinecone. "
            "I'm just a warning, not an error.",
            exc_info=exc,
        )

try:
    from reco.tracer import Tracer

    tracer = Tracer()
except ImportError:
    # reco is only available in P/A service and P/A plugins
    tracer = None


class PineconeBaseModel(SimilarItemsMixin, KnowsItemMixin):
    """
    Simple Pinecone base model class

    A PineconeBaseModel can invoke a Pinecone index to retrieve scored similar items for a given reference item.

    :param index_name: name of the Pinecone index
    :param namespace: (optional) namespace within the Pinecone index
    """

    def __init__(
        self,
        *,
        index_name: str,
        namespace: str,
    ):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        # this is set only once since we have only one Pinecone Index per PineconeBaseModel
        try:
            # if index_name not in pinecone.list_indexes():
            pc.describe_index(index_name)
        except Exception as exc:
            # should be a pinecone.core.client.exceptions.NotFoundException or an urllib3.exceptions.MaxRetryError
            raise ValueError(f"Pinecone index '{index_name}' does not exist.") from exc
        self._index_name = index_name
        self._namespace = namespace
        self._index = pc.Index(index_name)
        # timeout in (fraction of) seconds or tuple of (connect, read) timeouts
        self._timeout_seconds: Union[int, float, Tuple[Union[int, float], Union[int, float]]] = PINECONE_TIMEOUT_SECONDS

        self.description = f"{self.__class__.__name__} using index:namespace '{index_name}:{namespace}'"

    def __repr__(self):
        return f"{self.__class__.__name__} using index:namespace '{self._index_name}:{self._namespace}'"

    def knows_item(self, externalid: str):
        # fetch with retries
        item = self.fetch_item(externalid)
        return len(item.vectors) > 0 if item is not None else False

    def fetch_item(self, externalid: str):
        with tracer.subsegment_context("pinecone-fetch") if tracer is not None else contextlib.nullcontext():
            fetch_args: Dict[str, Any] = dict(
                ids=[
                    # pinecone only handles ASCII ids reliably --> urlencode query id
                    urllib.parse.quote(externalid)
                ],
                _request_timeout=self._timeout_seconds,
            )
            if self._namespace:
                fetch_args["namespace"] = self._namespace
            try:
                return self._index.fetch(**fetch_args)
            except Exception as exc:
                logging.warning(
                    "Error during pinecone fetch",
                    exc_info=exc,
                )

    def _get_pinecone_top_k(
        self,
        item: Union[str, List[float]],
        *,
        select: Iterable[str] = (),
        n: Optional[int] = None,
        filters: Optional[Dict[str, Dict[str, Union[Collection[str], str]]]] = None,
    ) -> List[Tuple[str, float]]:
        """
        get k nearest items for item id

        :param item: externalid of reference item OR a vector (list of floats) in the latent embedding space
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param filters: model-specific pinecone filter params
        :return iterable of (externalid, score) tuples
        """
        if n is None:
            # reasonable default value, keep None as default arg for uniformity with other "models"
            n = 250
        scores: Iterable[Tuple[str, float]] = ()
        with tracer.subsegment_context("pinecone-query") if tracer is not None else contextlib.nullcontext():
            # query for n+1 items, because the first item is always the reference item and, thus, removed
            query_args: Dict[str, Any] = dict(
                top_k=n + 1,
                _request_timeout=self._timeout_seconds,
            )
            if isinstance(item, str):
                # pinecone only handles ASCII ids reliably --> urlencode query id
                query_args["id"] = urllib.parse.quote(item)
            elif isinstance(item, list):
                query_args["vector"] = item
            else:
                raise ValueError(f"item must be either a string or a list of floats, found '{type(item)}'.")
            if filters is not None:
                query_args["filter"] = filters
            if self._namespace:
                query_args["namespace"] = self._namespace
            with tracer.subsegment_context("pinecone-query") if tracer is not None else contextlib.nullcontext():
                try:
                    result = self._index.query(**query_args)
                    matches = result.get("matches", [])
                    scores = (
                        # pinecone only handles ASCII ids reliably --> urldecode result ids
                        (
                            urllib.parse.unquote(match.get("id")),
                            match.get("score", 0.0),
                        )
                        for match in matches[1:]
                    )
                except Exception as exc:
                    # probably item unknown --> 404 response is thrown as error by pinecone
                    if getattr(exc, "status", None) == 404:
                        # item not found, break the loop
                        logging.warning(
                            "Pinecone query for unknown item",
                            exc_info=exc,
                        )
                    else:
                        logging.warning(
                            "Error during pinecone query",
                            exc_info=exc,
                        )
        if any(select):
            return [score for score in scores if score[0] in select]
        else:
            return list(scores)

    def similar_items(
        self,
        item: str,
        *,
        select: Iterable[str] = (),
        n: Optional[int] = None,
        scale: bool = False,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        get similar items for item

        :param item: externalid of reference item
        :param select: allowed items' externalids
        :param n: max. number of returned items or None for all items
        :param scale: whether scores should be scaled to [0,1]
        :param kwargs: model-specific params
        :return iterable of (externalid, score) tuples
        """
        return self._get_pinecone_top_k(
            item=item,
            select=select,
            n=n,
            filters=kwargs.get("filters", None),
        )
