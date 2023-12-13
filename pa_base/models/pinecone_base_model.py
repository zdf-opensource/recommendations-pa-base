"""Pinecone base model for generating predictions by invoking a Sagemaker Endpoint

This module implements a simple model capable of invoking a Sagemaker Endpoint to retrieve predictions.

(c) 2022 ZDF
"""
#  Copyright (c) 2022, ZDF.
import contextlib
import logging
import urllib.parse
from collections.abc import Collection, Iterable
from typing import Dict, List, Optional, Tuple, Union

import pinecone

from pa_base.configuration.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from pa_base.models.base_model import KnowsItemMixin, SimilarItemsMixin

if not PINECONE_API_KEY:
    # try to load from AWS Secrets Manager if PINECONE_API_KEY is not set
    logging.info("PINECONE_API_KEY not set. Trying to load from AWS Secrets Manager.")
    try:
        import os

        import boto3

        from pa_base.configuration.config import DEFAULT_REGION

        pinecone_secret_name: str = os.environ.get("PINECONE_SECRET_NAME")
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


pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


try:
    from reco.tracer import Tracer

    tracer = Tracer()
except ImportError:
    # reco is only available in P/A service and P/A plugins
    tracer = None


class PineconeBaseModel(SimilarItemsMixin, KnowsItemMixin):
    """Simple Pinecone base model class

    A PineconeBaseModel can invoke a Pinecone index to retrieve scored similar items for a given reference item.

    :param index_name: A string providing a description of the model generation.
    """

    def __init__(
        self,
        *,
        index_name: str,
        namespace: str,
    ):
        # this is set only once since we have only one Pinecone Index per PineconeBaseModel
        try:
            # if index_name not in pinecone.list_indexes():
            pinecone.describe_index(index_name)
        except Exception as exc:
            # should be a pinecone.core.client.exceptions.NotFoundException or an urllib3.exceptions.MaxRetryError
            raise ValueError(f"Pinecone index '{index_name}' does not exist.") from exc
        self._index_name = index_name
        self._namespace = namespace

        self.description = f"{self.__class__.__name__} using index:namespace '{self._index_name}:{self._namespace}'"

    def __repr__(self):
        return f"{self.__class__.__name__} using index:namespace '{self._index_name}:{self._namespace}'"

    def knows_item(self, externalid: str):
        # fetch with retries and a fresh index each time
        max_retries = 2
        for i in range(1, max_retries + 1):
            context = (
                tracer.subsegment_context(f"pinecone-fetch-try-{i}") if tracer is not None else contextlib.nullcontext()
            )
            with context:
                fetch_args: Dict[str, str] = dict(
                    ids=[
                        # pinecone only handles ASCII ids reliably --> urlencode query id
                        urllib.parse.quote(externalid)
                    ],
                )
                if self._namespace:
                    fetch_args["namespace"] = self._namespace
                try:
                    with pinecone.Index(self._index_name, pool_threads=1) as index:
                        result = index.fetch(**fetch_args)  # .vectors[externalid].values
                        # return true if the item exists
                        # return externalid in result.vectors
                        return len(result.vectors) > 0
                except Exception as exc:
                    logging.warning(
                        f"Error during pinecone fetch -> retry count {i} / {max_retries}",
                        exc_info=exc,
                    )
        # successful return is already done in the retry loop above, thus we only return False here
        return False

    def _get_pinecone_top_k(
        self,
        item: Union[str, List[float]],
        *,
        select: Iterable[str] = (),
        n: Optional[int] = None,
        filters: Optional[Dict[str, Dict[str, Union[Collection[str], str]]]] = None,
    ) -> List[Tuple[str, float]]:
        """get k nearest items for item id

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
        context = tracer.subsegment_context("pinecone-query") if tracer is not None else contextlib.nullcontext()
        with context:
            # query for n+1 items, because the first item is always the reference item and, thus, removed
            query_args: Dict[str, Union[str, int, Dict[str, Collection[str]]]] = dict(
                top_k=n + 1,
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
            # result = self.index.query(**query_args)
            # query with retries and a fresh index each time
            max_retries = 2
            for i in range(1, max_retries + 1):
                context = (
                    tracer.subsegment_context(f"pinecone-query-try-{i}")
                    if tracer is not None
                    else contextlib.nullcontext()
                )
                with context:
                    try:
                        with pinecone.Index(self._index_name, pool_threads=1) as index:
                            result = index.query(**query_args)
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
                        logging.warning(
                            f"Error during pinecone query -> retry count {i} / {max_retries}.",
                            exc_info=exc,
                        )
                        if getattr(exc, "status") == 404:
                            # item not found, break the loop
                            break
                    else:
                        break

        if any(select):
            scores: List[Tuple[str, float]] = [score for score in scores if score[0] in select]
        else:
            scores: List[Tuple[str, float]] = list(scores)
        return scores

    def similar_items(
        self,
        item: str,
        *,
        select: Iterable[str] = (),
        n: Optional[int] = None,
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
        return self._get_pinecone_top_k(
            item=item,
            select=select,
            n=n,
            filters=kwargs.get("filters", None),
        )
