# Copyright (c) 2023, ZDF.
"""
Utility functions for service configuration.
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple

from pa_base.configuration.config import CONFIG_BUCKET
from pa_base.data.s3_util import download_yaml_from_s3, s3_file


def download_if_exists(config_name: str, s3_bucket: str = "") -> Dict[str, Any]:
    """
    load config from s3 and parse YAML into python structures, e.g., dict

    >>> type(_download_if_exists('cluster_configs.yml'))
    dict

    :param config_name: name of the config file including file type suffix such as ".yml"
    :param s3_bucket: name of an s3 bucket, uses default config bucket if not given
    :return: the config as dict
    """
    try:
        return download_config(config_name=config_name, s3_bucket=s3_bucket)
    except FileNotFoundError:
        logging.warning(f"Config '{config_name}' not found.")
        return {}


def download_json_if_exists(json_name: str, s3_bucket: str = "") -> Dict[str, Any]:
    """
    load JSON file from s3 and parse into python structures, e.g., dict

    >>> type(_download_json_if_exists('pa-pubsub-publisher-sa.json'))
    dict

    :param json_name: name of the JSON file including ".json" suffix
    :param s3_bucket: name of an s3 bucket, uses default config bucket if not given
    :return: the config as dict
    """
    if not s3_bucket:
        s3_bucket = CONFIG_BUCKET
    try:
        with s3_file(
            s3_bucket=s3_bucket,
            s3_key_prefix=json_name,
            filetypes=["json"],
        ) as file:
            return json.load(file)
    except FileNotFoundError:
        logging.warning(f"JSON file '{json_name}' not found.")
        return {}


def async_exec(*funcs_args: Tuple[Any, ...]) -> list:
    """
    execute multiple functions asynchronously as coroutines

    >>> def f(): return None
    >>> def g(x): return x
    >>> def h(y,z): return y,z
    >>> async_exec((f,), (g, 1), (h, 2, 3))
    [None, 1, (2, 3)]

    :param funcs_args: multiple tuples of (func, *args), i.e., (print, "1", "2", "3")
    :return: results in the same order as provided functions
    """

    async def parallel(*_funcs_args: Tuple[Any, ...]) -> tuple:
        # max_workers set to default of python 3.8 https://docs.python.org/3/library/concurrent.futures.html
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
            _loop = asyncio.get_event_loop()
            futures = [_loop.run_in_executor(executor, func[0], *func[1:]) for func in _funcs_args]
            return await asyncio.gather(*futures)

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(parallel(*funcs_args))
    except RuntimeError as err:
        logging.error("Async config download failed. Procedural retry follows.", exc_info=err)
        # retry in a non-parallel fashion if async exec fails
        # fut = asyncio.run_coroutine_threadsafe(parallel(*funcs_args), loop)
        # result = yield from asyncio.wait_for(fut, 50.0)
        # return result
        return [func[0](*func[1:]) for func in funcs_args]


def download_config(config_name: str, s3_bucket: str = "") -> Dict[str, Any]:
    """
    load config from s3 and parse YAML into python structures, e.g., dict

    >>> type(download_config('cluster_configs.yml'))
    dict

    :param config_name: name of the config file including file type suffix such as ".yml"
    :param s3_bucket: name of an s3 bucket, uses default config bucket if not given
    :return: the config as dict
    """
    logging.info(f"Trying to load config '{config_name}' from bucket '{s3_bucket}'.")
    if not s3_bucket:
        s3_bucket = CONFIG_BUCKET
        err = f"Using bucket '{s3_bucket}' instead."
        logging.info(err)
    return download_yaml_from_s3(s3_key_prefix=config_name, s3_bucket=s3_bucket)
