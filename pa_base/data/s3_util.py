# Copyright (c) 2023, ZDF.
"""
Helpers and abstractions for loading data from S3.
"""

import gzip
import json
import logging
import os
from contextlib import contextmanager, suppress
from tempfile import TemporaryFile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, overload

import boto3
import pandas as pd
import yaml

# silence logging for boto3/botocore
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)


def s3_upload_file(
    *,
    filepath: str,
    s3_bucket: str,
    s3_key_prefix: str = None,
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
) -> None:
    """
    uploads a given file to s3

    :param filepath: local path to file
    :param s3_bucket: target s3 bucket
    :param s3_key_prefix: object name in s3 (uses filename if empty)
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    """
    if s3_client is None:
        # use a session to enable async execution of this function
        s3_session = boto3.session.Session()
        s3_client = s3_session.client("s3")
    if s3_key_prefix is None:
        s3_key_prefix = os.path.basename(filepath)
    logging.info(f"Uploading '{filepath}' as '{s3_key_prefix}' to '{s3_bucket}'.")
    try:
        s3_client.upload_file(filepath, s3_bucket, s3_key_prefix)
    except Exception as exc:
        # probably a botocore.exceptions.ClientError or boto3.exceptions.S3UploadError
        logging.error(
            f"S3 upload of '{filepath}' as '{s3_key_prefix}' to '{s3_bucket}' failed.",
            exc_info=exc,
        )
    logging.info(f"S3 upload of '{filepath}' as '{s3_key_prefix}' to '{s3_bucket}' finished.")


# class CustomTemporaryFile(SpooledTemporaryFile):
#     def __init__(self, *, s3_key: str, filetype: str, **kwargs):
#         """
#         Extends SpooledTemporaryFile with a filetype property
#
#         :param filetype: filetype of the file, e.g., "csv", "json.gz", "pkl", ...
#         :param kwargs: init args for SpooledTemporaryFile
#         """
#         super().__init__(**kwargs)
#         self.s3_key = s3_key
#         self.filetype = filetype


def CustomTemporaryFile(*, s3_key: str, filetype: str, **kwargs):
    """
    Extends TemporaryFile with a filetype property

    :param s3_key: key of the file in S3, i.e., the file's name
    :param filetype: filetype of the file, e.g., "csv", "json.gz", "pkl", ...
    :param kwargs: init args for TemporaryFile
    """
    f = TemporaryFile(**kwargs)
    f.s3_key = s3_key
    f.filetype = filetype
    return f


def latest_s3_object(
    *,
    s3_bucket: str,
    s3_key_prefix: str,
    s3_key_suffix: str = "",
    filetypes: Sequence[str] = ("",),
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
) -> Optional[Tuple[Any, str]]:
    """
    finds the latest file matching filetypes and s3_key_prefix in S3_bucket

    :param s3_bucket: source bucket
    :param s3_key_prefix: prefix of object name
    :param s3_key_suffix: suffix of object name
    :param filetypes: decides which file extension should be matched (matches first available filetype)
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    :return: latest object in s3_bucket matching prefix & suffix & filetypes + the filetype that was matched
    :raise FileNotFoundError: if no (matching) file is found
    """
    if s3_client is None:
        # use a session to enable async execution of this function
        s3_session = boto3.session.Session()
        s3_client = s3_session.client("s3")
    # find latest matching file in bucket
    logging.debug(f"Using S3 input bucket {s3_bucket}")
    items = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_key_prefix).get("Contents")
    if not items:
        msg = f"No files for key '{s3_key_prefix}' in '{s3_bucket}'"
        logging.error(msg)
        raise FileNotFoundError(msg)
    # filter all items in right folder and only .csv files
    files: list
    filetype: str
    for filetype in filetypes:
        filetype = filetype.lstrip(".")
        fileext = ".".join(s for s in [s3_key_suffix, filetype] if s)
        files = [item for item in items if item["Key"].startswith(s3_key_prefix) and item["Key"].endswith(fileext)]
        if files:
            break
    else:
        msg = f"No files for key '{s3_key_prefix}' and filetypes '{filetypes}' in '{s3_bucket}'"
        logging.error(msg)
        raise FileNotFoundError(msg)
    files = sorted(files, key=lambda obj: int(obj["LastModified"].strftime("%s")), reverse=True)
    latest_file = files[0]
    logging.info(f"Latest file in '{s3_bucket}' is '{latest_file['Key']}'")
    return latest_file, filetype


def latest_s3_key(
    *,
    s3_bucket: str,
    s3_key_prefix: str,
    s3_key_suffix: str = "",
    filetypes: Sequence[str] = ("",),
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
) -> Optional[Tuple[str, str]]:
    """
    finds the latest file matching filetypes and s3_key_prefix in S3_bucket

    :param s3_bucket: source bucket
    :param s3_key_prefix: prefix of object name
    :param s3_key_suffix: suffix of object name
    :param filetypes: decides which file extension should be matched (matches first available filetype)
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    :return: s3 key ("filename") of the latest object in s3 & the filetype (e.g., "csv")
    :raise FileNotFoundError: if no (matching) file is found
    """
    latest_file, filetype = latest_s3_object(
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_key_prefix,
        s3_key_suffix=s3_key_suffix,
        filetypes=filetypes,
        s3_client=s3_client,
    )
    return latest_file["Key"], filetype


@overload
@contextmanager
def s3_file(
    *,
    s3_bucket: str,
    s3_key: str,
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
    output_dir: Optional[str] = None,
) -> CustomTemporaryFile:
    """
    *to be used as context manager*: downloads the latest file matching filetypes and s3_key_prefix from S3_bucket

    :param s3_bucket: source bucket
    :param s3_key: complete object name, e.g., from latest_s3_key()
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    :param output_dir: where to save the temporary file on rollover (the file becomes too large to be kept in memory)
    :return: yields the file as SpooledTemporaryFile & the filetype (e.g., "csv")
    """
    ...


@overload
@contextmanager
def s3_file(
    *,
    s3_bucket: str,
    s3_key_prefix: str,
    s3_key_suffix: str = "",
    filetypes: Sequence[str] = ("",),
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
    output_dir: Optional[str] = None,
) -> CustomTemporaryFile:
    """
    *to be used as context manager*: downloads the latest file matching filetypes and s3_key_prefix from S3_bucket

    :param s3_bucket: source bucket
    :param s3_key_prefix: prefix of object name
    :param s3_key_suffix: suffix of object name
    :param filetypes: decides which file extension should be matched (matches first available filetype)
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    :param output_dir: where to save the temporary file on rollover (the file becomes too large to be kept in memory)
    :return: yields the file as SpooledTemporaryFile & the filetype (e.g., "csv")
    """
    ...


@contextmanager
def s3_file(
    *,
    s3_bucket: str,
    s3_key: str = None,
    s3_key_prefix: str = None,
    s3_key_suffix: str = "",
    filetypes: Sequence[str] = ("",),
    s3_client: "{list_objects_v2, download_fileobj}" = None,  # noqa
    output_dir: Optional[str] = None,
) -> CustomTemporaryFile:
    """
    *to be used as context manager*: downloads the latest file matching filetypes and s3_key_prefix from S3_bucket

    :param s3_bucket: source bucket
    :param s3_key: complete object name, e.g., from latest_s3_key()
    :param s3_key_prefix: prefix of object name
    :param s3_key_suffix: suffix of object name
    :param filetypes: decides which file extension should be matched (matches first available filetype)
    :param s3_client: boto3 s3 client, uses a thread-safe new s3 session otherwise
    :param output_dir: where to save the temporary file on rollover (the file becomes too large to be kept in memory)
    :return: yields the file as SpooledTemporaryFile & the filetype (e.g., "csv")
    """
    if s3_client is None:
        # use a session to enable async execution of this function
        s3_session = boto3.session.Session()
        s3_client = s3_session.client("s3")
    latest_file: str
    filetype: str
    if s3_key is None:
        # find latest matching file in bucket
        latest_file, filetype = latest_s3_key(
            s3_bucket=s3_bucket,
            s3_key_prefix=s3_key_prefix,
            s3_key_suffix=s3_key_suffix,
            filetypes=filetypes,
            s3_client=s3_client,
        )
    else:
        latest_file = s3_key
        splits: List[str] = s3_key.split(".")
        filetype = splits[-1] if len(splits) > 1 else filetypes[0]
    # download latest cf dump
    # store downloaded files in memory if <= 20 kB
    with CustomTemporaryFile(s3_key=latest_file, filetype=filetype, dir=output_dir) as file:  # , max_size=20 * 1024
        try:
            s3_client.download_fileobj(s3_bucket, latest_file, file)
        except Exception as err:
            # probably a botocore.exceptions.ClientError
            msg = f"Can not load '{s3_bucket}:{latest_file}'."
            logging.error(msg)
            raise RuntimeError(msg) from err
        logging.info(f"Downloaded file '{s3_bucket}:{latest_file}' to '{file}'.")
        file.seek(0)
        yield file
        logging.debug(f"Closing temporary file '{file}'.")


def download_yaml_from_s3(
    *,
    s3_bucket: str,
    s3_key_prefix: str,
    s3_client: "{list_objects_v2, download_file}" = None,  # noqa
) -> Dict[str, Any]:
    """
    downloads the latest CSV / JSON.GZ matching S3_input_key_prefix from S3_input and parses as YAML

    >>> import pa_base.configuration.config.CONFIG_BUCKET
    ... type(
    ...     download_yaml_from_s3(
    ...         yaml_name='cluster_configs.yml',
    ...         s3_bucket=pa_base.configuration.config.CONFIG_BUCKET,
    ...     )
    ... )
    dict

    :param s3_bucket: source bucket
    :param s3_key_prefix: prefix of YAML file
    :param s3_client: boto3 s3 client, uses a new s3 session otherwise
    :return: yields the file as SpooledTemporaryFile
    """
    if s3_client is None:
        # use a session to enable async execution of this function
        session = boto3.session.Session()
        s3_client = session.client("s3")
    with s3_file(
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_key_prefix,
        filetypes=["yml", "yaml"],
    ) as file:
        # read downloaded file
        return yaml.safe_load(file)


def download_dataframe_from_s3(
    *,
    s3_bucket: str,
    s3_key_prefix: str,
    s3_key_suffix: str = "",
    usecols: List[str] = None,
    datecols: List[str] = None,
    date_format: str = None,
    transform_df: Callable[[pd.DataFrame, str], pd.DataFrame] = lambda df, _: df,
    s3_client: "{list_objects_v2, download_file}" = None,  # noqa
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    downloads the latest CSV / JSON.GZ matching S3_input_key_prefix from S3_input and parses as DataFrame

    **JSON.GZ is preferred over CSV**

    :param s3_bucket: source bucket
    :param s3_key_prefix: prefix of CSV / JSON.GZ file
    :param s3_key_suffix: suffix of CSV / JSON.GZ file (prefix is by far preferred as it's natively supported in S3)
    :param usecols: columns expected in dataframe
    :param datecols: columns parsed as dates
    :param date_format: date format such as "%Y-%m-%d %H:%M:%S.%f %Z" if known, None otherwise. May improve performance.
    :param transform_df: a function applying arbitrary operations on the df directly after loading: (df, filetype)->df
    :param s3_client: boto3 s3 client, uses a new s3 session otherwise
    :param output_dir: where to save the temporary file on rollover (the file becomes too large to be kept in memory)
    :return: yields the file as SpooledTemporaryFile
    """
    if s3_client is None:
        # use a session to enable async execution of this function
        s3_session = boto3.session.Session()
        s3_client = s3_session.client("s3")
    with s3_file(
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        s3_key_prefix=s3_key_prefix,
        s3_key_suffix=s3_key_suffix,
        filetypes=["json.gz", "csv", "parquet", "parquet.gz"],
        output_dir=output_dir,
    ) as file:
        filetype = file.filetype
        # read downloaded file
        df: Optional[pd.DataFrame] = None
        if filetype == "json.gz":
            with gzip.open(file, "rb") as input_json:
                # Manually parse lines as json and use from_records, because this handles columns missing in some lines.
                # pd.read_json ignores these columns.
                # pd.concat(map(pd.read_json(...), input_json), axis=1).transpose() is _very_ slow.
                df = pd.DataFrame.from_records(
                    map(lambda line: json.loads(line), input_json),
                    columns=usecols,
                )
        elif filetype == "csv" or filetype == "csv.gz":
            df = pd.read_csv(
                file,
                sep=",",
                # use lambda for usecols to handle missing cols robustly
                usecols=lambda col: (usecols is None) or (col in usecols),
                parse_dates=datecols,
                date_parser=lambda x: pd.to_datetime(
                    x,
                    utc=True,
                    infer_datetime_format=True,
                    format=date_format,
                    errors="coerce",
                ).tz_localize(None),
            )
        elif filetype == "parquet":
            df = pd.read_parquet(
                file,
                columns=usecols,
            )
        elif filetype == "parquet.gz":
            with gzip.open(file, "rb") as gzfile:
                df = pd.read_parquet(
                    gzfile,
                    columns=usecols,
                )
        else:
            raise ValueError(f"Cannot load filetype '{filetype}'.")
    df = transform_df(df, filetype)
    if filetype in {"json.gz", "parquet", "parquet.gz"}:
        # parse dates and convert to utc naive
        # only needed for json.gz, because pd.read_csv already does this at parsing time and parquet has date format built in
        if isinstance(datecols, list) and len(datecols):
            df[datecols] = df[datecols].apply(
                lambda x: pd.to_datetime(
                    x,
                    utc=True,
                    infer_datetime_format=True,
                    format=date_format,
                    errors="coerce",
                ).dt.tz_localize(None)
            )
    logging.info(f"Raw row count in dataframe: {len(df)}")

    with suppress(Exception):
        col_diff = set(usecols) - set(df.columns)
        if col_diff:
            logging.warning(f"Missing columns: {col_diff}")
        else:
            logging.info("Dataframe contains all expected columns.")

    # ensure column order with reindex
    return df.reindex(columns=usecols)
