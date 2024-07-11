# Copyright (c) 2024, ZDF.
"""
ZDF-specific dataframes.
"""
import glob
import json
import logging
import os
import shutil
import typing as t
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from pa_base.configuration.config import (
    CLICKSTREAM_BUCKET_NAME,
    CONTENT_SYNC_BUCKET_NAME,
    CONTENT_SYNC_CAPTIONS_PREFIX,
    CONTENT_SYNC_DUMP_PREFIX,
    CONTENTPOOL_INPUT_BUCKET_NAME,
    DESCRIPTIVE_METADATA_SYNC_BUCKET_NAME,
    DESCRIPTIVE_METADATA_SYNC_DUMP_PREFIX,
    S3_PREPROCESSED_INTERACTIONS,
    TZ,
)
from pa_base.data.s3_util import (
    download_dataframe_from_s3,
    latest_s3_key,
    s3_file,
    s3_upload_file,
)
from pa_base.zdf.configuration.config import (
    KIKA_PARTNER_TAG,
    KIKA_VORSCHULE_TAG,
    ZDF_CONTENT_DUMP_BOOLCOLS,
    ZDF_CONTENT_DUMP_CATCOLS,
    ZDF_CONTENT_DUMP_COLS,
    ZDF_CONTENT_DUMP_DATECOLS,
    ZDF_CONTENT_DUMP_INTCOLS,
    ZDF_CONTENT_DUMP_STRCOLS,
)
from pa_base.zdf.models.util import get_model_target

try:
    from pa_base.zdf.configuration.dynamic_configs import (
        DESCRIPTIVE_METADATA_COLUMNS,
        DESCRIPTIVE_METADATA_DICT_COLUMNS,
        PLAY_COVERAGE_THRESHOLD,
        PLAY_MAX_AGE_DAYS,
    )
except ImportError as exc:
    logging.error(
        "Could not import dynamic_configs, setting 'empty defaults' for PLAY_COVERAGE_THRESHOLD, PLAY_MAX_AGE_DAYS, DESCRIPTIVE_METADATA_COLUMNS, and, DESCRIPTIVE_METADATA_DICT_COLUMNS.",
        exc_info=exc,
    )
    PLAY_COVERAGE_THRESHOLD = 0
    PLAY_MAX_AGE_DAYS = 0
    DESCRIPTIVE_METADATA_COLUMNS = []
    DESCRIPTIVE_METADATA_DICT_COLUMNS = []


def get_denoised_data(
    *,
    use_ids_data=False,
    use_cl_data=True,
    cl_n_days=14,
    cl_n_days_offset=0,
    cl_frontend="pctablet",
    ids_n_days=PLAY_MAX_AGE_DAYS,
    count_threshold=20,
    user_count_threshold=None,
    coverage_threshold=PLAY_COVERAGE_THRESHOLD,
    pseudonymize=False,
    upload_S3=False,
    tmpdir="/tmp",
    cache_dir=".",
):
    """
    get data from ids and/or cl, denoise and sort it (timestamps asc)

    :param use_ids_data: whether dumped ids data should be considered
    :param use_cl_data: whether ZDFtracking data should be considered
    :param cl_n_days: number of days to go back in time (now - X days) for cl_data
    :param cl_frontend: which frontend's data should be considered (tv, mobile, pctablet),
        can either be a single frontend (str), or a List[str] or comma-separated str for multiple frontends
    :param ids_n_days: number of days to go back in time (now - X days) for ids_data
    :param count_threshold: minimum number of interactions per item to be included
    :param user_count_threshold: minimum number of interactions per user to be included
    :param coverage_threshold: minimum coverage (progress/duration) for an interaction to be included
    :param pseudonymize: ``True`` if trackingIds should be replaced by pseudonyms
    :param upload_S3: ``True`` if resulting dataframe should be uploaded to S3
    :param tmpdir: writable and existing directory where the resulting dataframe can be stored for S3 upload
    :param cache_dir:  writable and existing directory where temporary data (ids_data, cl_data) can be downloaded to
    :return: dataframe with one interaction per row
    """
    if not use_ids_data and not use_cl_data:
        return None
    if isinstance(cl_frontend, str):
        cl_frontend = cl_frontend.split(",")
    elif not isinstance(cl_frontend, list):
        raise ValueError("cl_frontend has to be a (comma-separated) str or list")

    # get usage data, note: all timestamps in IDS and CL data are UTC!
    now_local = datetime.now(TZ)
    data_ids: t.Optional[pd.DataFrame] = None
    if use_ids_data and ids_n_days:
        raise NotImplementedError("IDS data is currently not supported.")
        # play_min_ids_lt = now_local - timedelta(days=ids_n_days)
        # data_ids = get_ids_data_df()
        # logging.debug(f"IDS raw row count {len(data_ids)}")
        # # get content metadata
        # metadata = get_content_df()
        # data_ids = data_ids.merge(
        #     metadata[["video_duration", "path_level_1"]],
        #     on="externalid",
        #     how="left",
        #     copy=False,
        # )
        # del metadata
        # logging.log(LOGLEVEL_VERBOSE, f"IDS raw row count after merge {len(data_ids)}")
        # logging.log(LOGLEVEL_VERBOSE, f"IDS nulls {data_ids.isnull().sum()}")
        # data_ids.dropna(inplace=True)
        # logging.log(LOGLEVEL_VERBOSE, f"IDS raw row count after dropna {len(data_ids)}")
        # logging.log(LOGLEVEL_VERBOSE, "Metadata merged for IDS data")
        # # denoising #1: skip items with low coverage (percentage video watched)
        # data_ids[
        #     "coverage"
        # ] = data_ids.current_position / data_ids.video_duration.astype(np.float32)
        # data_ids = data_ids[data_ids.coverage > coverage_threshold]
        # logging.log(
        #     LOGLEVEL_VERBOSE,
        #     f"Low coverage (below {coverage_threshold}) removed from IDS data. "
        #     f"Row count {len(data_ids)}",
        # )
        # # denoising #2: remove old entries
        # data_ids["datetime_local"] = data_ids.timestamp.apply(
        #     lambda x: datetime.fromtimestamp(x, TZ)
        # )
        # data_ids["date"] = data_ids.datetime_local.dt.date
        # data_ids = data_ids[data_ids.datetime_local > play_min_ids_lt]
        # logging.log(
        #     LOGLEVEL_VERBOSE,
        #     f"Old entries (more than {ids_n_days} days old removed from IDS data. "
        #     f"Row count {len(data_ids)}",
        # )
        # # denoising #3: remove bogus new entries (timestamp in future)
        # data_ids = data_ids[data_ids.datetime_local < now_local]
        # logging.log(
        #     LOGLEVEL_VERBOSE,
        #     "Bogus entries (in the future) removed from IDS data. "
        #     f"Row count {len(data_ids)}",
        # )
        # # cleanup dataframe
        # data_ids.drop(["current_position", "video_duration"], inplace=True, axis=1)
        # data_ids.rename(columns={"path_level_1": "genre"}, inplace=True)
        # logging.debug(f"IDS denoised row count {len(data_ids)}")

    data_cl: t.Optional[pd.DataFrame] = None
    if use_cl_data and cl_n_days:
        play_min_cl_lt = now_local - timedelta(days=(cl_n_days + cl_n_days_offset))
        data_cl = pd.DataFrame()
        for frontend in cl_frontend:
            logging.info(f"Loading {cl_n_days} days' data for target: '{frontend}'.")
            single_frontend_cl = get_agg_cl_data_df(
                n_days=cl_n_days,
                frontend=frontend,
                cache_dir=cache_dir,
                offset_days=cl_n_days_offset,
            )
            if single_frontend_cl.empty:
                logging.info("No cl data found")
                continue
            # cleanup dataframe (NOTE re-add appid when IDS supports also)
            single_frontend_cl.drop(["viewing_minutes", "appid"], inplace=True, axis=1)
            # get timestamp as column
            single_frontend_cl.reset_index(inplace=True)
            logging.info(f"CL raw row count {len(single_frontend_cl)}")
            single_frontend_cl.drop_duplicates(subset=["externalid", "uid"], keep="first", inplace=True)
            logging.info(f"Duplicates removed. Row count {len(single_frontend_cl)}")
            # denoise #1: skip items with low coverage
            single_frontend_cl = single_frontend_cl[single_frontend_cl.coverage > coverage_threshold]
            logging.info(f"Low coverage < {coverage_threshold} removed. Row count {len(single_frontend_cl)}")
            # denoising #2: remove old entries
            # single_frontend_cl["datetime_local"] = single_frontend_cl.timestamp.apply(lambda x: datetime.fromtimestamp(x, TZ))
            single_frontend_cl["datetime_local"] = pd.to_datetime(
                single_frontend_cl.timestamp, unit="s", utc=True
            ).dt.tz_convert(TZ)
            logging.info("Calculated datetimes.")
            single_frontend_cl["date"] = single_frontend_cl.datetime_local.dt.date
            logging.info("Calculated dates.")
            single_frontend_cl = single_frontend_cl[single_frontend_cl.datetime_local > play_min_cl_lt]
            logging.info(f"Old entries > {cl_n_days} removed. Row count {len(single_frontend_cl)}")
            # denoising #3: remove bogus new entries (timestamp in future)
            single_frontend_cl = single_frontend_cl[single_frontend_cl.datetime_local < now_local]
            logging.info(f"Bogus entries (in the future) removed. Row count {len(single_frontend_cl)}")
            logging.info(f"CL denoised row count {len(single_frontend_cl)}")
            logging.info(f"Concatenating data for target '{frontend}' to dataframe.")
            data_cl = pd.concat(
                [data_cl, single_frontend_cl],
                axis=0,
                sort=False,
                copy=True,  # old pandas bug with TZ-aware timestamp https://github.com/pandas-dev/pandas/issues/25257
            )
            logging.info(f"Concatenated data for target '{frontend}' to dataframe.")

    # patch together data
    if (data_ids is not None) and (data_cl is not None):
        data = pd.concat([data_ids, data_cl], ignore_index=True, sort=False, copy=False)
    elif data_ids is not None:
        data = data_ids
    elif data_cl is not None:
        data = data_cl
    else:
        raise ValueError("Neither data_ids nor data_cl available.")
    # reduce size in memory
    data["genre"] = data["genre"].astype("category")
    data["externalid"] = data["externalid"].astype("category")
    data["uid"] = data["uid"].astype("category")
    logging.info(
        f"Aggregate denoised row count {len(data)} prior to thresholding.",
    )
    # denoising #4: filter items with less than 'threshold' appearances
    # data = data[
    #     data["externalid"].groupby(data["externalid"]).transform("size")
    #     >= count_threshold
    # ]
    data = data[data.externalid.map(data.externalid.value_counts(sort=False) >= count_threshold)]
    logging.info(
        f"Aggregate denoised row count {len(data)} after thresholding item count",
    )
    # reduce size in memory
    data["genre"] = data["genre"].cat.remove_unused_categories()
    data["externalid"] = data["externalid"].cat.remove_unused_categories()
    data["uid"] = data["uid"].cat.remove_unused_categories()
    # denoising #5 (optional): filter users with less than 'user_count_threshold' items
    if user_count_threshold:
        # data = data[
        #     data["uid"].groupby(data["uid"]).transform("size") >= user_count_threshold
        # ]
        data = data[data.uid.map(data.uid.value_counts(sort=False) >= user_count_threshold)]
    logging.info(
        f"Aggregate denoised row count {len(data)} after thresholding user item count",
    )
    # reduce size in memory
    data["genre"] = data["genre"].cat.remove_unused_categories()
    data["externalid"] = data["externalid"].cat.remove_unused_categories()
    data["uid"] = data["uid"].cat.remove_unused_categories()

    # sort by uid, timestamp
    logging.info("Sorting usage by timestamp")
    data.sort_values(by=["timestamp"], ascending=True, inplace=True)

    # PERAUT-1254 transformed externalid & uid to categorical
    #   this could potentially lead to problems in consumers expecting str
    #   which could be fixed by the following lines:
    #   data["externalid"] = data["externalid"].astype(str)
    #   data["uid"] = data["uid"].astype(str)

    # pseudonymizing
    if pseudonymize:
        logging.info("Pseudonymizing usage data")
        data["uid"] = data["uid"].astype("category").cat.codes

    logging.info(f"Final row count in denoised clickstream data is {len(data)}.")

    if upload_S3:
        timestamp = TZ.localize(datetime.now()).strftime("%Y%m%d%H%M")
        output_file = f"{tmpdir}/{timestamp}_interactions.csv.gz"
        data.to_csv(output_file, compression="gzip", header=True, index=False)
        logging.info("Start S3 upload.")
        s3_upload_file(
            filepath=output_file,
            s3_bucket=S3_PREPROCESSED_INTERACTIONS,
            s3_key_prefix=os.path.basename(output_file),
        )
        logging.info("S3 upload finished.")
        os.remove(output_file)
    return data


def get_content_df(
    s3_bucket: str = "",
    s3_key: str = "",
    usecols: t.List[str] = ZDF_CONTENT_DUMP_COLS,
    datecols: t.List[str] = ZDF_CONTENT_DUMP_DATECOLS,
    catcols: t.List[str] = ZDF_CONTENT_DUMP_CATCOLS,
    intcols: t.List[str] = ZDF_CONTENT_DUMP_INTCOLS,
    strcols: t.List[str] = ZDF_CONTENT_DUMP_STRCOLS,
    boolcols: t.List[str] = ZDF_CONTENT_DUMP_BOOLCOLS,
) -> pd.DataFrame:
    """get latest content as pandas dataframe"""
    df = download_dataframe_from_s3(
        s3_bucket=s3_bucket or CONTENT_SYNC_BUCKET_NAME,
        s3_key_prefix=s3_key or CONTENT_SYNC_DUMP_PREFIX,
        usecols=usecols,
        datecols=datecols,
    )
    # optimize memory usage by marking columns as bool
    df[boolcols] = df.loc[:, boolcols].astype("bool")
    # optimize memory usage by marking columns as categorical
    df[catcols] = df.loc[:, catcols].fillna("").astype("category")
    # optimize memory usage by marking columns as small ints
    df[intcols] = df.loc[:, intcols].astype("int16")
    # fill NAs
    df.loc[:, strcols] = df.loc[:, strcols].fillna("")
    df.loc[:, "airtimebegins"] = (
        df.loc[:, "airtimebegins"]
        .fillna("")
        .str.split(",")
        .map(
            lambda airtimebegins: {pd.to_datetime(x, errors="coerce").tz_convert(None) for x in airtimebegins}
            if airtimebegins is not None
            else set()
        )
    )
    # extent content with index columns for kika tags
    df["has_kika_partner_tag"] = (
        df["editorial_tags"].str.split(",", expand=False).map(lambda item_tags: KIKA_PARTNER_TAG in item_tags)
    )
    df["has_kika_vorschule_tag"] = (
        df["editorial_tags"].str.split(",", expand=False).map(lambda item_tags: KIKA_VORSCHULE_TAG in item_tags)
    )

    # deserialize json columns
    for column in ["zdfinfo_metadata", "search_service_tagging_results"]:
        df.loc[:, column] = df.loc[:, column].map(json.loads)
    df.set_index("externalid", inplace=True, drop=False)
    return df


def get_descriptive_metadata_df(
    *,
    s3_bucket: str = DESCRIPTIVE_METADATA_SYNC_BUCKET_NAME,
    s3_key: str = DESCRIPTIVE_METADATA_SYNC_DUMP_PREFIX,
    usecols: t.List[str] = DESCRIPTIVE_METADATA_COLUMNS,
    dictcols: t.List[str] = DESCRIPTIVE_METADATA_DICT_COLUMNS,
) -> pd.DataFrame:
    """Download latest descriptive metadata content dump.

    :param s3_bucket: The S3 bucket where the descriptive metadata content is located.
    :param s3_key: The filename (key) of the descriptive metadata content in the `s3_bucket`.
    :returns: A dataframe that contains the descriptive metadata content.
    """
    logging.info(f"Loading descriptive metadata from S3 '{s3_bucket}:{s3_key}'.")

    df = download_dataframe_from_s3(s3_bucket=s3_bucket, s3_key_prefix=s3_key, usecols=usecols)

    # TODO [CM]: Move this to utility file?
    def str_to_dict(item):
        # Only deserialise columns that are not empty (i.e pd.NA),
        # otherwise JSON raises an exception.
        return item if pd.isna(item) else json.loads(item)

    # Deserialize json columns.
    for column in dictcols:
        df.loc[:, column] = df.loc[:, column].map(str_to_dict)

    # Rename column externalId to externalid so that it is consistent with
    # that of the content pool.
    df.rename(columns={"externalId": "externalid"}, inplace=True)

    # Set externalid as the index.
    df.set_index("externalid", inplace=True, drop=False)

    return df


def get_fsdb_matches_df() -> pd.DataFrame:
    """
    download fsdb items that have been matched to the content (i.e., added externalid column)
    :return: DataFrame with externalid + FSDB columns, e.g., to be left-merged on externalid into content
    """

    def transform_df(df: pd.DataFrame, _: str) -> pd.DataFrame:
        """
        download_dataframe_from_s3 expects this function signature but we don't care about
        the file type passed as second parameter

        :param df: dataframe loaded by download_dataframe_from_s3
        :param _: discarded (file type of source in S3)
        :return: transformed df
        """
        df.rename(
            columns={
                "externalId": "externalid",
                "clipArchNr": "clip_archnr",
                "clipDeskripts": "clip_deskripts",
                "clipKats": "clip_kats",
                "clipUtContent": "clip_ut_content",
                "clipVisualContent": "clip_visual_content",
                "prodNr": "prodNr_y",
                # "title" missing?
            },
            inplace=True,
        )
        return df

    return download_dataframe_from_s3(
        s3_bucket=CONTENTPOOL_INPUT_BUCKET_NAME,
        s3_key_prefix="fsdb_matched",
        # usecols=[
        #     "externalId",
        #     "clipArchNr",
        #     "clipDeskripts",
        #     "clipKats",
        #     "clipUtContent",
        #     "clipVisualContent",
        #     "prodNr",
        # ],
        transform_df=transform_df,
    )


def get_agg_cl_data_df(
    *,
    frontend="pctablet",
    n_days=1,
    appid_prefix=None,
    cache_dir="/data/cache/tmp/",
    offset_days=0,
) -> pd.DataFrame:
    """
    get 'n_days' of aggregated clickstream data from 'frontend'

    :param frontend: load only interactions on that frontend (pctablet, mobile, tv)
    :param n_days: load interactions from n past days
    :param appid_prefix: OPTIONAL load only interactions prefix-matching this appid
    :param cache_dir: local cache dir for downloading the csv
    :param offset_days: OPTIONAL load only n_days-offset_days total days, but offset from today by offset_days
    :return: pandas dataframe with one row per interaction
    """
    assert n_days > 0
    assert offset_days < n_days
    # check for the latest aggregated data packages in S3 bucket
    s3_prefixes = []
    for day in range(offset_days, n_days):
        date = datetime.now() - timedelta(days=day)
        s3_prefix_base = f"{frontend}_1d_aggregate"
        s3_prefixes.append("{}/{}".format(s3_prefix_base, date.strftime("%Y/%-m/%-d")))
    s3_keys = []
    # get all .csv files from S3 and create dataframes for each day
    usecols = [
        "uid",
        "externalid",
        "genre",
        "appid",
        "timestamp",
        "coverage",
        "viewing_minutes",
    ]
    dtypes = {
        # doesn't work with categoricals (probably because of too old pandas version)
        # "uid": pd.CategoricalDtype,
        # "externalid": pd.CategoricalDtype,
        # "genre": pd.CategoricalDtype,
        # "appid": pd.CategoricalDtype,
        # timestamps are dtyped later to have the possibility to detect underflows
        # "timestamp": np.uint32,
        "coverage": np.float32,
        "viewing_minutes": np.uint32,
    }
    df = pd.DataFrame()
    success_counter = 0
    dump_files = []
    for s3_key_index, prefix in enumerate(s3_prefixes):
        logging.info(
            f"Try to load interactions from '{CLICKSTREAM_BUCKET_NAME}'.",
            extra={"prefix": prefix},
        )
        # find latest s3 object matching prefix
        try:
            s3_key, _ = latest_s3_key(
                s3_bucket=CLICKSTREAM_BUCKET_NAME,
                s3_key_prefix=prefix,
                filetypes=("csv",),
            )
        except FileNotFoundError as exc:
            logging.warning(
                f"Cannot find an S3 key for '{CLICKSTREAM_BUCKET_NAME}:{prefix}*.csv'. Skipping (index={s3_key_index}).",
                exc_info=exc,
            )
            # just ignore missing interactions for one day
            continue
        s3_keys.append(s3_key)
        # transform s3 key to valid filename & path
        dump_file = os.path.join(cache_dir, s3_key.replace("/", "_"))
        # download if it is the first (=newest) aggregate or file is not in cache
        if s3_key_index == 0:
            logging.info("Only using tempfile for first index.")
            # don't keep 'latest' file as it changes throughout a single day (deletes automatically on close)
            try:
                with s3_file(s3_bucket=CLICKSTREAM_BUCKET_NAME, s3_key=s3_key) as remote_file:
                    dataframe = pd.read_csv(
                        remote_file,
                        index_col="timestamp",
                        usecols=usecols,
                        dtype=dtypes,
                        memory_map=True,
                    )
            except FileNotFoundError as exc:
                logging.warning(
                    f"Error loading S3 file for '{CLICKSTREAM_BUCKET_NAME}:{s3_key}'. Skipping (index=0).",
                    exc_info=exc,
                )
                # just ignore missing interactions for "today" -> often missing shortly after midnight
                continue
        else:
            # keep files on disk as older days no longer change
            dump_files.append(dump_file)
            if not os.path.exists(dump_file):
                logging.info(f"Permanently keeping file at '{dump_file}'.")
                try:
                    with open(dump_file, "wb") as local_file, s3_file(
                        s3_bucket=CLICKSTREAM_BUCKET_NAME, s3_key=s3_key
                    ) as remote_file:
                        # keep on disk after tempfile has been deleted
                        shutil.copyfileobj(remote_file, local_file)
                except RuntimeError as err:
                    logging.warning(f"Can not load '{s3_key}', reason {err}.")
                    continue
            logging.debug(f"Using file from disk at '{dump_file}'.")
            dataframe = pd.read_csv(
                dump_file,
                index_col="timestamp",
                usecols=usecols,
                dtype=dtypes,
                memory_map=True,
            )
        # filter and cast timestamp column (used as index)
        dataframe = dataframe[dataframe.index.notnull()]
        if any(dataframe.index < 0):
            raise ValueError("get_agg_cl_data_df only works for positive timestamps")
        # dataframe = dataframe[dataframe.index >= 0]
        dataframe.index = dataframe.index.astype(np.uint32)
        # capping to max value (1.)
        dataframe.loc[dataframe.coverage > 1, "coverage"] = 1.0
        # save memory by using categories, since there are many distinct uids just keep them as str (object)
        # dataframe.uid = dataframe.uid.astype("category")
        dataframe.externalid = dataframe.externalid.astype("category")
        dataframe.genre = dataframe.genre.astype("category")
        dataframe.appid = dataframe.appid.astype("category")
        # filter appid prefixes
        if appid_prefix:
            dataframe = dataframe.loc[dataframe.appid.str.startswith(appid_prefix)]
        if not df.empty:
            # concat transforms categories back to object if they differ -> make all categories equal in both dataframes
            for col in dataframe.select_dtypes(include="category").columns:
                all_cats = df[col].cat.categories.union(dataframe[col].cat.categories)
                df[col] = df[col].cat.set_categories(all_cats)
                dataframe[col] = dataframe[col].cat.set_categories(all_cats)
        df = pd.concat([df, dataframe], axis=0, sort=False, copy=False)
        success_counter += 1
    # cleanup cache - delete any file not in dump_files
    cache_files = glob.glob(f"{cache_dir}/*.csv")
    remove_files = [fn for fn in cache_files if fn not in dump_files]
    for fn in remove_files:
        os.remove(fn)
    logging.info(
        f"Loaded aggregated clickstream data of {success_counter} days. "
        f"Requested: {n_days}-{offset_days}={n_days - offset_days} days."
    )
    return df


def get_agg_cl_short_term_most_viewed_df() -> pd.DataFrame:
    """
    get short-term most-viewed of aggregated clickstream data for top-5 sorted by count_distinct_views

    :return: a dataframe with the following columns:
        "externalid", "id", "path", "video_type", "count_distinct_views", "target"
    """
    # check for the latest aggregated data packages in S3 bucket
    s3_prefixes = []
    n_days = 2  # yesterday and today to always have enough for 24h top-5
    for day in range(n_days):
        date = datetime.now() - timedelta(days=day)
        s3_prefix_base = "top5"
        s3_prefixes.append("{}/{}".format(s3_prefix_base, date.strftime("%Y/%-m/%-d")))
    usecols = [
        "externalId",  # renamed to externalid later for consistency with other dataframes
        "id",
        "path",
        "video_type",
        "count_distinct_views",
        "watched_date",
    ]
    # get all .csv files from S3 and create dataframes for each day
    daily_dfs = []
    for prefix_index, prefix in enumerate(s3_prefixes):
        try:
            logging.debug(f"Loading short-term most-viewed from S3 '{CLICKSTREAM_BUCKET_NAME}:{prefix}'.")
            # convert watched_date to naive utc to work around a bug in concat with tzinfo
            dataframe = download_dataframe_from_s3(
                s3_bucket=CLICKSTREAM_BUCKET_NAME,
                s3_key_prefix=prefix,
                usecols=usecols,
                datecols=["watched_date"],
                # specifying "format" additionally to "infer_datetime_format=True" reduces runtime from 120s to 4s
                date_format="%Y-%m-%d %H:%M:%S.%f %Z",
            )
            dataframe.rename(columns={"externalId": "externalid"}, inplace=True)
            dataframe.set_index("externalid", inplace=True, drop=False)
            # convert viewing minutes
            dataframe.count_distinct_views = dataframe.count_distinct_views.astype(np.float32)
            # drop all rows containing nans
            dataframe.dropna(axis=0, inplace=True)
            daily_dfs.append(dataframe)
        except FileNotFoundError as exc:
            logging.warning("Error loading one shard of short-term most-viewed data.", exc_info=exc)
    logging.info(f"Loaded short-term most-viewed data of {len(daily_dfs)} days. Requested: {n_days} days.")
    if not daily_dfs:
        logging.info("Short-term most-viewed data is empty.")
        return pd.DataFrame(columns=usecols + ["target"])
    df: pd.DataFrame = pd.concat(daily_dfs, axis=0, sort=False, copy=False)
    # add target links
    logging.debug("Adding target column to bigquery most-viewed")
    df["target"] = "https://www.zdf.de" + df["path"].astype(str) + "/" + df["id"] + ".html"
    logging.debug("Sorting bigquery most-viewed")
    df.sort_values(by="count_distinct_views", ascending=False, inplace=True)
    return df


def get_captions_df() -> pd.DataFrame:
    """
    downloads subtitles from S3

    :return: a dataframe containing "externalid", "caption" columns, indexed by externalid and filtered for German
    """
    usecols = ["externalId", "language", "caption"]
    df: pd.DataFrame = download_dataframe_from_s3(
        s3_bucket=CONTENT_SYNC_BUCKET_NAME,
        s3_key_prefix=CONTENT_SYNC_CAPTIONS_PREFIX,
        usecols=usecols,
    )
    df = df[df["language"] == "deu"]
    df.drop(columns=["language"], inplace=True)
    df.rename(columns={"externalId": "externalid"}, inplace=True)
    df.set_index(df["externalid"].values, drop=False, inplace=True)
    df["caption"] = df["caption"].str.replace("\n", " ")
    return df


def get_interactions_df(tmpdir="/tmp"):
    """get latest interactions dump as pandas dataframe"""

    def timestamp_to_int(df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce", downcast="integer")
        return df

    interactions: pd.DataFrame = download_dataframe_from_s3(
        s3_bucket=S3_PREPROCESSED_INTERACTIONS,
        s3_key_prefix="",
        s3_key_suffix="_interactions.csv.gz",
        usecols=["externalid", "uid", "timestamp"],
        transform_df=timestamp_to_int,
        output_dir=tmpdir,
    )
    logging.info("Found %i interactions.", len(interactions))
    return interactions


def get_recos_data() -> pd.DataFrame:
    """
    get recommendations data

    :return: a dataframe with the following columns:
        "recoId", "configuration", "abGroup", "appId", "assetId", "pageId", "date", "views", "plays", "bookmarks", "recommendations", "targetAssetid", "trackingId"
    """
    # check for the latest aggregated data packages in S3 bucket

    # most current file always covers 1h
    s3_prefix_base = "reco_recoId"
    s3_prefix = "{}/{}".format(s3_prefix_base, "recoId_export_latest.json.gz")

    # expected columns in df
    usecols = [
        "recoId",
        "configuration",
        "abGroup",
        "appId",
        "assetId",
        "pageId",
        "position",
        "date",
        "views",
        "plays",
        "bookmarks",
        "recommendations",
        "refDocId",
        "targetAssetId",
        "trackingId",
        "userSegmentId",
        "target",
        "playbacktimeInMin",
        "duration",
        "coverage",
    ]

    # get all .json.gz files from S3 and create dataframes for each day
    logging.info(f"Loading recommendations data from S3 '{CLICKSTREAM_BUCKET_NAME}:{s3_prefix}'.")

    dataframe = download_dataframe_from_s3(
        s3_bucket=CLICKSTREAM_BUCKET_NAME,
        s3_key_prefix=s3_prefix,
        usecols=usecols,
    )

    if dataframe is None:
        return pd.DataFrame(columns=usecols)

    # replace NaN with ""
    dataframe = dataframe.fillna("")
    # index by recoId
    dataframe.set_index("recoId", inplace=True, drop=False)
    # add the target column
    dataframe["target"] = dataframe["appId"].apply(lambda x: get_model_target(x, None))

    logging.info(
        f"Loaded recommendations data {len(dataframe)} points from S3 '{CLICKSTREAM_BUCKET_NAME}:{s3_prefix}'."
    )
    return dataframe
