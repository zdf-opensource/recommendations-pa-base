# Copyright (c) 2023, ZDF.
"""
Static configuration and evironment variables.
"""

import logging
import os

import pytz

# Site mandatory!
SITE = os.environ.get("SITE") or "undefined"
if SITE == "undefined":
    raise ValueError("SITE needs to be defined")
_valid_sites = {"test", "mig", "dev", "int", "prod"}
if SITE not in _valid_sites:
    raise ValueError(f"SITE needs to be one of '{_valid_sites}'")


# timezone
TZ = pytz.timezone("Europe/Berlin")

# content pool local file locations (for models and data)
CP_LOCAL_DATA_PATH = "."

####
# BEGIN Bucket Config Section
_S3_BASENAME = os.environ.get("S3_BASENAME", "de.zdf.recos.v2")

# Config Bucket
CONFIG_BUCKET = os.environ.get("CONFIG_BUCKET", f"{_S3_BASENAME}.config.{SITE}")

# S3 bucket with aggregated clickstream data - only in new account structure
CLICKSTREAM_BUCKET_NAME = os.environ.get(
    "CLICKSTREAM_BUCKET_NAME",
    f"{_S3_BASENAME}.clickstream.preprocessed.{SITE}",
)

# bucket for other inputs, such as fsdb dumps
CONTENTPOOL_INPUT_BUCKET_NAME = f"{_S3_BASENAME}.contentpool.input.{SITE}"

# bucket & prefixes of content sync in S3
CONTENT_SYNC_BUCKET_NAME = f"{_S3_BASENAME}.content.sophora.{SITE}"
CONTENT_SYNC_PREFIX = f"sophora_content_{SITE}_latest"
CONTENT_SYNC_BRANDS_PREFIX = f"sophora_brands_{SITE}_latest"
CONTENT_SYNC_CAPTIONS_PREFIX = f"captions_{SITE}_latest"

# buckets for preprocessed etl output
S3_PREPROCESSED_METADATA = f"{_S3_BASENAME}.content.preprocessed.metadata.{SITE}"
S3_PREPROCESSED_METADATA_TEST = f"{_S3_BASENAME}.content.preprocessed.metadata.test"
S3_PREPROCESSED_METADATA_MIG = f"{_S3_BASENAME}.content.preprocessed.metadata.mig"
S3_PREPROCESSED_SEQUENCES = f"{_S3_BASENAME}.content.preprocessed.sequences.{SITE}"
S3_PREPROCESSED_INTERACTIONS = (
    f"{_S3_BASENAME}.content.preprocessed.interactions.{SITE}"
)

# buckets for models
# content-based
S3_MODELS_VISUAL = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_TEXT = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_TRENDING = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_MOST_VIEWED = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_TOP5 = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_LAST_CHANCE = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_PREVIEW = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_CONTENT_CLUSTERS = f"{_S3_BASENAME}.models.content.{SITE}"
S3_MODELS_ZDFINFO_TOPIC_CLUSTERS = f"{_S3_BASENAME}.models.content.{SITE}"
# usage-based
S3_MODELS_COLLABORATIVE = f"{_S3_BASENAME}.models.collaborative.{SITE}"
S3_MODELS_NCF = ""
S3_MODELS_SEQUENTIAL = f"{_S3_BASENAME}.models.sequential.{SITE}"
S3_MODELS_SAR = f"{_S3_BASENAME}.models.rl.{SITE}"
S3_MODELS_CMAB = f"{_S3_BASENAME}.models.rl.{SITE}"
# END Bucket Config Section
####

# Default AWS Region (if nothing else specified)
DEFAULT_REGION = "eu-central-1"

# Default Redismodel port (if nothing else specified)
DEFAULT_REDISMODEL_PORT = "6379"


# Redis
REDIS_HOST = os.environ.get("REDIS_HOST", "")
REDIS_PORT = os.environ.get("REDIS_PORT", "")

# Redis Fast-Next-Video
REDISFNV_HOST = os.environ.get("REDISFNV_HOST", "")
REDISFNV_PORT = os.environ.get("REDISFNV_PORT", "")
REDISFNV_TOKEN = os.environ.get("REDISFNV_TOKEN", "")

# Redis Models
REDISMODEL_HOST = os.environ.get("REDISMODEL_HOST", "")
REDISMODEL_PORT = os.environ.get("REDISMODEL_PORT", "")
REDISMODEL_TOKEN = os.environ.get("REDISMODEL_TOKEN", "")

if not REDISMODEL_HOST:
    # REDIS ADDRESS
    try:
        import boto3

        client = boto3.client("ssm", region_name=DEFAULT_REGION)
        response = client.get_parameter(Name="Redists1ModelDNS")
        REDISMODEL_HOST = response["Parameter"]["Value"]
    except Exception:
        pass

if not REDISMODEL_PORT:
    REDISMODEL_PORT = DEFAULT_REDISMODEL_PORT

# Loglevel
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
# custom loglevel even below debug
LOGLEVEL_VERBOSE = 5
# logging format: json, none
LOGFORMAT = os.environ.get("LOGFORMAT", "json")

# Docker test
DOCKER_MODE = os.environ.get("DOCKER_MODE", False)  # switch PROD/TEST off=False

# for development: APIv2 feature switch (only used when run with python, productive services use gunicorn)
USE_FASTAPI = os.getenv("USE_FASTAPI", "true").lower() == "true"

# App Version
VERSION = os.environ.get("IDENTIFIER", "")

# Syslog Target
SYSLOG_HOST = os.environ.get("SYSLOG_HOST", "")
SYSLOG_PORT = os.environ.get("SYSLOG_PORT", "")

# NF API key
NF_API_KEY = os.environ.get("NF_API_KEY")

# Pinecone API KEY
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT: str = "eu-west1-gcp"
if not PINECONE_API_KEY:
    logging.warning(
        "Could not load Pinecone API key, expected as environment variable PINECONE_API_KEY! "
        "Resuming without the possibility to use pinecone. I'm just a warning, not an error."
    )


# Xayn API KEY
XAYN_FRONT_OFFICE_URL: str = os.environ.get("XAYN_FRONT_OFFICE_URL")
XAYN_FRONT_OFFICE_KEY: str = os.environ.get("XAYN_FRONT_OFFICE_KEY")


# useful FSDB columns (after matching)
FSDB_MATCHES_COLUMNS = (
    "externalid",
    "clip_archnr",
    "clip_deskripts",
    "clip_kats",
    "clip_ut_content",
    "clip_visual_content",
    "prodNr_y",
    "title_y",
)
