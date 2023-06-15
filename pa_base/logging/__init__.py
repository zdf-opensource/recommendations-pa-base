# Copyright (c) 2022, ZDF.
import enum
import logging
from logging.config import dictConfig
from typing import Union

from pa_base.configuration.config import LOGLEVEL_VERBOSE
from pa_base.logging.json_formatter import CustomJsonFormatter


@enum.unique
class LogFormat(str, enum.Enum):
    NONE = "none"
    JSON = "json"


# custom loglevel below debug
logging.addLevelName(LOGLEVEL_VERBOSE, "VERBOSE")


def reconfigure_logging(
    file: str,
    loglevel: Union[int, str],
    logformat: Union[str, LogFormat] = LogFormat.NONE,
    filemode: str = "w",
) -> None:
    """
    reconfigure logging with file and stream handlers
    :param file: complete path to file, e.g., f"{__file__}.log"
    :param loglevel: log level or level name, e.g., logging.DEBUG or "DEBUG"
    :param filemode: how to open the file, either "w" to overwrite or "a" to append
    """
    # collect errors and log them at the end after logging is configured
    errors = []
    if isinstance(loglevel, str):
        levels = {
            "VERBOSE": LOGLEVEL_VERBOSE,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        try:
            loglevel: int = levels[loglevel.upper()]
        except KeyError:
            errors.append(
                f"Unknown loglevel {loglevel}, falling back to 'logging.INFO'."
            )
            loglevel: int = logging.INFO
    if not isinstance(logformat, LogFormat):
        try:
            logformat: LogFormat = LogFormat(logformat.lower())
        except ValueError:
            errors.append(
                f"Unknown logformat {logformat}, falling back to 'LogFormat.NONE'."
            )
            logformat: LogFormat = LogFormat.NONE
    dictConfig(
        {
            "version": 1,
            "formatters": {
                "default": {
                    # use JsonFormatter on int and prod for structured logging into CloudWatch
                    "()": (
                        # disable JSON-structured logging in local dev environment by setting LOGFORMAT=none:
                        CustomJsonFormatter
                        if logformat == LogFormat.JSON
                        else "logging.Formatter"
                    ),
                    "format": (
                        "[%(asctime)s] %(levelname)s PID %(process)d"
                        " in %(module)s: %(message)s"
                    ),
                },
            },
            "handlers": {
                "stream": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",  # default is stderr
                    "formatter": "default",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": file,
                    "mode": filemode,
                    "formatter": "default",
                    "encoding": "utf-8",
                },
            },
            "root": {"level": loglevel, "handlers": ["stream", "file"]},
            "loggers": {
                "gunicorn": {"propagate": True},
                "uvicorn": {"propagate": True},
                "uvicorn.access": {"propagate": True},
            },
        }
    )
    if errors:
        logging.info(
            f"Some error{'s' if len(errors) > 1 else ''} occurred during logging configuration: {' '.join(errors)}"
        )
