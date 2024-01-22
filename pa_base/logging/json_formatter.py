# Copyright (c) 2024, ZDF.
"""
A custom json formatter for structured logging.
"""
from contextlib import suppress

from pa_base.configuration.config import VERSION

try:
    # optional third-party may not be available
    from flask import g, request
    from pythonjsonlogger import jsonlogger

    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            with suppress(Exception):
                reco_id = str(g.get("recoid").get("cache_key", ""))
                log_record["recoId"] = reco_id
            with suppress(Exception):
                app_id = (
                    request.args.get("appId")
                    or request.form.get("appId")
                    or (request.get_json(silent=True, cache=True) or {}).get("appId")
                )
                log_record["appId"] = app_id
            with suppress(Exception):
                path = request.path
                log_record["path"] = path
            with suppress(Exception):
                log_record["pageId"] = request.args.get("pageId")
            with suppress(Exception):
                log_record["version"] = VERSION
            with suppress(Exception):
                zdf_x_request_id = request.headers.get("X-Request-Id")
                log_record["X-Request-Id"] = zdf_x_request_id

except ImportError:
    import logging

    CustomJsonFormatter = logging.Formatter
