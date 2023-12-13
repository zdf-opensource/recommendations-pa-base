# Copyright (c) 2023, ZDF.
"""
Util functions for recommendation models.
"""
import logging
import typing as t

from pa_base.zdf.configuration.dynamic_configs import (
    DEFAULT_MODEL_TARGET,
    TARGET_TO_APPID_PREFIXES_MAPPING,
)

try:
    from device_detector import DeviceDetector
except ImportError as exc:
    logging.warning("Could not import device-detector", exc_info=exc)


# UA and model target handling
SPECIAL_UA: t.Dict[str, t.Dict[str, str]] = {
    "python-requests/2.22.0": {"debug": "true"},
    "smarttvbackend-MIT/v3.1": {"debug": "false", "device_type": "tv"},
    "GuzzleHttp/6.3.3 curl/7.58.0 PHP/7.2.17-0ubuntu0.18.04.1": {
        "debug": "false",
        "device_type": "smartphone",
    },
}


def get_model_target(appid: str, user_agent: str) -> str:
    """get the model frontend target for given appid and user agent"""
    if not appid and not user_agent:
        return DEFAULT_MODEL_TARGET
    appid = str(appid).lower()
    user_agent = str(user_agent)

    # evaluate appid
    if appid:
        if appid.startswith(TARGET_TO_APPID_PREFIXES_MAPPING["tv"]):
            # MUST be before "mobile" case, because android(tv) prefixes match partially
            return "tv"  # app_class = "native_tv" / "hbbtv"
        elif appid.startswith(TARGET_TO_APPID_PREFIXES_MAPPING["pctablet"]):
            return "pctablet"  # app_class = "rws" / "pwa"
        elif appid.startswith(TARGET_TO_APPID_PREFIXES_MAPPING["mobile"]):
            return "mobile"  # app_class = "native"
        else:
            logging.info(f"Unknown appId '{appid}'")

    if user_agent:
        # evaluate user_agent if app_id is ambiguous or unavailable
        # check if its a known SPECIAL_UA
        if user_agent in SPECIAL_UA:
            device_type = SPECIAL_UA[user_agent].get("device_type", "")
        else:
            try:
                device_type = DeviceDetector(user_agent).parse().device_type()
            except Exception:  # noqa
                device_type = ""
    else:
        device_type = ""

    # we have three model targets: tv, pc (including tablet) and mobile
    if device_type == "tv":  # or app_class in {"native_tv", "hbbtv"}:
        return "tv"
    elif device_type == "smartphone":  # or app_class == "native":
        return "mobile"
    elif device_type in {"tablet", "desktop"}:  # or app_class in {"rws", "pwa"}:
        return "pctablet"
    return DEFAULT_MODEL_TARGET
