# Copyright (c) 2023, ZDF.
"""
Dynamic configurations loaded on service startup.
"""

import logging

logging.warning(
    "pa_base.configuration.dynamic_configs is deprecated, use pa_base.zdf.configuration.dynamic_configs for ZDF-specific configuration instead. If you're using this in tenant code, please inform ZDF."
)

from pa_base.zdf.configuration.dynamic_configs import *  # noqa: E402, F401, F403


class Singleton(type):
    """
    Singleton metaclass.

    Based on a highly explanatory StackOverlow answer: https://stackoverflow.com/a/6798042/3410474

    Usage:
    >>> class MyClass(metaclass=Singleton):
    ...     pass
    """

    # store all previous instances
    _instances = {}
    # store all classes that have been called --> prevent ending in a loop instantiating a heavy Singleton again and again
    _called_instances = set()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._called_instances:
            cls._called_instances |= {cls}
            cls._instances[cls] = super().__call__(*args, **kwargs)
        elif cls not in cls._instances:
            raise KeyError(f"Instance of {cls.__name__} not found, failed __init__() previously?")
        return cls._instances[cls]


class ZdfDynmicConfiguration(metaclass=Singleton):
    def __init__(self):
        print("init")
