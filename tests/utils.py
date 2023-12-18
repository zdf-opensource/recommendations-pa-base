"""Utilities used for testing."""
import redis


def redis_delete_all_keys(r: redis.Redis, suffix: str):
    for key in r.scan_iter(f"*{suffix}*"):
        r.delete(key)
