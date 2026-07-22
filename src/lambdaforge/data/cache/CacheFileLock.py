"""Backward-compatible cache-specific adapter for the common file lock."""

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class CacheFileLock(CrossProcessFileLock):
    """Retain the cache lock import path while sharing the runtime implementation."""
