"""Abstract byte-storage contract for dataset cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lambdaforge.data.cache.CacheRecord import CacheRecord
from lambdaforge.data.cache.CacheUsage import CacheUsage


class CacheBackend(ABC):
    """Persist serialized dataset records behind opaque SHA-256 keys."""

    @abstractmethod
    def read(self, key: str) -> CacheRecord | None:
        """Return a closable record, or ``None`` when the key is absent."""
        raise NotImplementedError

    @abstractmethod
    def write(self, key: str, payload: bytes) -> bool:
        """Atomically store a payload and report whether it fitted the quota."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, key: str) -> None:
        """Remove one record if present."""
        raise NotImplementedError

    def remove_if_unchanged(self, key: str, token: str | None) -> bool:
        """Remove conditionally when a backend can identify record generations."""
        self.remove(key)
        return True

    @abstractmethod
    def clear(self) -> None:
        """Remove every record owned by this backend namespace."""
        raise NotImplementedError

    @property
    @abstractmethod
    def current_bytes(self) -> int:
        """Return bytes currently occupied by cache record files."""
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_count(self) -> int:
        """Return the current number of cache records."""
        raise NotImplementedError

    def usage(self) -> CacheUsage:
        """Return a usage snapshot; custom backends may override atomically."""
        return CacheUsage(entries=self.entry_count, bytes=self.current_bytes)
