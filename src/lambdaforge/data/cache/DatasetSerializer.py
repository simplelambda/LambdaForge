"""Abstract serialization contract for cached dataset samples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DatasetSerializer(ABC):
    """Convert arbitrary dataset samples to and from immutable bytes."""

    @property
    def format_fingerprint(self) -> str:
        """Identify the concrete serialization format for cache-key isolation."""
        return f"{type(self).__module__}.{type(self).__qualname__}"

    @abstractmethod
    def dumps(self, value: Any) -> bytes:
        """Serialize one cache key or dataset sample."""
        raise NotImplementedError

    @abstractmethod
    def loads(self, payload: Any) -> Any:
        """Deserialize one bytes-like cache payload."""
        raise NotImplementedError
