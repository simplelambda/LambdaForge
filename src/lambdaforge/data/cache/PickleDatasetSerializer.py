"""Default pickle serializer for trusted local dataset caches."""

from __future__ import annotations

import pickle
from typing import Any

from lambdaforge.data.cache.DatasetSerializer import DatasetSerializer


class PickleDatasetSerializer(DatasetSerializer):
    """Serialize samples with a configurable pickle protocol.

    Pickle cache files are code-execution capable and must only be read from a
    directory controlled by the same trusted user and project.
    """

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
        if protocol < 0 or protocol > pickle.HIGHEST_PROTOCOL:
            raise ValueError(f"protocol must be between 0 and {pickle.HIGHEST_PROTOCOL}.")
        self.protocol = protocol

    @property
    def format_fingerprint(self) -> str:
        """Identify the explicit pickle protocol used for persistent bytes."""
        return f"python-pickle:{self.protocol}"

    def dumps(self, value: Any) -> bytes:
        """Serialize one value to immutable bytes."""
        return pickle.dumps(value, protocol=self.protocol)

    def loads(self, payload: Any) -> Any:
        """Deserialize one trusted bytes-like payload."""
        return pickle.loads(payload)
