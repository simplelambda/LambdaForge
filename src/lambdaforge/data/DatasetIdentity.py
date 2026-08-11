"""Logical dataset identity independent from storage location."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Describe which immutable dataset content a task consumes."""

    provider: str
    value: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.value.strip():
            raise ValueError("Dataset identity provider and value cannot be empty.")

    @property
    def digest(self) -> str:
        """Return a uniform SHA-256 key for caches and scientific identities."""
        payload = json.dumps(
            {"provider": self.provider, "value": self.value},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """Return the stable persisted identity descriptor."""
        return {"provider": self.provider, "value": self.value, "digest": self.digest}
