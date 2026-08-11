"""Typed scientific configuration identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ScientificIdentity:
    """Hash only model/data/code choices that can alter scientific results."""

    digest: str
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ScientificIdentity:
        """Create a deterministic identity from normalized values."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(f"sha256:{hashlib.sha256(encoded).hexdigest()}", dict(payload))
