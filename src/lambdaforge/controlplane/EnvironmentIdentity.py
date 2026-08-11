"""Content identity for a remotely managed Python environment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    """Identify exact framework/project wheels and dependency policy."""

    environment_id: str
    wheels: tuple[Mapping[str, Any], ...]
    python_requirement: str
    offline: bool

    @classmethod
    def create(
        cls,
        wheels: Sequence[Mapping[str, Any]],
        *,
        python_requirement: str,
        offline: bool,
    ) -> EnvironmentIdentity:
        """Derive a deterministic identity from installable bytes and policy."""
        normalized = tuple(
            sorted(
                (
                    {
                        "name": str(value["name"]),
                        "sha256": str(value["sha256"]),
                        "size_bytes": int(value["size_bytes"]),
                    }
                    for value in wheels
                ),
                key=lambda item: str(item["name"]),
            )
        )
        payload = {
            "identity_version": 1,
            "wheels": normalized,
            "python_requirement": python_requirement,
            "offline": bool(offline),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(f"env-{digest[:24]}", normalized, python_requirement, bool(offline))

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable environment manifest."""
        return {
            "environment_identity_version": 1,
            "environment_id": self.environment_id,
            "wheels": [dict(value) for value in self.wheels],
            "python_requirement": self.python_requirement,
            "offline": self.offline,
        }
