"""Content identity for a remotely managed Python environment."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    """Identify exact framework/project wheels and dependency policy."""

    environment_id: str
    wheels: tuple[Mapping[str, Any], ...]
    python_requirement: str
    offline: bool
    dependency_policy: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        wheels: Sequence[Mapping[str, Any]],
        *,
        python_requirement: str,
        offline: bool,
        dependency_policy: Mapping[str, Any] | None = None,
    ) -> EnvironmentIdentity:
        """Derive a deterministic identity from installable bytes and policy."""
        normalized = tuple(
            sorted(
                (
                    FrozenJsonMapping(
                        {
                            "name": str(value["name"]),
                            "sha256": str(value["sha256"]),
                            "size_bytes": int(value["size_bytes"]),
                        }
                    )
                    for value in wheels
                ),
                key=lambda item: str(item["name"]),
            )
        )
        policy = FrozenJsonMapping(dependency_policy)
        payload = {
            "identity_version": 2,
            "wheels": normalized,
            "python_requirement": python_requirement,
            "offline": bool(offline),
            "dependency_policy": policy,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(f"env-{digest[:24]}", normalized, python_requirement, bool(offline), policy)

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable environment manifest."""
        return {
            "environment_identity_version": 2,
            "environment_id": self.environment_id,
            "wheels": [copy.deepcopy(value) for value in self.wheels],
            "python_requirement": self.python_requirement,
            "offline": self.offline,
            "dependency_policy": copy.deepcopy(self.dependency_policy),
        }
