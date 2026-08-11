"""Reusable placement and resource preset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.execution.ResourceRequest import ResourceRequest


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Name one cluster plus its default portable resource request."""

    name: str
    cluster: str
    resources: ResourceRequest

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> ExecutionProfile:
        """Build a validated profile from catalogue YAML."""
        cluster = value.get("cluster")
        if not isinstance(cluster, str) or not cluster.strip():
            raise ValueError(f"Execution profile {name!r} requires cluster.")
        resources = value.get("resources", {})
        if not isinstance(resources, Mapping):
            raise TypeError(f"Execution profile {name!r} resources must be a mapping.")
        return cls(name, cluster, ResourceRequest.from_mapping(resources))

    def to_dict(self) -> dict[str, object]:
        """Return a portable profile descriptor."""
        return {"name": self.name, "cluster": self.cluster, "resources": self.resources.to_dict()}
