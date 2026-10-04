"""Operational execution fleets, deliberately outside scientific Work configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ClusterHealth(str, Enum):
    """Separate operator eligibility from observational connectivity/readiness."""

    ONLINE = "online"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    DRAINING = "draining"
    DISABLED = "disabled"


def positive_limit(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer or null.")
    return value


@dataclass(frozen=True, slots=True)
class FleetMember:
    """Operator caps; none of these settings author candidates, seeds or objectives."""

    cluster: str
    max_gpus: int | None = None
    max_runs: int | None = None
    max_inflight_jobs: int | None = None
    required: bool = False
    state: ClusterHealth = ClusterHealth.ONLINE

    def __post_init__(self) -> None:
        if not isinstance(self.state, ClusterHealth):
            raise TypeError("Fleet member state must be a ClusterHealth value.")
        if not isinstance(self.cluster, str) or not self.cluster.strip():
            raise ValueError("Fleet member requires a cluster name.")
        for name in ("max_gpus", "max_runs", "max_inflight_jobs"):
            positive_limit(getattr(self, name), f"Fleet member {name}")
        if not isinstance(self.required, bool):
            raise TypeError("Fleet member required must be a boolean.")
        if self.state not in {
            ClusterHealth.ONLINE,
            ClusterHealth.DRAINING,
            ClusterHealth.DISABLED,
        }:
            raise ValueError("Authored fleet state must be online, draining or disabled.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FleetMember:
        unknown = set(value) - {
            "cluster",
            "max_gpus",
            "max_runs",
            "max_inflight_jobs",
            "required",
            "state",
        }
        if unknown:
            raise ValueError(f"Unknown fleet member fields: {sorted(unknown)}.")
        return cls(
            cluster=value.get("cluster", ""),
            max_gpus=positive_limit(value.get("max_gpus"), "max_gpus"),
            max_runs=positive_limit(value.get("max_runs"), "max_runs"),
            max_inflight_jobs=positive_limit(value.get("max_inflight_jobs"), "max_inflight_jobs"),
            required=value.get("required", False),
            state=ClusterHealth(value.get("state", "online")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "max_gpus": self.max_gpus,
            "max_runs": self.max_runs,
            "max_inflight_jobs": self.max_inflight_jobs,
            "required": self.required,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class Fleet:
    """A named coordinator role and execution targets; not an independent JobGroup."""

    name: str
    members: tuple[FleetMember, ...]
    coordinator: str = "local"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.name):
            raise ValueError(
                "Fleet name must be 1–80 letters, digits, dots, underscores or dashes."
            )
        if not self.coordinator or not self.members:
            raise ValueError("Fleet requires a coordinator and at least one member.")
        if len({member.cluster for member in self.members}) != len(self.members):
            raise ValueError("Fleet members must be unique.")

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> Fleet:
        unknown = set(value) - {"coordinator", "members"}
        if unknown:
            raise ValueError(f"Unknown fleet fields: {sorted(unknown)}.")
        members = value.get("members")
        if not isinstance(members, list) or not all(isinstance(item, Mapping) for item in members):
            raise TypeError("Fleet members must be a list of mappings.")
        coordinator = value.get("coordinator", "local")
        if not isinstance(coordinator, str):
            raise TypeError("Fleet coordinator must be a named cluster or local.")
        return cls(name, tuple(FleetMember.from_mapping(item) for item in members), coordinator)

    def member(self, cluster: str) -> FleetMember:
        try:
            return next(item for item in self.members if item.cluster == cluster)
        except StopIteration as error:
            raise KeyError(f"Cluster {cluster!r} is not in fleet {self.name!r}.") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_version": 1,
            "name": self.name,
            "coordinator": self.coordinator,
            "members": [member.to_dict() for member in self.members],
        }
