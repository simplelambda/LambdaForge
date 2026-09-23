"""Versioned deterministic project seed streams."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.ProjectContext import ProjectContext


@dataclass(frozen=True, slots=True)
class SeedIdentity:
    """One concrete seed and the stream coordinate that produced it."""

    value: int
    ordinal: int
    role: str
    stream_version: str
    namespace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ordinal": self.ordinal,
            "role": self.role,
            "stream_version": self.stream_version,
            "namespace": self.namespace,
        }


@dataclass(frozen=True, slots=True)
class ProjectSeedStream:
    """A practically extensible reproducible seed stream scoped to a project and role.

    SHA-256 derives a keyed affine permutation over 31-bit ordinals.  The high bit separates
    replicate and confirmation domains.  Consequently the first 2**31 values in either stream
    are collision-free, stable across platforms/interpreters and valid for NumPy/PyTorch's common
    32-bit seed APIs.  This is an algorithmic stream, not a finite hard-coded seed table.
    """

    namespace: str
    role: str = "replicate"
    version: str = "project-sha256-v1"

    def __post_init__(self) -> None:
        if self.role not in {"replicate", "confirmation"}:
            raise ValueError("Seed stream role must be replicate or confirmation.")
        if not self.namespace:
            raise ValueError("Seed stream namespace cannot be empty.")

    @classmethod
    def for_project(
        cls, project: ProjectContext, *, role: str = "replicate"
    ) -> ProjectSeedStream:
        return cls(project.project_id, role)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProjectSeedStream:
        version = str(value.get("stream_version", value.get("version", "project-sha256-v1")))
        if version != "project-sha256-v1":
            raise ValueError(f"Unsupported seed stream version: {version!r}.")
        return cls(str(value["namespace"]), str(value.get("role", "replicate")), version)

    def at(self, ordinal: int) -> SeedIdentity:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("Seed ordinal must be a non-negative integer.")
        modulus = 1 << 31
        if ordinal >= modulus:
            raise ValueError("Seed ordinal exceeds the collision-free 31-bit stream domain.")
        material = (
            f"lambdaforge\0seed-stream\0{self.version}\0{self.namespace}\0{self.role}"
        ).encode()
        digest = hashlib.sha256(material).digest()
        multiplier = int.from_bytes(digest[:4], "big") | 1
        offset = int.from_bytes(digest[4:8], "big")
        low_bits = (multiplier * ordinal + offset) % modulus
        role_bit = 0 if self.role == "replicate" else modulus
        value = role_bit | low_bits
        return SeedIdentity(value, ordinal, self.role, self.version, self.namespace)

    def take(self, count: int, *, start: int = 0) -> tuple[SeedIdentity, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Seed count must be a non-negative integer.")
        return tuple(self.at(ordinal) for ordinal in range(start, start + count))

    def values(self, count: int, *, start: int = 0) -> tuple[int, ...]:
        return tuple(identity.value for identity in self.take(count, start=start))

    def iter_from(self, ordinal: int = 0) -> Iterator[SeedIdentity]:
        while True:
            yield self.at(ordinal)
            ordinal += 1

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": "project-stream",
            "namespace": self.namespace,
            "role": self.role,
            "stream_version": self.version,
        }


class SeedProvider:
    """Resolve domain-separated project streams and explicit authored seed identities."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    def stream(self, role: str = "replicate") -> ProjectSeedStream:
        return ProjectSeedStream.for_project(self.project, role=role)

    def explicit(self, values: tuple[int, ...]) -> tuple[SeedIdentity, ...]:
        return tuple(
            SeedIdentity(value, ordinal, "explicit", "authored-v1", self.project.project_id)
            for ordinal, value in enumerate(values)
        )


__all__ = ["ProjectSeedStream", "SeedIdentity", "SeedProvider"]
