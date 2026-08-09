"""Immutable registry filter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RegistryQuery:
    """Filter catalog records by canonical fields, tags and metadata."""

    statuses: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None

    @classmethod
    def create(
        cls,
        *,
        status: str | Sequence[str] | None = None,
        name: str | Sequence[str] | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        fingerprint: str | None = None,
    ) -> RegistryQuery:
        """Normalize ergonomic scalar-or-sequence filters."""

        def values(value: str | Sequence[str] | None) -> tuple[str, ...]:
            if value is None:
                return ()
            return (value,) if isinstance(value, str) else tuple(value)

        return cls(values(status), values(name), tuple(tags), dict(metadata or {}), fingerprint)
