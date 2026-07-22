"""Immutable generic artifact-retention rule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.retention.ArtifactCompressionOptions import (
    ArtifactCompressionOptions,
)
from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.ArtifactRetentionAction import ArtifactRetentionAction


@dataclass(frozen=True)
class ArtifactRetentionRule:
    """Match regular run files for compression or pruning."""

    action: ArtifactRetentionAction
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    min_size_bytes: int = 0
    compression: ArtifactCompressionOptions | None = None

    def __post_init__(self) -> None:
        """Validate direct construction without weakening the YAML contract."""
        if not isinstance(self.action, ArtifactRetentionAction):
            raise TypeError("retention rule action must be an ArtifactRetentionAction.")
        if self.action is ArtifactRetentionAction.PRUNE_CHECKPOINT:
            raise ValueError("prune_checkpoint is internal and cannot appear in YAML rules.")
        self._direct_patterns(self.include, "include", allow_empty=False)
        self._direct_patterns(self.exclude, "exclude", allow_empty=True)
        if isinstance(self.min_size_bytes, bool) or not isinstance(self.min_size_bytes, int):
            raise TypeError("retention rule min_size_bytes must be an integer.")
        if self.min_size_bytes < 0:
            raise ValueError("retention rule min_size_bytes cannot be negative.")
        if self.action is ArtifactRetentionAction.PRUNE:
            if self.compression is not None:
                raise ValueError("Prune rules cannot define compression options.")
            return
        if self.compression is None:
            object.__setattr__(self, "compression", ArtifactCompressionOptions())
        elif not isinstance(self.compression, ArtifactCompressionOptions):
            raise TypeError(
                "Compress rule compression must be an ArtifactCompressionOptions object."
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionRule:
        """Parse and semantically validate one YAML rule."""
        if not isinstance(value, Mapping):
            raise TypeError("Each retention rule must be a mapping.")
        allowed = {"action", "include", "exclude", "min_size_bytes", "compression"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown retention rule options: {unknown}.")
        raw_action = value.get("action")
        if not isinstance(raw_action, str):
            raise TypeError("retention rule action must be a string.")
        action = ArtifactRetentionAction(raw_action)
        if action is ArtifactRetentionAction.PRUNE_CHECKPOINT:
            raise ValueError("prune_checkpoint is internal and cannot appear in YAML rules.")
        include = cls._patterns(value.get("include"), "include", allow_empty=False)
        exclude = cls._patterns(value.get("exclude", []), "exclude", allow_empty=True)
        min_size = value.get("min_size_bytes", 0)
        raw_compression = value.get("compression")
        if action is ArtifactRetentionAction.PRUNE and "compression" in value:
            raise ValueError("Prune rules cannot define compression options.")
        compression = (
            ArtifactCompressionOptions.from_mapping(raw_compression)
            if action is ArtifactRetentionAction.COMPRESS
            else None
        )
        return cls(
            action=action,
            include=include,
            exclude=exclude,
            min_size_bytes=min_size,
            compression=compression,
        )

    def matches(self, relative_path: str, size: int) -> bool:
        """Return whether this rule selects a regular unprotected file."""
        return (
            size >= self.min_size_bytes
            and ArtifactPathGuard.matches(relative_path, self.include)
            and not ArtifactPathGuard.matches(relative_path, self.exclude)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible rule."""
        payload: dict[str, Any] = {
            "action": self.action.value,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "min_size_bytes": self.min_size_bytes,
        }
        if self.compression is not None:
            payload["compression"] = self.compression.to_dict()
        return payload

    @staticmethod
    def _patterns(value: Any, label: str, *, allow_empty: bool) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise TypeError(f"retention rule {label} must be a list of patterns.")
        if any(not isinstance(pattern, str) for pattern in value):
            raise TypeError(f"retention rule {label} patterns must be strings.")
        patterns = tuple(
            ArtifactPathGuard.validate_pattern(pattern, rule_pattern=True) for pattern in value
        )
        if not allow_empty and not patterns:
            raise ValueError(f"retention rule {label} cannot be empty.")
        return patterns

    @staticmethod
    def _direct_patterns(
        value: tuple[str, ...],
        label: str,
        *,
        allow_empty: bool,
    ) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"retention rule {label} must be a tuple of patterns.")
        if any(not isinstance(pattern, str) for pattern in value):
            raise TypeError(f"retention rule {label} patterns must be strings.")
        if not allow_empty and not value:
            raise ValueError(f"retention rule {label} cannot be empty.")
        for pattern in value:
            ArtifactPathGuard.validate_pattern(pattern, rule_pattern=True)
