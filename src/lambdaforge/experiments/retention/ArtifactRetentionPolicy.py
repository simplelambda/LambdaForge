"""Typed top-level artifact-retention policy."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.ArtifactRetentionMode import ArtifactRetentionMode
from lambdaforge.experiments.retention.ArtifactRetentionRule import ArtifactRetentionRule
from lambdaforge.experiments.retention.CheckpointRetention import CheckpointRetention


@dataclass(frozen=True)
class ArtifactRetentionPolicy:
    """Own validated defaults for checkpoint and generic artifact retention."""

    mode: ArtifactRetentionMode = ArtifactRetentionMode.DISABLED
    checkpoint_keep: CheckpointRetention = CheckpointRetention.ALL
    prune_unselected_checkpoints: bool = False
    protect: tuple[str, ...] = ()
    rules: tuple[ArtifactRetentionRule, ...] = ()
    archive_name: str = "artifacts.zip"
    archive_compression_level: int = 6
    lock_timeout_seconds: float = 60.0

    _ARCHIVE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*[.]zip$")

    def __post_init__(self) -> None:
        """Enforce every policy invariant for direct object construction."""
        if not isinstance(self.mode, ArtifactRetentionMode):
            raise TypeError("retention mode must be an ArtifactRetentionMode.")
        if not isinstance(self.checkpoint_keep, CheckpointRetention):
            raise TypeError("retention checkpoint_keep must be a CheckpointRetention.")
        if not isinstance(self.prune_unselected_checkpoints, bool):
            raise TypeError("retention prune_unselected_checkpoints must be a bool.")
        self._direct_patterns(self.protect)
        if not isinstance(self.rules, tuple):
            raise TypeError("retention rules must be a tuple of ArtifactRetentionRule objects.")
        if any(not isinstance(rule, ArtifactRetentionRule) for rule in self.rules):
            raise TypeError("retention rules must contain ArtifactRetentionRule objects.")
        if not isinstance(self.archive_name, str):
            raise TypeError("retention archive_name must be a string.")
        if not self._ARCHIVE_PATTERN.fullmatch(self.archive_name):
            raise ValueError(
                "retention.archive.name must be a simple portable filename ending in '.zip'."
            )
        if isinstance(self.archive_compression_level, bool) or not isinstance(
            self.archive_compression_level,
            int,
        ):
            raise TypeError("retention.archive.compression_level must be an integer.")
        if not 0 <= self.archive_compression_level <= 9:
            raise ValueError("retention.archive.compression_level must be between 0 and 9.")
        if isinstance(self.lock_timeout_seconds, bool) or not isinstance(
            self.lock_timeout_seconds,
            (int, float),
        ):
            raise TypeError("retention.lock_timeout_seconds must be a number.")
        timeout = float(self.lock_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("retention.lock_timeout_seconds must be a finite positive number.")
        object.__setattr__(self, "lock_timeout_seconds", timeout)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ArtifactRetentionPolicy:
        """Read the optional top-level retention block from an experiment."""
        value = config.get("retention", {})
        if not isinstance(value, Mapping):
            raise TypeError("retention must be a mapping.")
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionPolicy:
        """Parse a retention block and reject ambiguous programmatic values."""
        if not isinstance(value, Mapping):
            raise TypeError("retention must be a mapping.")
        allowed = {
            "mode",
            "checkpoints",
            "protect",
            "rules",
            "archive",
            "lock_timeout_seconds",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown retention options: {unknown}.")

        raw_mode = value.get("mode", "disabled")
        if not isinstance(raw_mode, str):
            raise TypeError("retention.mode must be a string.")
        mode = ArtifactRetentionMode(raw_mode)
        checkpoints = value.get("checkpoints", {})
        if not isinstance(checkpoints, Mapping):
            raise TypeError("retention.checkpoints must be a mapping.")
        checkpoint_unknown = sorted(set(checkpoints) - {"keep", "prune_unselected"})
        if checkpoint_unknown:
            raise ValueError(f"Unknown retention.checkpoints options: {checkpoint_unknown}.")
        raw_keep = checkpoints.get("keep", "all")
        if not isinstance(raw_keep, str):
            raise TypeError("retention.checkpoints.keep must be a string.")
        keep = CheckpointRetention(raw_keep)
        prune_unselected = checkpoints.get("prune_unselected", False)
        if not isinstance(prune_unselected, bool):
            raise TypeError("retention.checkpoints.prune_unselected must be a bool.")

        protect = cls._patterns(value.get("protect", []))
        raw_rules = value.get("rules", [])
        if not isinstance(raw_rules, list):
            raise TypeError("retention.rules must be a list.")
        rules = tuple(ArtifactRetentionRule.from_mapping(rule) for rule in raw_rules)

        archive = value.get("archive", {})
        if not isinstance(archive, Mapping):
            raise TypeError("retention.archive must be a mapping.")
        archive_unknown = sorted(set(archive) - {"name", "compression_level"})
        if archive_unknown:
            raise ValueError(f"Unknown retention.archive options: {archive_unknown}.")
        archive_name = archive.get("name", "artifacts.zip")
        if not isinstance(archive_name, str):
            raise TypeError("retention.archive.name must be a string.")
        compression_level = archive.get("compression_level", 6)

        timeout = value.get("lock_timeout_seconds", 60.0)
        return cls(
            mode=mode,
            checkpoint_keep=keep,
            prune_unselected_checkpoints=prune_unselected,
            protect=protect,
            rules=rules,
            archive_name=archive_name,
            archive_compression_level=compression_level,
            lock_timeout_seconds=float(timeout),
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 of the effective policy."""
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compression_level_for(self, rule: ArtifactRetentionRule) -> int:
        """Resolve a rule override against the archive-level default."""
        if rule.compression is not None and rule.compression.level is not None:
            return rule.compression.level
        return self.archive_compression_level

    def to_dict(self) -> dict[str, Any]:
        """Return the complete effective policy for audit manifests."""
        return {
            "mode": self.mode.value,
            "checkpoints": {
                "keep": self.checkpoint_keep.value,
                "prune_unselected": self.prune_unselected_checkpoints,
            },
            "protect": list(self.protect),
            "rules": [rule.to_dict() for rule in self.rules],
            "archive": {
                "name": self.archive_name,
                "compression_level": self.archive_compression_level,
            },
            "lock_timeout_seconds": self.lock_timeout_seconds,
        }

    @staticmethod
    def _patterns(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise TypeError("retention.protect must be a list of patterns.")
        if any(not isinstance(pattern, str) for pattern in value):
            raise TypeError("retention.protect patterns must be strings.")
        return tuple(
            ArtifactPathGuard.validate_pattern(pattern, rule_pattern=False) for pattern in value
        )

    @staticmethod
    def _direct_patterns(value: tuple[str, ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError("retention protect must be a tuple of patterns.")
        if any(not isinstance(pattern, str) for pattern in value):
            raise TypeError("retention protect patterns must be strings.")
        for pattern in value:
            ArtifactPathGuard.validate_pattern(pattern, rule_pattern=False)
