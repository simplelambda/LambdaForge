"""One fingerprinted filesystem operation in a retention plan."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from lambdaforge.experiments.retention.ArtifactRetentionAction import ArtifactRetentionAction


class ArtifactRetentionOperation:
    """Describe an immutable candidate and the state needed for race detection."""

    FIELD_NAMES = frozenset(
        {
            "run_relative",
            "relative_path",
            "action",
            "size_bytes",
            "sha256",
            "mtime_ns",
            "compression_level",
            "only_if_smaller",
        }
    )
    _SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
    _DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
    _CORE_PROTECTED = frozenset(
        {
            "config.yaml",
            "environment.json",
            "hparams.json",
            "train.log",
            "metrics.csv",
            "result.json",
        }
    )
    _GENERIC_PROTECTED_PREFIXES = frozenset(
        {
            ".lambdaforge",
            "aggregate",
            "checkpoints",
        }
    )

    def __init__(
        self,
        *,
        run_relative: str,
        relative_path: str,
        action: ArtifactRetentionAction | str,
        size_bytes: int,
        sha256: str,
        mtime_ns: int,
        compression_level: int | None = None,
        only_if_smaller: bool = False,
    ) -> None:
        self.run_relative = self._validated_relative_path(
            run_relative,
            label="run_relative",
        )
        self.relative_path = self._validated_relative_path(
            relative_path,
            label="relative_path",
        )
        if not isinstance(action, (ArtifactRetentionAction, str)):
            raise TypeError("Retention operation action must be a string or typed action.")
        self.action = (
            action
            if isinstance(action, ArtifactRetentionAction)
            else ArtifactRetentionAction(action)
        )
        self.size_bytes = self._validated_integer(size_bytes, label="size_bytes")
        if not isinstance(sha256, str):
            raise TypeError("Retention operation sha256 must be a string.")
        self.sha256 = sha256
        self.mtime_ns = self._validated_integer(mtime_ns, label="mtime_ns")
        if compression_level is not None:
            compression_level = self._validated_integer(
                compression_level,
                label="compression_level",
            )
        if not isinstance(only_if_smaller, bool):
            raise TypeError("Retention operation only_if_smaller must be a bool.")
        self.compression_level = compression_level
        self.only_if_smaller = only_if_smaller
        if self.size_bytes < 0 or self.mtime_ns < 0:
            raise ValueError("Retention operation size and mtime must be non-negative.")
        if self._SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("Retention operation sha256 must be a lowercase SHA-256.")
        self._validate_action_contract()

    @property
    def key(self) -> str:
        """Return the unique suite-relative operation key."""
        return f"{self.run_relative}/{self.relative_path}"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionOperation:
        """Reconstruct an operation from a durable journal."""
        if not isinstance(value, Mapping):
            raise TypeError("Retention operation must be an object.")
        cls._validate_fields(value)
        return cls(
            run_relative=value["run_relative"],
            relative_path=value["relative_path"],
            action=value["action"],
            size_bytes=value["size_bytes"],
            sha256=value["sha256"],
            mtime_ns=value["mtime_ns"],
            compression_level=value["compression_level"],
            only_if_smaller=value["only_if_smaller"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible mapping."""
        return copy.deepcopy(
            {
                "run_relative": self.run_relative,
                "relative_path": self.relative_path,
                "action": self.action.value,
                "size_bytes": self.size_bytes,
                "sha256": self.sha256,
                "mtime_ns": self.mtime_ns,
                "compression_level": self.compression_level,
                "only_if_smaller": self.only_if_smaller,
            }
        )

    @classmethod
    def _validate_fields(cls, value: Mapping[str, Any]) -> None:
        keys = set(value)
        if keys == cls.FIELD_NAMES:
            return
        missing = sorted(cls.FIELD_NAMES - keys)
        unknown = sorted(str(key) for key in keys - cls.FIELD_NAMES)
        raise ValueError(
            f"Retention operation fields are malformed: missing={missing}, unknown={unknown}."
        )

    @classmethod
    def _validated_relative_path(cls, value: object, *, label: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Retention operation {label} must be a string.")
        if not value or "\0" in value or "\\" in value:
            raise ValueError(
                f"Retention operation {label} must be a non-empty relative POSIX path."
            )
        pure = PurePosixPath(value)
        if (
            pure.is_absolute()
            or value.startswith("//")
            or cls._DRIVE_PATTERN.match(value)
            or ".." in pure.parts
            or not pure.parts
            or pure.as_posix() != value
        ):
            raise ValueError(
                f"Retention operation {label} is not a canonical relative POSIX path: {value!r}."
            )
        return value

    @staticmethod
    def _validated_integer(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Retention operation {label} must be an integer.")
        return value

    def _validate_action_contract(self) -> None:
        first = PurePosixPath(self.relative_path).parts[0]
        if self.action is ArtifactRetentionAction.COMPRESS:
            if self.compression_level is None:
                raise ValueError("Compression operations require compression_level.")
            if not 0 <= self.compression_level <= 9:
                raise ValueError("Compression operation level must be between 0 and 9.")
        elif self.compression_level is not None:
            raise ValueError("Only compression operations may define compression_level.")

        if self.action is not ArtifactRetentionAction.COMPRESS and self.only_if_smaller:
            raise ValueError("only_if_smaller is valid only for compression operations.")
        if self.action is ArtifactRetentionAction.PRUNE_CHECKPOINT:
            path = PurePosixPath(self.relative_path)
            if first != "checkpoints" or path.suffix != ".ckpt":
                raise ValueError(
                    "Checkpoint pruning operations must target checkpoints/*.ckpt paths."
                )
            return
        if self.relative_path in self._CORE_PROTECTED or first in self._GENERIC_PROTECTED_PREFIXES:
            raise ValueError(
                f"Generic retention operations cannot target protected path {self.relative_path!r}."
            )
