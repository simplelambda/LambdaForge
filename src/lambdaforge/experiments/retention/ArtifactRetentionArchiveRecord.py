"""Strict durable metadata for one verified retention archive."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.experiments.retention.ArtifactRetentionAction import (
    ArtifactRetentionAction,
)
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)


class ArtifactRetentionArchiveRecord:
    """Validate archive metadata before recovery logic trusts its paths or members."""

    FIELD_NAMES = frozenset(
        {
            "run_relative",
            "path",
            "compression_level",
            "sha256",
            "size_bytes",
            "members",
        }
    )
    _SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
    _DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
    _ARCHIVE_STEM_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

    def __init__(
        self,
        *,
        run_relative: str,
        path: str,
        compression_level: int,
        sha256: str,
        size_bytes: int,
        members: Sequence[str],
    ) -> None:
        self.run_relative = run_relative
        self.path = path
        self.compression_level = compression_level
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.members = tuple(members)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionArchiveRecord:
        """Parse one exact JSON archive record without coercing malformed values."""
        if not isinstance(value, Mapping):
            raise TypeError("Retention archive metadata must be an object.")
        cls._validate_fields(value)
        run_relative = cls._validated_relative_path(
            value["run_relative"],
            label="run_relative",
        )
        path = cls._validated_relative_path(value["path"], label="path")
        compression_level = cls._validated_integer(
            value["compression_level"],
            label="compression_level",
        )
        if not 0 <= compression_level <= 9:
            raise ValueError("Retention archive compression_level must be between 0 and 9.")
        sha256 = value["sha256"]
        if not isinstance(sha256, str):
            raise TypeError("Retention archive sha256 must be a string.")
        if cls._SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError("Retention archive sha256 must be a lowercase SHA-256.")
        size_bytes = cls._validated_integer(value["size_bytes"], label="size_bytes")
        if size_bytes <= 0:
            raise ValueError("Retention archive size_bytes must be positive.")
        raw_members = value["members"]
        if isinstance(raw_members, (str, bytes, bytearray)) or not isinstance(
            raw_members,
            Sequence,
        ):
            raise TypeError("Retention archive members must be a sequence.")
        members = tuple(
            cls._validated_relative_path(member, label="member") for member in raw_members
        )
        if not members:
            raise ValueError("Retention archive members cannot be empty.")
        if len(members) != len(set(members)) or members != tuple(sorted(members)):
            raise ValueError("Retention archive members must be unique and sorted.")
        return cls(
            run_relative=run_relative,
            path=path,
            compression_level=compression_level,
            sha256=sha256,
            size_bytes=size_bytes,
            members=members,
        )

    @classmethod
    def validated_many(
        cls,
        values: object,
        *,
        plan_id: str,
        operations: Sequence[ArtifactRetentionOperation],
        archive_name: str | None,
    ) -> tuple[dict[str, Any], ...]:
        """Validate archive paths and membership against fingerprinted operations."""
        if cls._SHA256_PATTERN.fullmatch(plan_id) is None:
            raise ValueError("Retention archive plan_id must be a lowercase SHA-256.")
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
            raise TypeError("Retention archives must be a sequence.")
        operation_by_key = {operation.key: operation for operation in operations}
        records: list[ArtifactRetentionArchiveRecord] = []
        groups: set[tuple[str, int]] = set()
        paths: set[tuple[str, str]] = set()
        archived_keys: set[str] = set()
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise TypeError(f"Retention archive {index} must be an object.")
            record = cls.from_mapping(value)
            record._validate_path_identity(plan_id=plan_id, archive_name=archive_name)
            group = (record.run_relative, record.compression_level)
            if group in groups:
                raise ValueError(f"Retention archives contain duplicate group {group!r}.")
            groups.add(group)
            archive_path = (record.run_relative, record.path)
            if archive_path in paths:
                raise ValueError(f"Retention archives contain duplicate path {record.path!r}.")
            paths.add(archive_path)
            for member in record.members:
                key = f"{record.run_relative}/{member}"
                if key in archived_keys:
                    raise ValueError(f"Retention archive member {key!r} is duplicated.")
                operation = operation_by_key.get(key)
                if (
                    operation is None
                    or operation.action is not ArtifactRetentionAction.COMPRESS
                    or operation.compression_level != record.compression_level
                ):
                    raise ValueError(
                        f"Retention archive member {key!r} does not match a compression "
                        "operation in the plan."
                    )
                archived_keys.add(key)
            records.append(record)
        return tuple(record.to_dict() for record in records)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible archive record."""
        return copy.deepcopy(
            {
                "run_relative": self.run_relative,
                "path": self.path,
                "compression_level": self.compression_level,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "members": list(self.members),
            }
        )

    def _validate_path_identity(self, *, plan_id: str, archive_name: str | None) -> None:
        pure = PurePosixPath(self.path)
        if len(pure.parts) != 3 or pure.parts[:2] != (".lambdaforge", "retention"):
            raise ValueError(
                "Retention archive path must be directly under .lambdaforge/retention."
            )
        suffix = f"-l{self.compression_level}-{plan_id[:12]}.zip"
        filename = pure.name
        if archive_name is not None:
            expected = f"{Path(archive_name).stem}{suffix}"
            if filename != expected:
                raise ValueError(f"Retention archive path {self.path!r} does not match the plan.")
            return
        if not filename.endswith(suffix):
            raise ValueError(
                f"Retention archive path {self.path!r} does not match its plan_id and level."
            )
        stem = filename[: -len(suffix)]
        if self._ARCHIVE_STEM_PATTERN.fullmatch(stem) is None:
            raise ValueError(f"Retention archive filename {filename!r} is not portable.")

    @classmethod
    def _validated_relative_path(cls, value: object, *, label: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Retention archive {label} must be a string.")
        if not value or "\0" in value or "\\" in value:
            raise ValueError(f"Retention archive {label} must be a non-empty relative POSIX path.")
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
                f"Retention archive {label} is not a canonical relative POSIX path: {value!r}."
            )
        return value

    @staticmethod
    def _validated_integer(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Retention archive {label} must be an integer.")
        return value

    @classmethod
    def _validate_fields(cls, value: Mapping[str, Any]) -> None:
        keys = set(value)
        if keys == cls.FIELD_NAMES:
            return
        missing = sorted(cls.FIELD_NAMES - keys)
        unknown = sorted(str(key) for key in keys - cls.FIELD_NAMES)
        raise ValueError(
            f"Retention archive fields are malformed: missing={missing}, unknown={unknown}."
        )
