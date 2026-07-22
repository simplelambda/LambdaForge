"""Immutable deterministic artifact-retention preview."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus


class ArtifactRetentionPlan(JsonResult):
    """Expose a read-only plan that never mutates the suite by itself."""

    VERSION = 1
    FIELD_NAMES = frozenset(
        {
            "retention_plan_version",
            "plan_id",
            "status",
            "base_dir",
            "receipt_id",
            "policy_fingerprint",
            "archive_name",
            "operations",
            "warnings",
            "reason",
        }
    )
    _SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
    _ARCHIVE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*[.]zip$")

    def __init__(
        self,
        *,
        status: ArtifactRetentionStatus | str,
        base_dir: str | Path,
        receipt_id: str | None,
        policy_fingerprint: str,
        archive_name: str,
        operations: Sequence[ArtifactRetentionOperation] = (),
        warnings: Sequence[str] = (),
        reason: str | None = None,
        plan_id: str | None = None,
    ) -> None:
        if not isinstance(status, (ArtifactRetentionStatus, str)):
            raise TypeError("Retention plan status must be a string or typed status.")
        self.status = (
            status
            if isinstance(status, ArtifactRetentionStatus)
            else ArtifactRetentionStatus(status)
        )
        if not isinstance(base_dir, (str, Path)):
            raise TypeError("Retention plan base_dir must be a string or Path.")
        self.base_dir = str(base_dir)
        if not self.base_dir or "\0" in self.base_dir:
            raise ValueError("Retention plan base_dir must be a non-empty filesystem path.")
        if receipt_id is not None:
            self._validate_sha256(receipt_id, label="receipt_id")
        self.receipt_id = receipt_id
        self._validate_sha256(policy_fingerprint, label="policy_fingerprint")
        self.policy_fingerprint = policy_fingerprint
        if not isinstance(archive_name, str):
            raise TypeError("Retention plan archive_name must be a string.")
        if self._ARCHIVE_PATTERN.fullmatch(archive_name) is None:
            raise ValueError("Retention plan archive_name must be a simple portable ZIP name.")
        self.archive_name = archive_name
        if isinstance(operations, (str, bytes, bytearray)) or not isinstance(
            operations,
            Sequence,
        ):
            raise TypeError("Retention plan operations must be a sequence.")
        if any(not isinstance(item, ArtifactRetentionOperation) for item in operations):
            raise TypeError(
                "Retention plan operations must contain ArtifactRetentionOperation objects."
            )
        self.operations = tuple(operations)
        operation_keys = [operation.key for operation in self.operations]
        if len(operation_keys) != len(set(operation_keys)):
            raise ValueError("Retention plan operations must have unique run-relative paths.")
        self.warnings = self._validated_strings(warnings, label="warnings")
        if reason is not None and not isinstance(reason, str):
            raise TypeError("Retention plan reason must be a string or None.")
        self.reason = reason
        computed_plan_id = self._compute_plan_id()
        if plan_id is not None:
            self._validate_sha256(plan_id, label="plan_id")
            if plan_id != computed_plan_id:
                raise ValueError("Retention plan_id does not match the canonical plan payload.")
        self.plan_id = computed_plan_id
        payload = FrozenJsonMapping(
            {
                "retention_plan_version": self.VERSION,
                "plan_id": self.plan_id,
                "status": self.status.value,
                "base_dir": self.base_dir,
                "receipt_id": self.receipt_id,
                "policy_fingerprint": self.policy_fingerprint,
                "archive_name": self.archive_name,
                "operations": [operation.to_dict() for operation in self.operations],
                "warnings": list(self.warnings),
                "reason": self.reason,
            }
        )
        self._freeze_mapping(dict(payload))

    @property
    def is_ready(self) -> bool:
        """Return whether this plan may enter the apply transaction."""
        return self.status is ArtifactRetentionStatus.PREVIEW

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionPlan:
        """Reconstruct a plan persisted in a recovery journal."""
        if not isinstance(value, Mapping):
            raise TypeError("Retention plan must be an object.")
        cls._validate_fields(value)
        version = value["retention_plan_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.VERSION:
            raise ValueError(
                f"Unsupported retention plan version {version!r}; expected {cls.VERSION}."
            )
        raw_operations = value["operations"]
        if isinstance(raw_operations, (str, bytes, bytearray)) or not isinstance(
            raw_operations,
            Sequence,
        ):
            raise TypeError("Retention plan operations must be a sequence.")
        parsed_operations: list[ArtifactRetentionOperation] = []
        for index, operation in enumerate(raw_operations):
            if not isinstance(operation, Mapping):
                raise TypeError(f"Retention plan operation {index} must be an object.")
            parsed_operations.append(ArtifactRetentionOperation.from_mapping(operation))
        warnings = cls._validated_strings(value["warnings"], label="warnings")
        return cls(
            status=value["status"],
            base_dir=value["base_dir"],
            receipt_id=value["receipt_id"],
            policy_fingerprint=value["policy_fingerprint"],
            archive_name=value["archive_name"],
            operations=tuple(parsed_operations),
            warnings=warnings,
            reason=value["reason"],
            plan_id=value["plan_id"],
        )

    def summary(self) -> str:
        """Render a compact human-readable preview."""
        bytes_selected = sum(operation.size_bytes for operation in self.operations)
        return (
            f"Retention {self.status.value}: operations={len(self.operations)}, "
            f"bytes_selected={bytes_selected}, plan_id={self.plan_id}"
            + (f", reason={self.reason}" if self.reason else "")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent ordinary JSON mapping."""
        return copy.deepcopy(dict(self))

    def _compute_plan_id(self) -> str:
        payload = {
            "version": self.VERSION,
            "base_dir": self.base_dir,
            "receipt_id": self.receipt_id,
            "policy_fingerprint": self.policy_fingerprint,
            "archive_name": self.archive_name,
            "operations": [operation.to_dict() for operation in self.operations],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _validate_fields(cls, value: Mapping[str, Any]) -> None:
        keys = set(value)
        if keys == cls.FIELD_NAMES:
            return
        missing = sorted(cls.FIELD_NAMES - keys)
        unknown = sorted(str(key) for key in keys - cls.FIELD_NAMES)
        raise ValueError(
            f"Retention plan fields are malformed: missing={missing}, unknown={unknown}."
        )

    @classmethod
    def _validate_sha256(cls, value: object, *, label: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"Retention plan {label} must be a string.")
        if cls._SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Retention plan {label} must be a lowercase SHA-256.")

    @staticmethod
    def _validated_strings(value: object, *, label: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError(f"Retention plan {label} must be a sequence of strings.")
        if any(not isinstance(item, str) for item in value):
            raise TypeError(f"Retention plan {label} must contain only strings.")
        return tuple(value)
