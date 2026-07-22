"""Typed durable result of one retention transaction."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.retention.ArtifactRetentionAction import ArtifactRetentionAction
from lambdaforge.experiments.retention.ArtifactRetentionArchiveRecord import (
    ArtifactRetentionArchiveRecord,
)
from lambdaforge.experiments.retention.ArtifactRetentionOperation import (
    ArtifactRetentionOperation,
)
from lambdaforge.experiments.retention.ArtifactRetentionStatus import ArtifactRetentionStatus


class ArtifactRetentionResult(JsonResult):
    """Record applied operations, immutable archives, bytes and conflicts."""

    VERSION = 1
    FIELD_NAMES = frozenset(
        {
            "retention_result_version",
            "plan_id",
            "status",
            "receipt_id",
            "policy_fingerprint",
            "operations",
            "archives",
            "selected_bytes",
            "reclaimed_bytes",
            "warnings",
            "errors",
        }
    )
    _OPERATION_FIELD_NAMES = ArtifactRetentionOperation.FIELD_NAMES | {"state"}
    _SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
    _STATE_ACTIONS = {
        "preserved_not_smaller": frozenset({ArtifactRetentionAction.COMPRESS}),
        "archived_and_removed": frozenset({ArtifactRetentionAction.COMPRESS}),
        "pruned": frozenset(
            {
                ArtifactRetentionAction.PRUNE,
                ArtifactRetentionAction.PRUNE_CHECKPOINT,
            }
        ),
    }

    def __init__(
        self,
        *,
        plan_id: str,
        status: ArtifactRetentionStatus | str,
        receipt_id: str | None,
        policy_fingerprint: str,
        operations: Sequence[Mapping[str, Any]] = (),
        archives: Sequence[Mapping[str, Any]] = (),
        selected_bytes: int = 0,
        reclaimed_bytes: int = 0,
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
    ) -> None:
        self.plan_id = str(plan_id)
        self.status = (
            status
            if isinstance(status, ArtifactRetentionStatus)
            else ArtifactRetentionStatus(status)
        )
        self.receipt_id = str(receipt_id) if receipt_id is not None else None
        self.policy_fingerprint = str(policy_fingerprint)
        self.operations = tuple(copy.deepcopy(dict(item)) for item in operations)
        self.archives = tuple(copy.deepcopy(dict(item)) for item in archives)
        self.selected_bytes = int(selected_bytes)
        self.reclaimed_bytes = int(reclaimed_bytes)
        self.warnings = tuple(str(item) for item in warnings)
        self.errors = tuple(str(item) for item in errors)
        payload = FrozenJsonMapping(
            {
                "retention_result_version": self.VERSION,
                "plan_id": self.plan_id,
                "status": self.status.value,
                "receipt_id": self.receipt_id,
                "policy_fingerprint": self.policy_fingerprint,
                "operations": list(self.operations),
                "archives": list(self.archives),
                "selected_bytes": self.selected_bytes,
                "reclaimed_bytes": self.reclaimed_bytes,
                "warnings": list(self.warnings),
                "errors": list(self.errors),
            }
        )
        self._freeze_mapping(dict(payload))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArtifactRetentionResult:
        """Parse a previously published retention manifest."""
        if not isinstance(value, Mapping):
            raise TypeError("Retention result must be an object.")
        cls._validate_fields(value)
        version = value["retention_result_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != cls.VERSION:
            raise ValueError(
                f"Unsupported retention result version {version!r}; expected {cls.VERSION}."
            )
        plan_id = cls._validated_sha256(value["plan_id"], label="plan_id")
        receipt_id = value["receipt_id"]
        if receipt_id is not None:
            receipt_id = cls._validated_sha256(receipt_id, label="receipt_id")
        policy_fingerprint = cls._validated_sha256(
            value["policy_fingerprint"],
            label="policy_fingerprint",
        )
        status = value["status"]
        if not isinstance(status, str):
            raise TypeError("Retention result status must be a string.")
        ArtifactRetentionStatus(status)
        operation_rows, operations = cls._validated_operation_rows(value["operations"])
        archives = ArtifactRetentionArchiveRecord.validated_many(
            value["archives"],
            plan_id=plan_id,
            operations=operations,
            archive_name=None,
        )
        cls._validate_archive_states(operation_rows, operations, archives)
        selected_bytes = cls._validated_nonnegative_integer(
            value["selected_bytes"],
            label="selected_bytes",
        )
        reclaimed_bytes = cls._validated_nonnegative_integer(
            value["reclaimed_bytes"],
            label="reclaimed_bytes",
        )
        if reclaimed_bytes > selected_bytes:
            raise ValueError("Retention result reclaimed_bytes cannot exceed selected_bytes.")
        return cls(
            plan_id=plan_id,
            status=status,
            receipt_id=receipt_id,
            policy_fingerprint=policy_fingerprint,
            operations=operation_rows,
            archives=archives,
            selected_bytes=selected_bytes,
            reclaimed_bytes=reclaimed_bytes,
            warnings=cls._validated_strings(value["warnings"], label="warnings"),
            errors=cls._validated_strings(value["errors"], label="errors"),
        )

    @classmethod
    def read_json(cls, path: str) -> ArtifactRetentionResult:
        """Load a published result from disk."""
        import json

        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("Retention result JSON must contain an object.")
        return cls.from_mapping(payload)

    def summary(self) -> str:
        """Render a compact human-readable transaction outcome."""
        return (
            f"Retention {self.status.value}: operations={len(self.operations)}, "
            f"selected_bytes={self.selected_bytes}, reclaimed_bytes={self.reclaimed_bytes}, "
            f"plan_id={self.plan_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent ordinary JSON mapping."""
        return copy.deepcopy(dict(self))

    @classmethod
    def _validated_operation_rows(
        cls,
        value: object,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[ArtifactRetentionOperation, ...]]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError("Retention result operations must be a sequence.")
        rows: list[dict[str, Any]] = []
        operations: list[ArtifactRetentionOperation] = []
        keys: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise TypeError(f"Retention result operation {index} must be an object.")
            item_keys = set(item)
            if item_keys != cls._OPERATION_FIELD_NAMES:
                missing = sorted(cls._OPERATION_FIELD_NAMES - item_keys)
                unknown = sorted(str(key) for key in item_keys - cls._OPERATION_FIELD_NAMES)
                raise ValueError(
                    "Retention result operation fields are malformed: "
                    f"missing={missing}, unknown={unknown}."
                )
            operation = ArtifactRetentionOperation.from_mapping(
                {name: item[name] for name in ArtifactRetentionOperation.FIELD_NAMES}
            )
            state = item["state"]
            if not isinstance(state, str):
                raise TypeError("Retention result operation state must be a string.")
            valid_actions = cls._STATE_ACTIONS.get(state)
            if valid_actions is None or operation.action not in valid_actions:
                raise ValueError(
                    f"Retention result state {state!r} is invalid for "
                    f"action {operation.action.value!r}."
                )
            if operation.key in keys:
                raise ValueError(f"Retention result operation {operation.key!r} is duplicated.")
            keys.add(operation.key)
            row = operation.to_dict()
            row["state"] = state
            rows.append(row)
            operations.append(operation)
        return tuple(rows), tuple(operations)

    @classmethod
    def _validate_archive_states(
        cls,
        rows: tuple[dict[str, Any], ...],
        operations: tuple[ArtifactRetentionOperation, ...],
        archives: tuple[dict[str, Any], ...],
    ) -> None:
        archived_keys = {
            f"{archive['run_relative']}/{member}"
            for archive in archives
            for member in archive["members"]
        }
        for row, operation in zip(rows, operations, strict=True):
            if operation.action is not ArtifactRetentionAction.COMPRESS:
                continue
            expected = (
                "archived_and_removed"
                if operation.key in archived_keys
                else "preserved_not_smaller"
            )
            if row["state"] != expected:
                raise ValueError(
                    f"Retention result state for {operation.key!r} disagrees with archives."
                )

    @classmethod
    def _validate_fields(cls, value: Mapping[str, Any]) -> None:
        keys = set(value)
        if keys == cls.FIELD_NAMES:
            return
        missing = sorted(cls.FIELD_NAMES - keys)
        unknown = sorted(str(key) for key in keys - cls.FIELD_NAMES)
        raise ValueError(
            f"Retention result fields are malformed: missing={missing}, unknown={unknown}."
        )

    @classmethod
    def _validated_sha256(cls, value: object, *, label: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"Retention result {label} must be a string.")
        if cls._SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"Retention result {label} must be a lowercase SHA-256.")
        return value

    @staticmethod
    def _validated_nonnegative_integer(value: object, *, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Retention result {label} must be an integer.")
        if value < 0:
            raise ValueError(f"Retention result {label} must be non-negative.")
        return value

    @staticmethod
    def _validated_strings(value: object, *, label: str) -> tuple[str, ...]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError(f"Retention result {label} must be a sequence of strings.")
        if any(not isinstance(item, str) for item in value):
            raise TypeError(f"Retention result {label} must contain only strings.")
        return tuple(value)
