"""Pinned read-only historical evidence, distinct from independent scientific products."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work.result_projection import attempt_path, curves, owned, projection, read_mapping


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultRequirement:
    """Human selector before resolution; exact immutable evidence identity after pinning."""

    execution: str
    run: str | None = None
    attempt: int | None = None
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution, str)
            or not self.execution.strip()
            or len(self.execution) > 256
        ):
            raise ValueError("Result input requires a non-empty historical Execution selector.")
        if self.run is not None and (
            not isinstance(self.run, str) or not re.fullmatch(r"run-[A-Za-z0-9_-]+", self.run)
        ):
            raise ValueError("Result input Run must be an exact native Run ID.")
        if self.attempt is not None and (
            type(self.attempt) is not int or self.attempt < 1 or self.run is None
        ):
            raise ValueError("Result input Attempt requires a positive integer and exact Run.")
        if self.evidence_id is not None and (
            not isinstance(self.evidence_id, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.evidence_id)
        ):
            raise ValueError("Result input evidence_id must be an exact SHA-256 identity.")

    @classmethod
    def from_mapping(cls, value: Any) -> ResultRequirement:
        if (
            not isinstance(value, Mapping)
            or set(value) - {"execution", "run", "attempt", "evidence_id"}
            or "execution" not in value
        ):
            raise ValueError("Result input requires {result: {execution: NAME_OR_ID}}.")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("execution", self.execution),
                ("run", self.run),
                ("attempt", self.attempt),
                ("evidence_id", self.evidence_id),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ResultInput:
    """Immutable metadata; selected artifact access verifies bytes explicitly.

    Evidence is not promoted: removing the producer can make this dependency unavailable.
    Use ProductInput for independent model/report bytes that must outlive their Execution.
    Imported evidence remains read-only and is never executable recovery.
    """

    requirement: ResultRequirement
    metadata: Mapping[str, Any]
    configuration: Mapping[str, Any]
    _root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", FrozenJsonMapping(self.metadata))
        object.__setattr__(self, "configuration", FrozenJsonMapping(self.configuration))

    @property
    def execution_id(self) -> str:
        return str(self.metadata["execution_id"])

    @property
    def evidence_id(self) -> str:
        assert self.requirement.evidence_id is not None
        return self.requirement.evidence_id

    @property
    def selected(self) -> Mapping[str, Any]:
        rows = [
            row
            for row in self.metadata.get("runs", ())
            if self.requirement.run is None or row["run_id"] == self.requirement.run
        ]
        if len({row["run_id"] for row in rows}) > 1:
            raise ValueError(
                "Select an exact Run to access metrics/results/artifacts; Runs are never merged."
            )
        if self.requirement.attempt is not None:
            rows = [row for row in rows if row["attempt_number"] == self.requirement.attempt]
        if not rows:
            raise KeyError("The selected historical Run/Attempt has no persisted evidence.")
        return max(rows, key=lambda row: row["attempt_number"])

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.selected.get("metrics", FrozenJsonMapping({}))

    @property
    def result(self) -> Any:
        return self.selected.get("result")

    @property
    def outputs(self) -> Mapping[str, Any]:
        return self.selected.get("outputs", FrozenJsonMapping({}))

    def metric_curves(self, *, points: int = 200) -> Mapping[str, Any]:
        self.verify()
        return FrozenJsonMapping(curves(attempt_path(self._root, self.selected), points=points))

    def verify(self) -> None:
        """Detect changed authoritative evidence after preview/pinning, without substitution."""
        persisted = read_mapping(owned(self._root, "result.json"), aggregate=True)
        configuration = read_mapping(owned(self._root, "configuration.json"))
        if _digest({"result": persisted, "configuration": configuration}) != self.evidence_id:
            raise ValueError(
                "Historical result evidence changed after it was pinned; reselect explicitly."
            )

    def artifact(self, name: str) -> Path:
        """Verify exactly the chosen retained file/directory; never load other artifacts."""
        from lambdaforge.work.managed import fingerprint

        self.verify()
        descriptors = [item for item in self.selected.get("artifacts", ()) if item["name"] == name]
        if len(descriptors) != 1:
            raise KeyError(f"No unique registered artifact {name!r} in the selected Attempt.")
        descriptor = descriptors[0]
        path = owned(attempt_path(self._root, self.selected), descriptor["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Historical artifact {name!r} is not retained; use a published ProductInput."
            )
        if fingerprint(path) != (descriptor["sha256"], descriptor["size_bytes"]):
            raise ValueError(f"Historical artifact content differs: {name}.")
        return path


def resolve_result_input(
    value: Any, source_dir: Path, *, results_root: Path | None = None
) -> tuple[ResultInput, Path]:
    """Resolve known native/imported records only, never arbitrary historical directories."""
    from lambdaforge.work.ResultStore import ResultStore

    requirement = ResultRequirement.from_mapping(value)
    configured = (
        str(results_root) if results_root else os.environ.get("LAMBDAFORGE_RESULT_INPUT_ROOT")
    )
    if configured is None and os.environ.get("LAMBDAFORGE_EXECUTION_MODE") == "worker":
        raise ValueError(
            "Worker historical evidence root is missing; "
            "explicitly materialize the verified export on the destination."
        )
    root = (
        Path(configured)
        if configured
        else ProjectContext.discover(source_dir).root / ".lambdaforge/runs"
    )
    store = ResultStore(root)
    selected = store.select(requirement.execution)
    if selected.get("already_deleted"):
        raise ValueError(
            "Historical Execution was deleted; choose an independent published product."
        )
    directory = store._evidence_dir(Path(selected["_manifest_path"]))
    manifest = owned(directory, "result.json")
    persisted = read_mapping(manifest, aggregate=True)
    configuration = read_mapping(owned(directory, "configuration.json"))
    projection(directory, persisted)
    if persisted.get("status") not in {
        "succeeded",
        "failed",
        "completed_with_failures",
        "cancelled",
        "interrupted",
        "pruned",
    }:
        raise ValueError(
            "Historical dependencies require a finalized snapshot, not mutable live evidence."
        )
    evidence_id = _digest({"result": persisted, "configuration": configuration})
    if requirement.evidence_id is not None and requirement.evidence_id != evidence_id:
        raise ValueError("Historical result identity differs from the frozen dependency.")
    pinned = ResultRequirement(
        str(persisted["execution_id"]), requirement.run, requirement.attempt, evidence_id
    )
    result = ResultInput(pinned, persisted, configuration, directory)
    if pinned.run is not None:
        projection(directory, persisted, view="outputs", run_id=pinned.run, attempt=pinned.attempt)
    return result, manifest


def result_identity(result: ResultInput) -> dict[str, Any]:
    return {
        "result": result.execution_id,
        "evidence_id": result.evidence_id,
        "run": result.requirement.run,
        "attempt": result.requirement.attempt,
    }
