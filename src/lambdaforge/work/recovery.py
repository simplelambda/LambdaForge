"""Small, fail-closed recovery contract for an existing scientific execution."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.work.models import WorkResult


def read_owned_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"Recovery requires a regular persisted file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid recovery document: {path}")
    return value


def validate_execution(path: Path) -> dict[str, Any]:
    """Validate an exact execution directory, never a guessed latest sibling."""
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Recovery requires an absolute, non-symlinked execution directory.")
    manifest = read_owned_json(path / "execution.json")
    if manifest.get("execution_id") != path.name or not path.name.startswith("execution-"):
        raise ValueError("Recovery execution identity does not match its owned directory.")
    if manifest.get("ownership", {}).get("execution_dir") != "owned":
        raise ValueError("Recovery execution has no framework ownership evidence.")
    state = read_owned_json(path / "hpo-control" / "state.json")
    if state.get("execution_id") != path.name or int(state.get("state_version", 0)) < 4:
        raise ValueError("Recovery requires a compatible persisted HPO controller state.")
    for run in state.get("runs", ()):
        if not isinstance(run, Mapping):
            raise ValueError("Recovery Run inventory is corrupt.")
        result_path = Path(str(run.get("result_path", "")))
        if not result_path.is_relative_to(path / "runs") or result_path.resolve() != result_path:
            raise ValueError("Recovery Run result is outside its owned execution.")
        result = read_owned_json(result_path)
        if result.get("execution_id") != path.name:
            raise ValueError("Recovery Run belongs to another execution.")
    return manifest


def latest_outcomes(outcomes: Sequence[WorkResult]) -> tuple[WorkResult, ...]:
    """Resolve logical Run state while retaining physical Attempts in controller history."""
    latest: dict[tuple[Any, ...], WorkResult] = {}
    for result in outcomes:
        key = (
            result.name,
            int((result.trial or {"index": 0})["index"]),
            result.seed,
            result.study_phase,
            int((result.fidelity or {}).get("target", 0)),
        )
        latest[key] = result
    return tuple(latest.values())
