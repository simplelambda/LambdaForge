"""Compact Attempt-owned bulk data while preserving durable scientific evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lambdaforge.work.managed import fingerprint, owned_path
from lambdaforge.work.models import WorkResult, atomic_json


def compact_attempt(result: WorkResult) -> dict[str, Any]:
    """Remove only failed partial outputs or verified published-only duplicates.

    Result envelopes, logs, metrics, environment provenance and checkpoints remain.
    The operation is idempotent and refuses to discard an internal artifact unless
    its published destination still has the recorded content identity.
    """
    run_dir = result.run_dir.resolve()
    plan = retention_plan(result)
    removed: list[str] = []
    reclaimed = int(plan["reclaimable_bytes"])
    artifacts_root = owned_path(run_dir, "artifacts")
    for relative in plan["paths"]:
        source = owned_path(run_dir, str(relative), must_exist=True)
        _remove(source)
        removed.append(str(relative))
    if result.ok:
        if artifacts_root.is_dir() and not artifacts_root.is_symlink():
            _remove_empty(artifacts_root)
    receipt = {
        "retention_version": 1,
        "status": result.status,
        "removed": removed,
        "reclaimed_bytes": reclaimed,
        "preserved": [
            "result.json",
            result.logs,
            result.environment_manifest,
            "metrics.jsonl",
            "checkpoints",
            "unpublished artifacts",
        ],
    }
    atomic_json(run_dir / "retention.json", receipt)
    return receipt


def retention_plan(result: WorkResult) -> dict[str, Any]:
    """Return exact currently safe-to-remove Attempt paths without mutation."""
    run_dir = result.run_dir.resolve()
    artifacts_root = owned_path(run_dir, "artifacts")
    if not result.ok:
        failed_paths: tuple[str, ...] = ("artifacts",) if artifacts_root.exists() else ()
        return {
            "paths": failed_paths,
            "reclaimable_bytes": _size(artifacts_root) if failed_paths else 0,
            "reason": "failed-attempt",
        }
    planned_paths: list[str] = []
    reclaimed = 0
    for artifact in result.artifacts:
        if artifact.metadata.get("retention") != "published-only":
            continue
        source = owned_path(run_dir, artifact.path)
        if not source.exists():
            continue
        published_value = artifact.metadata.get("published_to")
        if not isinstance(published_value, str) or not published_value:
            continue
        published = Path(published_value)
        if published.is_symlink() or not published.exists():
            continue
        if published.resolve() == source.resolve():
            continue
        try:
            digest, size = fingerprint(published)
        except (OSError, ValueError):
            continue
        if digest != artifact.sha256 or size != artifact.size_bytes:
            continue
        planned_paths.append(artifact.path)
        reclaimed += _size(source)
    return {
        "paths": tuple(planned_paths),
        "reclaimable_bytes": reclaimed,
        "reason": "verified-published-duplicate",
    }


def compact_incomplete_attempt(attempt_dir: Path) -> dict[str, Any]:
    """Remove an unfinalized artifact tree from a terminal interrupted Attempt."""
    directory = attempt_dir.resolve()
    artifacts = owned_path(directory, "artifacts")
    reclaimed = _size(artifacts) if artifacts.exists() else 0
    if artifacts.exists():
        _remove(artifacts)
    receipt = {
        "retention_version": 1,
        "status": "interrupted",
        "removed": ["artifacts"] if reclaimed else [],
        "reclaimed_bytes": reclaimed,
        "preserved": ["logs", "metrics", "environment provenance", "checkpoints"],
    }
    atomic_json(directory / "retention.json", receipt)
    return receipt


def load_work_result(path: Path) -> WorkResult | None:
    """Return a persisted WorkResult using the canonical runner decoder when valid."""
    if not path.is_file() or path.is_symlink():
        return None
    try:
        from lambdaforge.work.runner import _work_result_from_mapping

        value = json.loads(path.read_text(encoding="utf-8"))
        return _work_result_from_mapping(value) if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _remove(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Retention cleanup refuses symbolic links: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _remove_empty(path: Path) -> None:
    for child in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            try:
                child.rmdir()
            except OSError:
                pass
    try:
        path.rmdir()
    except OSError:
        pass


def _size(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )


__all__ = [
    "compact_attempt",
    "compact_incomplete_attempt",
    "load_work_result",
    "retention_plan",
]
