"""Reference-based terminal checkpoint retention, independent of consumer model policy."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.managed import fingerprint, owned_path
from lambdaforge.work.models import atomic_json


def checkpoint_plan(execution: Path, *, grace_seconds: float) -> list[dict[str, Any]]:
    """Caller holds the Execution writer lock and has verified terminal Job ownership.

    Missing/corrupt aggregate or policy stays protected. Failed/interrupted Studies keep
    recovery state. A verified explicit publication can release named intermediates sooner.
    """
    result_path = owned_path(execution, "result.json")
    if not result_path.is_file():
        return []
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(result, dict) or result.get("status") != "succeeded":
        return []
    expired = time.time() - result_path.stat().st_mtime >= grace_seconds
    candidates: list[dict[str, Any]] = []
    for run in (execution / "runs").glob("run-*"):
        if run.is_symlink() or not run.is_dir():
            continue
        root = owned_path(execution, run / "checkpoints")
        if not root.is_dir():
            continue
        policy_path = owned_path(root, ".storage-policy.json")
        policy: dict[str, Any] = {"storage_policy_version": 1}
        if policy_path.exists():
            try:
                policy = json.loads(policy_path.read_text())
                if not isinstance(policy, dict) or policy.get("storage_policy_version") != 1:
                    continue
                if not isinstance(policy.get("pins", {}), dict):
                    continue
            except (OSError, ValueError):
                continue
        try:
            pins = [owned_path(root, name) for name in policy.get("pins", {})]
        except (TypeError, ValueError):
            continue  # Corrupt retention evidence protects the complete collection.
        publications = policy.get("publications", {})
        if not isinstance(publications, dict):
            continue
        for child in root.iterdir():
            if child.name.startswith(".") or child.is_symlink():
                continue
            if any(
                pin == child or pin.is_relative_to(child) or child.is_relative_to(pin)
                for pin in pins
            ):
                continue
            reason = "terminal-unreferenced-checkpoint" if expired else None
            declared = publications.get(child.name)
            if isinstance(declared, Mapping):
                try:
                    if (declared.get("sha256"), declared.get("size_bytes")) == fingerprint(child):
                        if _publication_valid(declared.get("durable")):
                            reason = "verified-publication-released-checkpoint"
                except (OSError, ValueError):
                    pass
            if reason is None:
                continue
            # Do not remove trees containing unowned links/special entries.
            try:
                digest, size = fingerprint(child)
            except (OSError, ValueError):
                continue
            candidates.append(
                {
                    "category": "checkpoints",
                    "path": str(child),
                    "bytes": size,
                    "sha256": digest,
                    "reason": reason,
                    "execution": str(execution),
                    "run_id": run.name,
                }
            )
    return candidates


def compact_checkpoints(execution: Path, candidates: list[dict[str, Any]]) -> int:
    """Idempotently discard proven redundant checkpoint bytes, retaining an audit receipt."""
    receipt_path = owned_path(execution, "checkpoint-retention.json")
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"removed": []}
    if not isinstance(prior, dict) or not isinstance(prior.get("removed"), list):
        raise ValueError("Invalid checkpoint retention journal; refusing deletion.")
    # Resume only already renamed, exact owned trash. Never replay an uncommitted
    # intent against an original path that may now have a new pin or publication.
    for item in prior.get("pending", []):
        trash = owned_path(execution, item["trash"])
        relative = trash.relative_to(execution)
        if (
            len(relative.parts) != 4
            or relative.parts[0] != "runs"
            or not relative.parts[1].startswith("run-")
            or relative.parts[2] != "checkpoints"
            or not relative.parts[3].startswith(".retired-")
        ):
            raise ValueError("Unsafe checkpoint retention trash; refusing deletion.")
        if trash.exists():
            _remove(trash)
            prior["removed"].append({key: value for key, value in item.items() if key != "trash"})
    prior["pending"] = []
    if receipt_path.exists():
        atomic_json(receipt_path, prior)
    removed: list[dict[str, Any]] = []
    for item in candidates:
        path = owned_path(execution, str(item["path"]))
        with CrossProcessFileLock(
            path.parent / ".storage-policy.lock",
            shared=False,
            timeout_seconds=0.1,
            poll_interval_seconds=0.01,
        ):
            if not path.exists():
                continue
            # Pins may change between preview and apply. Recheck exact evidence
            # under the same metadata lock used by pin/unpin.
            if not _still_eligible(path, item):
                continue
            trash = owned_path(path.parent, f".retired-{uuid4().hex}")
            pending = {**item, "trash": str(trash)}
            prior.update(
                {
                    "checkpoint_retention_version": 1,
                    "updated_at": time.time(),
                    "pending": [pending],
                    "preserved": ["Attempt results", "logs", "metrics", "provenance", "pins"],
                }
            )
            atomic_json(receipt_path, prior)
            os.rename(path, trash)
            _remove(trash)
            prior["removed"].append(dict(item))
            prior["pending"] = []
            atomic_json(receipt_path, prior)
            removed.append(item)
    return sum(int(item["bytes"]) for item in removed)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _still_eligible(path: Path, item: Mapping[str, Any]) -> bool:
    """Recheck only this checkpoint, never rehash every sibling for every deletion."""
    policy_path = owned_path(path.parent, ".storage-policy.json")
    try:
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        if (
            not isinstance(policy, dict)
            or policy.get("storage_policy_version", 1) != 1
            or not isinstance(policy.get("pins", {}), dict)
        ):
            return False
        for name in policy.get("pins", {}):
            pin = owned_path(path.parent, name)
            if pin == path or pin.is_relative_to(path) or path.is_relative_to(pin):
                return False
        if fingerprint(path) != (item["sha256"], item["bytes"]):
            return False
        if item["reason"] == "verified-publication-released-checkpoint":
            return _publication_valid(
                policy.get("publications", {}).get(path.name, {}).get("durable")
            )
        return item["reason"] == "terminal-unreferenced-checkpoint"
    except (OSError, TypeError, ValueError, AttributeError):
        return False


def _publication_valid(witness: Any) -> bool:
    if not isinstance(witness, Mapping):
        return False
    if witness.get("kind") == "artifact":
        path = Path(str(witness.get("path", "")))
        return (
            path.is_absolute()
            and not path.is_symlink()
            and path.exists()
            and fingerprint(path)
            == (
                witness.get("sha256"),
                witness.get("size_bytes"),
            )
        )
    if witness.get("kind") == "dataset":
        from lambdaforge.data.DatasetOperations import DatasetOperations

        for placement in witness.get("placements", ()):
            root = Path(str(placement.get("root", "")))
            if root.is_absolute() and not root.is_symlink() and root.is_dir():
                if DatasetOperations.verify(root, str(witness["dataset_id"])).get("valid"):
                    return True
    return False
