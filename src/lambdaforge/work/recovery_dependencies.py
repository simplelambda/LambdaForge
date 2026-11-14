"""Durable logical cache reachability and explicit bytes-only legacy recovery."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from lambdaforge.work.managed import ManagedFile, fingerprint, owned_path


def recovery_namespaces(run: Path) -> tuple[str, ...]:
    """Read exact Attempt namespace provenance, including pre-receipt legacy results."""
    names: set[str] = set()
    for attempt in (run / "attempts").glob("attempt-*"):
        for name in ("cache-namespace.json", "result.json"):
            path = attempt / name
            if path.is_file() and not path.is_symlink() and path.resolve() == path:
                record = json.loads(path.read_text())
                if (
                    record.get("run_id") != run.name
                    or record.get("execution_id") != run.parents[1].name
                ):
                    raise ValueError(
                        "Cache recovery namespace has foreign Run/Execution ownership."
                    )
                raw = record.get("cache_namespace") or str(
                    record.get("scientific_fingerprint", "")
                ).removeprefix("sha256:")
                if re.fullmatch(r"[a-f0-9]{64}", raw):
                    names.add(raw)
    return tuple(sorted(names))


def cache_recovery_references(jobs: Path) -> dict[str, list[str]]:
    """Conservative metadata closure over owned retained Runs; no cache payload reads.

    Legacy maps cannot identify an exact namespace, so every recorded revision namespace
    of that Run remains protected. Removing its owning Job releases those references.
    """
    references: dict[str, list[str]] = {}
    manifests = set(jobs.glob("job-*/work/.lambdaforge/runs/*/execution-*/execution.json"))
    for result in jobs.glob("job-*/result.json"):
        if result.resolve() != result:
            raise ValueError("Unsafe local recovery reference.")
        record = json.loads(result.read_text())
        raw = record.get("execution_dir")
        if isinstance(raw, str) and Path(raw).is_absolute():
            candidate = Path(raw) / "execution.json"
            if candidate.is_file() and candidate.resolve() == candidate:
                owner = json.loads(candidate.read_text())
                if owner.get("job_id") == result.parent.name:
                    manifests.add(candidate)
    for manifest in sorted(manifests):
        if manifest.resolve() != manifest:
            raise ValueError("Unsafe recovery dependency manifest.")
        value = json.loads(manifest.read_text())
        execution = manifest.parent
        if value.get("ownership", {}).get("execution_dir") != "owned":
            continue
        if value.get("execution_id") != execution.name:
            raise ValueError("Recovery dependency Execution identity mismatch.")
        for run in (execution / "runs").glob("run-*"):
            if run.resolve() != run:
                raise ValueError("Unsafe recovery dependency Run.")
            maps = run / "checkpoints/maps"
            # Receipt is written before entering consumer code, including a callback that
            # fetched files but failed before committing its first map result.
            for namespace in recovery_namespaces(run):
                if maps.is_dir() or (run / "checkpoints/cache-dependencies.json").is_file():
                    references.setdefault(namespace, []).append(str(run))
    return references


def recovery_dependency_preview(execution: Path, cache_root: Path) -> dict[str, Any]:
    """Inventory retained map references and exact artifact-copy candidates, never download.

    A present file is not hash-verified by this metadata plan. Legacy result records with
    no dependencies cannot prove their source identity and are reported separately.
    """
    from lambdaforge.work.recovery import read_owned_json

    execution = execution.absolute()
    manifest = read_owned_json(execution / "execution.json")
    if (
        manifest.get("execution_id") != execution.name
        or manifest.get("ownership", {}).get("execution_dir") != "owned"
    ):
        raise ValueError("Unowned recovery dependency inventory.")
    cache_root = cache_root.absolute()
    if cache_root.resolve() != cache_root:
        raise ValueError("Unsafe recovery cache root.")
    rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    unbound = 0
    for run in (execution / "runs").glob("run-*"):
        namespaces = recovery_namespaces(run)
        artifacts: dict[tuple[str, int], list[str]] = {}
        unregistered_sources: dict[int, list[str]] = {}
        for result in (run / "attempts").glob("*/result.json"):
            value = read_owned_json(result)
            if value.get("run_id") != run.name or value.get("execution_id") != execution.name:
                raise ValueError("Foreign recovery Attempt.")
            for artifact in value.get("artifacts", []):
                target = owned_path(result.parent, artifact["path"])
                if target.is_file():
                    artifacts.setdefault((artifact["sha256"], artifact["size_bytes"]), []).append(
                        str(target)
                    )
        for attempt in (run / "attempts").glob("attempt-*"):
            if attempt.resolve() != attempt:
                raise ValueError("Unsafe Attempt source inventory.")
            for target in (attempt / "artifacts").rglob("*"):
                if target.is_file() and not target.is_symlink() and target.resolve() == target:
                    unregistered_sources.setdefault(target.stat().st_size, []).append(str(target))
        for path in (run / "checkpoints/maps").glob("*/*.json"):
            record = read_owned_json(path)
            dependencies = record.get("dependencies", [])
            if not dependencies:
                unbound += 1
            for wrapper in dependencies:
                ref = wrapper.get("$lambdaforge_managed_file", {})
                if ref.get("scope") != "cache":
                    continue
                key, digest, size = ref["key"], ref["sha256"], ref["size_bytes"]
                if ref.get("namespace") is not None and ref["namespace"] not in namespaces:
                    raise ValueError("Managed reference names an unowned recovery namespace.")
                candidates = (
                    tuple([ref["namespace"]]) if ref.get("namespace") in namespaces else namespaces
                )
                available = []
                for namespace in candidates:
                    store = cache_root / "work" / namespace
                    receipt = (
                        store
                        / ".managed/records"
                        / (hashlib.sha256(key.encode()).hexdigest() + ".json")
                    )
                    target = owned_path(store / "files", key)
                    if receipt.is_file() and not receipt.is_symlink():
                        receipt_value = read_owned_json(receipt)
                        if (
                            (receipt_value.get("sha256"), receipt_value.get("size_bytes"))
                            == (digest, size)
                            and target.is_file()
                            and target.stat().st_size == size
                        ):
                            available.append(namespace)
                sources = artifacts.get((digest, size), []) or unregistered_sources.get(size, [])
                rows[(run.name, key, digest, size)] = {
                    "run_id": run.name,
                    "key": key,
                    "sha256": digest,
                    "size_bytes": size,
                    "namespaces": list(candidates),
                    "available_namespaces": available,
                    "artifact_candidates": sources,
                    "requires_original": ref.get("observed", True),
                    "state": "present_unverified"
                    if available
                    else "copy_candidate"
                    if sources
                    else "unavailable",
                }
    entries = list(rows.values())
    return {
        "execution_id": execution.name,
        "entries": entries,
        "present": sum(e["state"] == "present_unverified" for e in entries),
        "copy_candidates": sum(e["state"] == "copy_candidate" for e in entries),
        "unavailable": sum(e["state"] == "unavailable" for e in entries),
        "unavailable_observed": sum(
            e["state"] == "unavailable" and e["requires_original"] for e in entries
        ),
        "source_unbound_records": unbound,
        "policy": (
            "No network replay. Missing original bytes require a new acquisition. "
            "Counts are not integrity proof."
        ),
    }


def repair_recovery_dependencies(
    execution: Path, cache_root: Path, *, apply: bool = False
) -> dict[str, Any]:
    """Explicit native bytes-only migration; copy only exact persisted artifact identities."""
    if not apply:
        return recovery_dependency_preview(execution, cache_root)
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    with CrossProcessFileLock(
        execution / ".controller.lock",
        shared=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
        create=False,
    ):
        return _repair_owned_dependencies(execution, cache_root)


def _repair_owned_dependencies(execution: Path, cache_root: Path) -> dict[str, Any]:
    import inspect

    if "file_name" not in inspect.signature(fingerprint).parameters:
        raise ValueError("Update the managed LF runtime before applying exact legacy byte repair.")
    plan = recovery_dependency_preview(execution, cache_root)
    from lambdaforge.work.cache import WorkCache

    repaired = 0
    for row in plan["entries"]:
        if row["state"] != "copy_candidate" or not row["namespaces"]:
            continue
        source = next(
            (
                Path(candidate)
                for candidate in row["artifact_candidates"]
                if fingerprint(Path(candidate), file_name=Path(row["key"]).name)
                == (row["sha256"], row["size_bytes"])
            ),
            None,
        )
        if source is None:
            raise ValueError("Recovery artifact candidate failed exact identity verification.")
        cache = WorkCache(cache_root / "work" / row["namespaces"][0], hold_gc_lease=True)
        try:

            def build(target: Path, origin: Path = source) -> None:
                shutil.copyfile(origin, target)

            def validate(
                managed: ManagedFile, digest: str = row["sha256"], size: int = row["size_bytes"]
            ) -> bool:
                return (managed.sha256, managed.size_bytes) == (digest, size)

            cache.file(
                row["key"],
                build=build,
                validate=validate,
            )
            repaired += 1
        finally:
            cache.close()
    return {
        **plan,
        "applied": True,
        "repaired": repaired,
        "source_completeness": "not_established"
        if plan["source_unbound_records"] or plan["unavailable"]
        else "metadata_only",
        "scientific_computation_started": False,
    }
