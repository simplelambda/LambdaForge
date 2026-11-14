"""Bounded remote-side storage inspection and reference-aware cache deletion."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class StorageOperations:
    """Operate only below exact configured internal roots."""

    @classmethod
    def status(cls, descriptor: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        from lambdaforge.controlplane.StorageAdmission import StorageAdmission

        roots = cls._roots(descriptor)
        if descriptor.get("metadata_only"):
            ledger = cls._read_json(roots["state"] / "storage-ledger.json") or {}
            result = {
                name: {
                    **ledger.get("categories", {}).get(name, {}),
                    "path": str(path),
                    "exists": path.exists(),
                    "measurement": "cached"
                    if name in ledger.get("categories", {})
                    else "not-measured",
                    "measured_at": ledger.get("measured_at"),
                }
                for name, path in roots.items()
            }
        else:
            result = {name: cls._usage(path) for name, path in roots.items()}
        result["filesystems"] = StorageAdmission.observe(descriptor)
        return result

    @classmethod
    def reconcile(cls, descriptor: Mapping[str, Any], *, apply: bool = False) -> dict[str, Any]:
        """Explicit deep measurement and ledger drift check; never delete scientific bytes.

        The ledger is diagnostic, not a replacement for physical statvfs admission. Root
        categories may overlap: their sizes must not be added as independent disk usage.
        """
        roots = cls._roots(descriptor)
        state = roots["state"]
        ledger = state / "storage-ledger.json"
        if ledger.is_symlink() or ledger.exists() and cls._read_json(ledger) is None:
            raise ValueError(
                f"Storage ledger is unreadable or corrupt: {ledger}. "
                "Inspect or move this diagnostic ledger before reconciling; "
                "scientific records have not been changed."
            )
        # Readers do not create metadata. An explicit apply serializes the full
        # measurement with another reconcile so older snapshots cannot win last.
        if apply:
            state.mkdir(parents=True, exist_ok=True)
            with CrossProcessFileLock(
                state / ".storage-ledger.lock",
                shared=False,
                timeout_seconds=10,
                poll_interval_seconds=0.05,
            ):
                return cls._reconcile_locked(descriptor, ledger, apply=True)
        return cls._reconcile_locked(descriptor, ledger, apply=False)

    @classmethod
    def _reconcile_locked(
        cls, descriptor: Mapping[str, Any], ledger: Path, *, apply: bool
    ) -> dict[str, Any]:
        from lambdaforge.work.models import atomic_json

        previous = cls._read_json(ledger)
        if ledger.exists() and previous is None:
            raise ValueError(f"Storage ledger changed or is corrupt: {ledger}.")
        if previous is not None and (
            previous.get("storage_ledger_version") != 1
            or not isinstance(previous.get("categories"), Mapping)
            or any(
                not isinstance(value, Mapping)
                or any(
                    isinstance(value.get(key), bool)
                    or not isinstance(value.get(key), int)
                    or value[key] < 0
                    for key in ("bytes", "files")
                )
                for value in previous.get("categories", {}).values()
            )
        ):
            raise ValueError(f"Unsupported storage ledger: {ledger}.")
        measured = cls.status(descriptor)
        filesystems = measured.pop("filesystems")
        # Updating the diagnostic ledger must not report its own writes as drift.
        state_usage = measured["state"]
        for metadata in (ledger, ledger.parent / ".storage-ledger.lock"):
            usage = cls._usage(metadata)
            for key in ("bytes", "allocated_bytes", "files"):
                state_usage[key] = max(0, state_usage.get(key, 0) - usage.get(key, 0))
        old = previous.get("categories", {}) if previous else {}
        drift = {
            name: {
                "bytes_delta": int(usage["bytes"]) - int(old.get(name, {}).get("bytes", 0)),
                "files_delta": int(usage["files"]) - int(old.get(name, {}).get("files", 0)),
            }
            for name, usage in measured.items()
        }
        snapshot = {
            "storage_ledger_version": 1,
            "measured_at": time.time(),
            "categories": measured,
            "filesystems": filesystems,
        }
        if apply:
            atomic_json(ledger, snapshot)
        return {
            **snapshot,
            "baseline_available": previous is not None,
            "previous_measured_at": previous.get("measured_at") if previous else None,
            "drift": drift,
            "applied": apply,
            "ledger_path": str(ledger),
            "note": (
                "Overlapping categories are not additive; physical free space is authoritative."
            ),
        }

    @classmethod
    def gc(
        cls,
        descriptor: Mapping[str, Any],
        references: Mapping[str, Sequence[str]],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Serialize cache collection against another collector on the same cluster."""
        if not apply:
            # Advisory snapshot only. Apply recomputes under ownership locks; a preview must
            # not create an otherwise absent cache root or a persistent lock file.
            return {**cls._gc(descriptor, references, apply=False), "apply_revalidates": True}
        roots = cls._roots(descriptor)
        cache_root = roots["environments"].parent
        cache_root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            cache_root / ".gc.lock",
            shared=False,
            timeout_seconds=0.1,
            poll_interval_seconds=0.1,
        ):
            if apply:
                cls._resume_deletions(roots)
            return cls._gc(descriptor, references, apply=apply)

    @classmethod
    def prune_environments(
        cls,
        descriptor: Mapping[str, Any],
        protected: Sequence[str],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Remove only obsolete LambdaForge environments after a verified replacement exists."""
        roots = cls._roots(descriptor)
        cache_root = roots["environments"].parent
        if apply:
            cache_root.mkdir(parents=True, exist_ok=True)
        lock = CrossProcessFileLock(
            cache_root / ".gc.lock",
            shared=False,
            timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        )
        try:
            if apply:
                lock.acquire()
        except TimeoutError:
            return {
                "candidates": [],
                "pruned": [],
                "applied": False,
                "blocked_reason": "Environment cleanup deferred while the cache is in use.",
            }
        try:
            markers = tuple(cache_root.glob(".environment-build-*.lock"))
            if markers:
                return {
                    "candidates": [],
                    "pruned": [],
                    "applied": False,
                    "blocked_reason": "Another managed environment build is active.",
                }
            environment_root = roots["environments"]
            keep = {str(value) for value in protected}
            keep.update(cls._active_environment_references(roots))
            candidates: list[dict[str, Any]] = []
            if environment_root.is_dir() and not environment_root.is_symlink():
                for child in sorted(environment_root.iterdir()):
                    if (
                        child.is_symlink()
                        or not child.is_dir()
                        or child.name in keep
                        or "*" in keep
                        or not child.name.startswith(("env-", ".env-"))
                        # Isolated tools are not obsolete main-environment replacements.
                        # Native explicit GC owns their retention and use-lease checks.
                        or child.name.startswith(("env-tool-", ".env-tool-"))
                        or not cls._complete("environments", child)
                    ):
                        continue
                    usage = cls._usage(child)
                    candidates.append(
                        {
                            "environment_id": child.name,
                            "path": str(child),
                            "bytes": usage["bytes"],
                            "files": usage["files"],
                        }
                    )
            pruned: list[str] = []
            if apply:
                root = environment_root.resolve()
                for item in candidates:
                    path = Path(str(item["path"])).resolve()
                    if path == root or not path.is_relative_to(root):
                        raise RuntimeError(f"Environment cleanup target escaped its root: {path}")
                    cls._delete_cache_entry(path, environment_root)
                    pruned.append(str(item["environment_id"]))
            return {
                "candidates": candidates,
                "pruned": pruned,
                "applied": apply,
                "blocked_reason": None,
            }
        finally:
            if apply:
                lock.release()

    @classmethod
    def delete_job(
        cls, descriptor: Mapping[str, Any], job_id: str, *, apply: bool = False
    ) -> dict[str, Any]:
        """Preview or remove one exact LambdaForge-owned job workspace."""
        if not job_id.startswith("job-") or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in job_id
        ):
            raise ValueError("Invalid LambdaForge job id.")
        root = cls._roots(descriptor)["job_workspaces"].resolve()
        from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue

        for queue in ExecutionQueue.for_storage(cls._roots(descriptor)["state"], root):
            if job_id in queue.retained_jobs():
                raise ValueError("Job workspace is retained by a waiting execution queue ticket.")
        target = (root / job_id).resolve(strict=False)
        if target.parent != root or target.name != job_id:
            raise RuntimeError("Job cleanup target escaped the configured job root.")
        usage = cls._usage(target)
        candidate = {
            "category": "job_workspace",
            "name": job_id,
            **usage,
            "reason": "selected-work",
        }
        if apply and usage["exists"]:
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError(f"Unsafe job workspace: {target}")
            shutil.rmtree(target)
        return {
            "candidates": [candidate] if usage["exists"] else [],
            "reclaimable_bytes": int(usage["bytes"]),
            "applied": apply,
            "preserved": ["datasets", "shared caches", "other work", "cluster environment"],
        }

    @classmethod
    def compact_job(
        cls, descriptor: Mapping[str, Any], job_id: str, *, apply: bool = False
    ) -> dict[str, Any]:
        """Discard only terminal Attempt bulk outputs below one exact Job workspace.

        A successful published-only artifact is removed only after its published copy
        still matches the persisted SHA-256 and size.  Failed or interrupted Attempts
        lose their partial ``artifacts`` tree, while logs, result envelopes, metrics,
        provenance and checkpoints remain available to the Research Console and ``lf logs``.
        """
        if not job_id.startswith("job-") or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in job_id
        ):
            raise ValueError("Invalid LambdaForge job id.")
        root = cls._roots(descriptor)["job_workspaces"].resolve()
        job = (root / job_id).resolve(strict=False)
        if job.parent != root or job.name != job_id or job.is_symlink():
            raise RuntimeError("Job compaction target escaped the configured job root.")
        work = job / "work"
        job_state = cls._read_json(job / "state.json")
        if job_state and job_state.get("state") not in {
            "succeeded",
            "failed",
            "cancelled",
            "timeout",
        }:
            return {
                "job_id": job_id,
                "candidates": [],
                "reclaimable_bytes": 0,
                "reclaimed_bytes": 0,
                "applied": False,
                "blocked_reason": "active-or-unknown-job",
            }
        candidates: list[dict[str, Any]] = []
        reclaimed = 0
        for manifest in work.glob(".lambdaforge/runs/*/execution-*/execution.json"):
            origin = cls._read_json(manifest) or {}
            aggregate = cls._read_json(manifest.parent / "result.json") or {}
            if (
                origin.get("ownership", {}).get("execution_dir") == "owned"
                and aggregate.get("status") != "succeeded"
                and any(manifest.parent.glob("runs/*/checkpoints/maps"))
            ):
                return {
                    "job_id": job_id,
                    "candidates": [],
                    "reclaimable_bytes": 0,
                    "reclaimed_bytes": 0,
                    "applied": False,
                    "blocked_reason": "retained-recovery-dependencies",
                }
        if work.is_dir() and not work.is_symlink():
            from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
            from lambdaforge.work.checkpoint_retention import checkpoint_plan, compact_checkpoints

            grace = ClusterStoragePolicy.from_mapping(
                descriptor, workspace=str(job)
            ).terminal_grace_seconds
            for execution in work.glob(".lambdaforge/runs/*/execution-*"):
                if execution.is_symlink() or execution.resolve() != execution:
                    continue
                lock = CrossProcessFileLock(
                    execution / ".controller.lock",
                    shared=False,
                    timeout_seconds=0.01,
                    poll_interval_seconds=0.005,
                )
                try:
                    if apply:
                        lock.acquire()
                    checkpoint_candidates = checkpoint_plan(execution, grace_seconds=grace)
                    candidates.extend({**item, "job_id": job_id} for item in checkpoint_candidates)
                    if apply:
                        reclaimed += compact_checkpoints(execution, checkpoint_candidates)
                except TimeoutError:
                    pass  # A recovery writer owns this Execution; never compete with it.
                finally:
                    if apply:
                        lock.release()
            from lambdaforge.work.retention import (
                compact_attempt,
                compact_incomplete_attempt,
                load_work_result,
                retention_plan,
            )

            for attempt in sorted(
                work.glob(".lambdaforge/runs/*/execution-*/runs/run-*/attempts/attempt-*")
            ):
                if (
                    attempt.is_symlink()
                    or not attempt.is_dir()
                    or not attempt.resolve().is_relative_to(work.resolve())
                ):
                    continue
                execution = attempt.parents[3]
                # A new recovery Job may be writing into this original terminal Job's
                # Execution. Its writer lock, not the old provider state, owns Attempts.
                writer = CrossProcessFileLock(
                    execution / ".controller.lock",
                    shared=False,
                    timeout_seconds=0.01,
                    poll_interval_seconds=0.005,
                )
                try:
                    if apply:
                        writer.acquire()
                except TimeoutError:
                    continue
                try:
                    result_path = attempt / "result.json"
                    result = load_work_result(result_path)
                    if result is not None and result.run_dir.resolve() != attempt.resolve():
                        result = None
                    # Existing but unreadable evidence is not proof of an interrupted Attempt.
                    # Fail closed: the only copy of a successful unpublished artifact may live here.
                    if result is None and result_path.exists():
                        continue
                    artifacts = attempt / "artifacts"
                    plan: dict[str, Any]
                    if result is not None:
                        plan = retention_plan(result)
                    else:
                        plan = {
                            "paths": ("artifacts",) if artifacts.exists() else (),
                            "reclaimable_bytes": int(cls._usage(artifacts)["bytes"]),
                            "reason": "interrupted-attempt",
                        }
                    reclaimable = int(plan["reclaimable_bytes"])
                    if reclaimable <= 0:
                        continue
                    planned_paths = tuple(str(value) for value in plan.get("paths", ()))
                    planned_candidates: list[dict[str, Any]] = []
                    for relative in planned_paths:
                        target = (attempt / relative).resolve(strict=False)
                        if not target.is_relative_to(attempt.resolve()):
                            raise RuntimeError(f"Attempt cleanup target escaped its root: {target}")
                        usage = cls._usage(target)
                        if not usage["exists"]:
                            continue
                        planned_candidates.append(
                            {
                                "category": "attempt_artifacts",
                                "job_id": job_id,
                                "attempt": attempt.name,
                                "path": str(target),
                                "bytes": int(usage["bytes"]),
                                "files": int(usage["files"]),
                                "reason": str(plan["reason"]),
                            }
                        )
                    if apply:
                        receipt = (
                            compact_attempt(result)
                            if result is not None
                            else compact_incomplete_attempt(attempt)
                        )
                        removed = int(receipt["reclaimed_bytes"])
                        if removed <= 0:
                            continue
                        reclaimed += removed
                    candidates.extend(planned_candidates)
                finally:
                    if apply:
                        writer.release()
        return {
            "job_id": job_id,
            "candidates": candidates,
            "reclaimable_bytes": sum(int(item["bytes"]) for item in candidates),
            "reclaimed_bytes": reclaimed,
            "applied": apply,
            "preserved": [
                "job state and logs",
                "result envelopes",
                "metrics and provenance",
                "required and pinned checkpoints",
                "unpublished successful artifacts",
            ],
        }

    @classmethod
    def _gc(
        cls,
        descriptor: Mapping[str, Any],
        references: Mapping[str, Sequence[str]],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Plan from current host references; a build protects only its own categories."""
        roots = cls._roots(descriptor)
        cache = roots["environments"].parent
        from lambdaforge.controlplane.StorageAdmission import StorageAdmission

        pressure_before = StorageAdmission.observe(descriptor)
        refs = {key: set(map(str, values)) for key, values in references.items()}
        if apply and descriptor.get("reference_repair_preview"):
            raise ValueError(
                "A hypothetical post-repair plan cannot authorize deletion; "
                "apply repairs and preview again."
            )
        preview = descriptor.get("reference_repair_preview") or {}
        inventory = cls.environment_references(roots, repair_preview=preview)
        from lambdaforge.work.recovery_dependencies import cache_recovery_references

        cache_chains = cache_recovery_references(roots["job_workspaces"])
        refs.setdefault("work_cache", set()).update(cache_chains)
        refs.setdefault("environments", set()).update(inventory["environments"])
        refs.setdefault("bundles", set()).update(inventory.get("tool_bundles", ()))
        terminal_jobs: set[str] = set(refs.get("terminal_jobs", ()))
        protected_jobs = refs.get("protected_jobs", set()) | set(
            inventory.get("protected_jobs", ())
        )
        for job in roots["job_workspaces"].glob("job-*"):
            if job.is_symlink() or not job.is_dir():
                continue
            state = cls._read_json(job / "state.json")
            request = cls._read_json(job / "request.json")
            terminal = state and state.get("state") in {
                "succeeded",
                "failed",
                "cancelled",
                "timeout",
            }
            if terminal and job.name not in protected_jobs:
                terminal_jobs.add(job.name)
            elif terminal:
                continue
            elif request is None and job.name not in inventory["retired_jobs"]:
                # Missing ownership is uncertainty, not an orphan.
                for category in ("bundles", "environments", "runtimes", "temporary"):
                    refs.setdefault(category, set()).add("*")
            elif request is not None:
                bundle = request.get("bundle_id")
                if bundle:
                    refs.setdefault("bundles", set()).add(str(bundle))
                refs.setdefault("temporary", set()).add("*")
        dependency_edges = []
        for receipt in roots["environments"].glob("*/.lambdaforge-environment.json"):
            record = cls._read_json(receipt) or {}
            dependency = record.get("dependency_environment")
            if isinstance(dependency, str):
                dependency_edges.append((receipt.parent.name, dependency))
        candidates: list[dict[str, Any]] = []
        protected_items: list[dict[str, Any]] = []
        for job_id in sorted(terminal_jobs - protected_jobs):
            if descriptor.get("environments_only"):
                break
            result = cls.compact_job(descriptor, job_id, apply=apply)
            candidates.extend(result["candidates"])
        markers = (*cache.glob(".environment-build-*.lock"), *cache.glob(".python-runtime-*.lock"))
        builds: set[str] = set()
        build_reasons: dict[str, str] = {}
        orphan_names: set[str] = set()
        repairs = cls._read_json(roots["state"] / "storage-reference-repairs.json") or {}
        for identity in repairs.get("retired_builds", {}):
            marker = cache / f".environment-build-{identity}.lock"
            if not marker.exists() and not cls._orphan_builder_process_present(marker):
                orphan_names.add(identity)
        for marker in markers:
            owner = cls._read_json(marker / "owner.json" if marker.is_dir() else marker) or {}
            for category in ("environments", "bundles", "runtimes"):
                refs.setdefault(category, set()).update(
                    owner.get("references", {}).get(category, ())
                )
            hypothetical = next(
                (
                    item
                    for item in preview.get("released_builds", ())
                    if item["name"] == marker.name
                ),
                None,
            )
            if hypothetical and hypothetical.get("owner") != cls._read_json(
                marker / "owner.json" if marker.is_dir() else marker
            ):
                raise ValueError("Build evidence changed after repair preview; inspect again.")
            if (
                cls._orphan_lease(marker)
                or hypothetical
                and not cls._orphan_builder_process_present(marker)
            ):
                orphan_names.add(
                    marker.name.removeprefix(".environment-build-")
                    .removeprefix(".python-runtime-")
                    .removesuffix(".lock")
                )
            else:
                import math

                heartbeat = owner.get("heartbeat")
                recent = (
                    isinstance(heartbeat, (int, float))
                    and math.isfinite(heartbeat)
                    and 0 <= time.time() - heartbeat < 60
                )
                build_reason = "environment-build" if recent else "unresolved-build-owner"
                build_id = (
                    marker.name.removeprefix(".environment-build-")
                    .removeprefix(".python-runtime-")
                    .removesuffix(".lock")
                )
                build_reasons[build_id] = build_reason
                builds.add(
                    marker.name.removeprefix(".environment-build-")
                    .removeprefix(".python-runtime-")
                    .removesuffix(".lock")
                )
                inventory["issues"].append(
                    {
                        "path": str(marker),
                        "reason": "unreadable-build-owner"
                        if not owner
                        else "build-owner-live-or-unresolved",
                        "owner_host": owner.get("host"),
                        "action": f"lf storage repair --release-build {marker.name}",
                    }
                )
        runtime_refs = cls._runtime_references(roots, tuple(refs.get("runtimes", ())))
        for code, dependency in dependency_edges:
            # Automatic quota plans can retain unselected code layers; explicit clean
            # selects all obsolete code layers and can collect their base in the same pass.
            if (
                descriptor.get("automatic_gc")
                or "*" in refs["environments"]
                or code in refs["environments"]
                or code in builds
            ):
                refs["environments"].add(dependency)
        categories: tuple[str, ...] = (
            "temporary",
            "work_cache",
            "package_cache",
            "conda_packages",
            "runtime_packages",
            "bundles",
            "stage_cache",
            "native_packages",
            "runtime_managers",
            "runtimes",
            "environments",
        )
        if descriptor.get("environments_only"):
            categories = ("environments",)
        all_cache: list[dict[str, Any]] = []
        now = time.time()
        maximum_age = descriptor.get("cache_max_age")
        for cost, category in enumerate(categories):
            root = roots[category]
            if root.is_symlink() or not root.is_dir():
                continue
            keep = runtime_refs if category == "runtimes" else refs.get(category, set())
            for child in sorted(root.iterdir()):
                if (
                    child.is_symlink()
                    or child.name.startswith(".")
                    and category not in {"environments", "runtimes", "temporary"}
                ):
                    continue
                usage = cls._usage(child)
                item = {
                    "category": category,
                    "name": child.name,
                    **usage,
                    "last_used_at": child.stat().st_mtime,
                    "rebuild_cost": cost,
                }
                reason: str | None = None
                if child.name in keep or "*" in keep:
                    reason = "active-or-recovery-reference"
                if category == "environments":
                    if child.name in builds or any(
                        child.name.startswith(f".{value}.tmp-") for value in builds
                    ):
                        matched = next(
                            value
                            for value in builds
                            if child.name == value or child.name.startswith(f".{value}.tmp-")
                        )
                        reason = build_reasons.get(matched, "unresolved-build-owner")
                    elif not cls._complete(category, child):
                        publication = cls._read_json(child / ".lambdaforge-publication.json") or {}
                        if (
                            publication.get("contract") == "stable-prefix-v1"
                            and publication.get("state") == "failed"
                            and publication.get("prefix") == str(child)
                            and isinstance(publication.get("token"), str)
                            and len(publication["token"]) == 32
                            and all(v in "0123456789abcdef" for v in publication["token"])
                        ):
                            item["reason"] = "failed-publication"
                        elif not any(
                            child.name.startswith(f".{value}.tmp-") for value in orphan_names
                        ):
                            reason = "unresolved-temporary-owner"
                        else:
                            item["reason"] = "orphaned-temporary"
                if category == "runtimes" and not cls._complete(category, child):
                    if any(child.name.startswith(f".{value}.tmp-") for value in orphan_names):
                        item["reason"] = "orphaned-temporary"
                    else:
                        reason = "unresolved-temporary-owner"
                if builds and category in {
                    "package_cache",
                    "conda_packages",
                    "runtime_packages",
                    "native_packages",
                    "runtime_managers",
                    "runtimes",
                }:
                    reason = "environment-build-package-reference"
                if category == "temporary" and not cls._orphan_lease(child):
                    reason = "unresolved-temporary-owner"
                if category == "work_cache" and cls._work_cache_busy(child):
                    reason = "active-work-cache"
                if category == "work_cache" and child.name in cache_chains:
                    reason = "retained-recovery-dependency"
                    item["reference_chains"] = cache_chains[child.name]
                if category in {"environments", "bundles"} and cls._tool_runtime_busy(child):
                    reason = "isolated-tool-runtime-in-use"
                if reason:
                    protected_items.append({**item, "reason": reason})
                else:
                    all_cache.append(item)
        selected: set[str] = set()
        for item in all_cache:
            stale = maximum_age is not None and now - item["last_used_at"] >= float(maximum_age)
            if (
                item.get("reason") == "orphaned-temporary"
                or stale
                or item["category"] == "work_cache"
                and not descriptor.get("automatic_gc")
                or item["category"] == "environments"
                and not descriptor.get("automatic_gc")
            ):
                candidates.append(
                    {**item, "reason": item.get("reason", "stale" if stale else "reconstructible")}
                )
                selected.add(item["path"])
        total_cache = sum(int(item["bytes"]) for item in (*all_cache, *protected_items))
        removed_cache = sum(int(item["bytes"]) for item in candidates if item["path"] in selected)
        maximum_size = descriptor.get("cache_max_size")
        if maximum_size is None and descriptor.get("automatic_gc"):
            maximum_size = int(
                StorageAdmission.filesystem(cache, descriptor)["capacity_bytes"] * 0.1
            )
        if maximum_size is not None:
            # Reconstructibility/rebuild cost first, then least recently used. No ML policy.
            for item in sorted(
                all_cache, key=lambda value: (value["rebuild_cost"], value["last_used_at"])
            ):
                if total_cache - removed_cache <= int(maximum_size):
                    break
                if item["path"] in selected:
                    continue
                candidates.append({**item, "reason": "cache-quota"})
                selected.add(item["path"])
                removed_cache += int(item["bytes"])
        target_reclaim = int(descriptor.get("target_reclaim_bytes", 0))
        if target_reclaim > removed_cache:
            for item in sorted(
                all_cache, key=lambda value: (value["rebuild_cost"], value["last_used_at"])
            ):
                if removed_cache >= target_reclaim:
                    break
                if item["path"] in selected:
                    continue
                candidates.append({**item, "reason": "storage-pressure"})
                selected.add(item["path"])
                removed_cache += int(item["bytes"])
        physical_estimate = cls._reclaimable_allocation([Path(item["path"]) for item in candidates])
        environment_usage = cls._usage(roots["environments"])
        reclaimed = cache_reclaimed = 0
        collected: list[dict[str, Any]] = []
        if apply:
            for item in sorted(
                candidates, key=lambda item: item.get("name", "").startswith("deps-")
            ):
                if item["category"] in {"attempt_artifacts", "checkpoints"}:
                    if not Path(item["path"]).exists():
                        reclaimed += int(item["bytes"])
                        collected.append(item)
                    else:
                        protected_items.append({**item, "reason": "retention-reference-changed"})
                    continue
                path = Path(item["path"])
                # Recheck the lifetime lease under an exclusive lock at mutation time.
                lease = (
                    cls._work_cache_lock(path)
                    if item["category"] == "work_cache"
                    else (
                        cls._tool_runtime_lock(path, create=True)
                        if item["category"] in {"environments", "bundles"}
                        and path.name.startswith(("env-tool-", "tool-code-"))
                        else None
                    )
                )
                try:
                    if lease is not None:
                        lease.acquire()
                    if item["category"] == "work_cache":
                        fresh = cache_recovery_references(roots["job_workspaces"])
                        if path.name in fresh:
                            protected_items.append(
                                {
                                    **item,
                                    "reason": "retained-recovery-dependency",
                                    "reference_chains": fresh[path.name],
                                }
                            )
                            continue
                    cls._delete_cache_entry(path, roots[item["category"]])
                    reclaimed += int(item["bytes"])
                    cache_reclaimed += int(item["bytes"])
                    collected.append(item)
                except TimeoutError:
                    protected_items.append({**item, "reason": "active-work-cache"})
                finally:
                    if lease is not None:
                        lease.release()
            for marker in markers:
                if cls._orphan_lease(marker):
                    cls._remove_owned(marker, cache)
        grouped: dict[str, dict[str, int]] = {}
        for item in candidates:
            row = grouped.setdefault(str(item["category"]), {"items": 0, "reclaimable_bytes": 0})
            row["items"] += 1
            row["reclaimable_bytes"] += int(item["bytes"])
        payload = {
            "candidates": candidates,
            "protected_items": protected_items,
            "protected_bytes": sum(int(item["bytes"]) for item in protected_items),
            "categories": grouped,
            "reclaimable_bytes": sum(int(item["bytes"]) for item in candidates),
            "reclaimable_allocated_bytes": physical_estimate,
            "environment_usage": environment_usage,
            "reclaimed_bytes": reclaimed,
            "cache_bytes": total_cache,
            "quota_basis": "apparent-file-bytes",
            "quota_unresolved_bytes": max(
                0,
                total_cache - (cache_reclaimed if apply else removed_cache) - int(maximum_size),
            )
            if maximum_size is not None
            else 0,
            "applied": apply,
            "blocked_reason": None,
            "reference_issues": inventory["issues"],
            "environment_count": sum(
                1 for item in (*all_cache, *protected_items) if item["category"] == "environments"
            ),
            "filesystems_before": pressure_before,
            "filesystems_after": StorageAdmission.observe(descriptor) if apply else pressure_before,
            "hypothetical_after_repairs": bool(preview),
        }
        if payload["quota_unresolved_bytes"] and cache_chains:
            payload["blocked_reason"] = "retained-recovery-storage-pressure"
            payload["next_action"] = (
                "Add storage or review/delete exact owned recovery history explicitly; "
                "automatic GC will not destroy retained acquisition dependencies."
            )
        if apply:
            state_root = roots["state"]
            state_root.mkdir(parents=True, exist_ok=True)
            with (state_root / "storage-gc.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": now,
                            "reason": "safe-collection",
                            "reclaimed_bytes": reclaimed,
                            "items": len(collected),
                            "planned_items": len(candidates),
                            "collected": [
                                {
                                    key: item.get(key)
                                    for key in ("category", "path", "bytes", "reason")
                                }
                                for item in collected
                            ],
                            "categories": grouped,
                            "filesystems_before": payload["filesystems_before"],
                            "filesystems_after": payload["filesystems_after"],
                        }
                    )
                    + "\n"
                )
        if not candidates:
            payload["blocked_reason"] = (
                "No eligible cache entries. Resolve the listed ownership/reference issues first."
                if inventory["issues"] or protected_items
                else "No obsolete cache entries found."
            )
        payload["outcome"] = ("removed" if collected else "no-change") if apply else "preview"
        payload["observed_free_bytes_delta"] = (
            {
                key: int(value.get("free_bytes", 0))
                - int(pressure_before.get(key, {}).get("free_bytes", 0))
                for key, value in payload["filesystems_after"].items()
            }
            if apply
            else {}
        )
        return payload

    @staticmethod
    def _reclaimable_allocation(paths: Sequence[Path]) -> int:
        """Count blocks once and only when every hard link will be removed by this plan."""
        inodes: dict[tuple[int, int], dict[str, Any]] = {}
        unique = 0
        roots = sorted(set(paths), key=lambda path: len(path.parts))
        selected: list[Path] = []
        for root in roots:
            if not any(root.is_relative_to(parent) for parent in selected):
                selected.append(root)
        for root in selected:
            for stat in StorageOperations._file_stats(root):
                if stat.st_nlink == 1:
                    unique += getattr(stat, "st_blocks", 0) * 512
                    continue
                key = (stat.st_dev, stat.st_ino)
                record = inodes.setdefault(
                    key,
                    {
                        "links": stat.st_nlink,
                        "selected": 0,
                        "blocks": getattr(stat, "st_blocks", 0) * 512,
                    },
                )
                record["selected"] += 1
        return unique + sum(
            value["blocks"] for value in inodes.values() if value["selected"] >= value["links"]
        )

    @staticmethod
    def _file_stats(path: Path) -> Iterator[os.stat_result]:
        """One stat per ordinary file; no symlink traversal or per-file path-object churn."""
        import stat

        if path.is_symlink() or not path.exists():
            return
        if path.is_file():
            yield path.stat()
            return
        pending = [str(path)]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(entry.path)
                            else:
                                metadata = entry.stat(follow_symlinks=False)
                                if stat.S_ISREG(metadata.st_mode):
                                    yield metadata
                        except FileNotFoundError:
                            continue
            except FileNotFoundError:
                continue

    @staticmethod
    def _work_cache_lock(path: Path, *, create: bool = True) -> CrossProcessFileLock:
        # Lease outside the removable tree survives collector/worker races.
        return CrossProcessFileLock(
            path.parent / f".{path.name}.lease",
            shared=False,
            timeout_seconds=0.01,
            poll_interval_seconds=0.005,
            create=create,
        )

    @classmethod
    def _work_cache_busy(cls, path: Path) -> bool:
        lease = cls._work_cache_lock(path, create=False)
        try:
            lease.acquire()
        except FileNotFoundError:
            return False  # No lease at this snapshot; apply always revalidates under a new lease.
        except OSError:
            # Contention and uninspectable ownership both protect the cache, never prove death.
            return True
        finally:
            lease.release()
        return False

    @staticmethod
    def _tool_runtime_lock(path: Path, *, create: bool = False) -> CrossProcessFileLock:
        return CrossProcessFileLock(
            path.parent.parent / f".tool-runtime-use-{path.name}.lock",
            shared=False,
            timeout_seconds=0.01,
            poll_interval_seconds=0.005,
            create=create,
        )

    @classmethod
    def _tool_runtime_busy(cls, path: Path) -> bool:
        if not path.name.startswith(("env-tool-", "tool-code-")):
            return False
        lock = cls._tool_runtime_lock(path)
        try:
            lock.acquire()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        finally:
            lock.release()
        return False

    @classmethod
    def _orphan_lease(cls, path: Path) -> bool:
        """Only positively dead local owners qualify; age/name alone never proves death."""
        import math
        import os
        import socket

        import psutil

        record = cls._read_json(path / "owner.json" if path.is_dir() else path)
        if not record or record.get("host") != socket.gethostname():
            return False
        try:
            pid = record["pid"]
            heartbeat = float(record["heartbeat"])
            if (
                type(pid) is not int
                or pid <= 0
                or not math.isfinite(heartbeat)
                or time.time() - heartbeat < 60
            ):
                return False
            process = psutil.Process(pid)
            birth = record.get("process_birth", record.get("created_at"))
            if birth is not None and process.create_time() > float(birth) + 1:
                return not cls._orphan_builder_process_present(path)
            if process.status() in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}:
                return not cls._orphan_builder_process_present(path)
            os.kill(pid, 0)
        except psutil.NoSuchProcess:
            return not cls._orphan_builder_process_present(path)
        except ProcessLookupError:
            return not cls._orphan_builder_process_present(path)
        except (KeyError, TypeError, ValueError, PermissionError, psutil.AccessDenied):
            return False
        return False

    @classmethod
    def reference_report(cls, descriptor: Mapping[str, Any]) -> dict[str, Any]:
        """Bounded metadata for explicit repair; never fetch scientific payloads."""
        roots = cls._roots(descriptor)
        inventory = cls.environment_references(roots)
        locks = []
        for path in sorted(roots["environments"].parent.glob(".environment-build-*.lock")):
            record = cls._read_json(path / "owner.json" if path.is_dir() else path)
            locks.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "owner": record,
                    "orphan_on_build_host": cls._orphan_lease(path),
                }
            )
        return {"issues": inventory["issues"], "builds": locks}

    @staticmethod
    def _orphan_builder_process_present(path: Path) -> bool:
        """A dead controller can leave pip/Conda children alive; protect their exact prefix.

        This process scan runs only for a stale, positively dead same-host owner, never
        in ordinary resource probes. Unreadable process evidence conservatively protects.
        """
        import psutil

        name = path.name
        needles: tuple[str, ...]
        if name.startswith(".environment-build-"):
            identity = name.removeprefix(".environment-build-").removesuffix(".lock")
            needles = (
                str(path.parent / "environments" / f".{identity}.tmp-"),
                str(path.parent / "environments" / identity) + os.sep,
            )
        elif name.startswith(".python-runtime-"):
            # Publication can change the runtime from request ID to resolved ID.
            # Without the controller we cannot narrow an in-flight verification safely.
            needles = (str(path.parent / "runtimes") + os.sep,)
        else:
            needles = (str(path.absolute()),)
        owner = StorageOperations._read_json(path / "owner.json" if path.is_dir() else path) or {}
        needles = (
            *needles,
            *(str(value) for value in owner.get("references", {}).get("paths", ())),
        )
        try:
            for process in psutil.process_iter():
                try:
                    if any(
                        needle in argument for argument in process.cmdline() for needle in needles
                    ):
                        return True
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, OSError):
                    return True
        except (psutil.Error, OSError):
            return True
        return False

    @staticmethod
    def _remove_owned(path: Path, root: Path) -> None:
        """Check lexical parents before resolving; refuse symlink ancestors."""
        if path == root or not path.absolute().is_relative_to(root.absolute()):
            raise ValueError(f"GC target escaped its exact category root: {path}")
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ValueError(f"GC refuses symbolic target/ancestor: {part}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    @classmethod
    def _delete_cache_entry(cls, path: Path, root: Path) -> None:
        """Journal before rename; a killed collector cannot leave a half-valid cache entry."""
        import os
        from uuid import uuid4

        from lambdaforge.work.models import atomic_json

        if path.is_symlink() or not path.absolute().is_relative_to(root.absolute()):
            raise ValueError("Unsafe cache deletion target.")
        token = uuid4().hex
        trash = root / f".gc-trash-{token}"
        receipt = root / f".gc-delete-{token}.json"
        atomic_json(receipt, {"original": str(path), "trash": str(trash)})
        if path.exists():
            os.replace(path, trash)
        cls._remove_owned(trash, root)
        receipt.unlink(missing_ok=True)

    @classmethod
    def _resume_deletions(cls, roots: Mapping[str, Path]) -> None:

        for category, root in roots.items():
            if category in {"state", "job_workspaces", "datasets"}:
                continue
            for receipt in root.glob(".gc-delete-*.json"):
                record = cls._read_json(receipt)
                if record is None:
                    continue
                original, trash = Path(record["original"]), Path(record["trash"])
                if (
                    original.parent != root
                    or trash.parent != root
                    or not trash.name.startswith(".gc-trash-")
                ):
                    raise ValueError("Corrupt cache deletion journal; refusing unsafe recovery.")
                if original.is_symlink() or trash.is_symlink():
                    raise ValueError("Symbolic cache deletion journal target.")
                if original.exists() and not trash.exists():
                    # Crash before rename: the original may now be referenced or rebuilt.
                    # Discard the intent and let a fresh reference-aware plan select it again.
                    receipt.unlink(missing_ok=True)
                    continue
                cls._remove_owned(trash, root)
                receipt.unlink(missing_ok=True)

    @staticmethod
    def _roots(value: Mapping[str, Any]) -> dict[str, Path]:
        for key in ("state_root", "cache_root", "run_root", "dataset_root"):
            if not value.get(key):
                continue
            authored = Path(str(value[key])).expanduser().absolute()
            if authored == Path(authored.anchor):
                raise ValueError(f"Storage {key} cannot be a filesystem root.")
            if any(part.is_symlink() for part in (authored, *authored.parents)):
                raise ValueError(f"Storage {key} cannot traverse symbolic links.")
        state = Path(str(value["state_root"])).expanduser().resolve()
        cache = Path(str(value["cache_root"])).expanduser().resolve()
        run = Path(str(value["run_root"])).expanduser().resolve()
        dataset = value.get("dataset_root")
        return {
            "state": state,
            "bundles": cache / "bundles",
            "environments": cache / "environments",
            "runtimes": cache / "runtimes",
            "runtime_managers": cache / "runtime-managers",
            "conda_packages": cache / "conda-pkgs",
            "runtime_packages": cache / "runtime-packages",
            "native_packages": cache / "native-packages",
            "package_cache": cache / "pip",
            "stage_cache": cache / "dataset-stages",
            "work_cache": cache / "work",
            "job_workspaces": run,
            "temporary": cache / "tmp",
            "datasets": Path(str(dataset)).expanduser().resolve()
            if dataset
            else state / "no-dataset-root",
        }

    @staticmethod
    def _usage(path: Path) -> dict[str, Any]:
        if not path.exists() or path.is_symlink():
            return {
                "path": str(path),
                "bytes": 0,
                "allocated_bytes": 0,
                "files": 0,
                "exists": False,
            }
        apparent = allocated = count = 0
        seen: set[tuple[int, int]] = set()
        for metadata in StorageOperations._file_stats(path):
            count += 1
            apparent += metadata.st_size
            inode = (metadata.st_dev, metadata.st_ino)
            if metadata.st_nlink == 1 or inode not in seen:
                allocated += getattr(metadata, "st_blocks", 0) * 512
            if metadata.st_nlink > 1:
                seen.add(inode)
        return {
            "path": str(path),
            "bytes": apparent,
            "allocated_bytes": allocated,
            "files": count,
            "exists": True,
        }

    @staticmethod
    def _complete(category: str, path: Path) -> bool:
        if category == "stage_cache":
            return any(candidate.is_file() for candidate in path.rglob("result.json"))
        if category == "runtimes":
            return (path / ".lambdaforge-python-runtime.json").is_file()
        if category == "native_packages":
            return (path / ".lambdaforge-native-cache.json").is_file()
        marker = "manifest.json" if category == "bundles" else ".lambdaforge-environment.json"
        return (path / marker).is_file()

    @staticmethod
    def _runtime_references(roots: Mapping[str, Path], explicit: Sequence[str]) -> set[str]:
        """Protect runtimes selected globally or embedded in retained environment receipts."""
        protected = {str(item) for item in explicit}
        active = roots["state"] / "active-python-runtime.json"
        candidates = [active]
        environment_root = roots["environments"]
        if environment_root.is_dir() and not environment_root.is_symlink():
            candidates.extend(environment_root.glob("*/.lambdaforge-environment.json"))
        for candidate in candidates:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            runtime = payload.get("runtime_id")
            if runtime is None:
                policy = payload.get("environment_policy", {})
                python = policy.get("python_runtime", {}) if isinstance(policy, dict) else {}
                runtime = python.get("runtime_id") if isinstance(python, dict) else None
            if runtime:
                protected.add(str(runtime))
        return protected

    @classmethod
    def _active_environment_references(cls, roots: Mapping[str, Path]) -> set[str]:
        """Protect the active pointer and direct jobs visible in durable remote state."""
        return set(cls.environment_references(roots)["environments"])

    @staticmethod
    def _reference_snapshot(job: Path) -> str:
        import hashlib

        stat = job.stat()
        digest = hashlib.sha256(f"{stat.st_dev}:{stat.st_ino}".encode())
        for name in ("state.json", "request.json"):
            path = job / name
            if path.is_symlink():
                raise ValueError(f"Symbolic ownership evidence: {path}")
            digest.update(name.encode())
            if path.exists():
                if path.stat().st_size > 4 * 1024**2:
                    raise ValueError(f"Oversized ownership evidence: {path}")
                digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _process_references(roots: Mapping[str, Path]) -> dict[str, Any]:
        """Observe same-user process lifetimes and exact paths, without exporting argv/secrets."""
        import psutil

        environments: set[str] = set()
        jobs: set[str] = set()
        inspectable = True
        env_root, job_root = roots["environments"], roots["job_workspaces"]
        try:
            for process in psutil.process_iter():
                try:
                    if process.pid == os.getpid() or process.uids().real != os.getuid():
                        continue
                    values = list(process.cmdline())
                    try:
                        values.append(process.cwd())
                    except psutil.AccessDenied:
                        # Same-user protected desktop/credential helpers need not expose
                        # cwd/environ. Their readable argv still bounds owned prefixes;
                        # descendants are independently enumerated below.
                        pass
                    try:
                        environment = process.environ()
                    except psutil.AccessDenied:
                        environment = {}
                    job = environment.get("LAMBDAFORGE_JOB_ID")
                    if job:
                        jobs.add(job)
                    values.extend(
                        environment.get(key, "") for key in ("VIRTUAL_ENV", "CONDA_PREFIX")
                    )
                    for value in values:
                        if not value.startswith("/"):
                            continue
                        path = Path(os.path.abspath(value))
                        for root, found in ((env_root, environments), (job_root, jobs)):
                            try:
                                relative = path.relative_to(root)
                            except ValueError:
                                continue
                            if relative.parts:
                                found.add(relative.parts[0])
                except psutil.NoSuchProcess:
                    continue
                except (psutil.AccessDenied, OSError):
                    inspectable = False
        except (psutil.Error, OSError):
            inspectable = False
        return {"environments": environments, "jobs": jobs, "inspectable": inspectable}

    @classmethod
    def environment_references(
        cls, roots: Mapping[str, Path], *, repair_preview: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Scope uncertainty to recorded references; expose unresolvable Jobs explicitly."""
        environment_root = roots["environments"].resolve()
        protected: set[str] = set()
        issues: list[dict[str, Any]] = []
        repairs = cls._read_json(roots["state"] / "storage-reference-repairs.json") or {}
        repairs = {**repairs, "retired_jobs": dict(repairs.get("retired_jobs", {}))}
        for item in (repair_preview or {}).get("retired_jobs", ()):
            repairs["retired_jobs"][item["job_id"]] = item
        retired: set[str] = set()
        recovery_jobs: set[str] = set()
        # Waiting FIFO tickets need exact predecessor ownership records even when their
        # terminal Jobs are otherwise eligible for automatic workspace cleanup.
        from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue

        queue_jobs: set[str] = set()
        for queue in ExecutionQueue.for_storage(roots["state"], roots["job_workspaces"]):
            queue_jobs.update(queue.retained_jobs())
        tool_bundles: set[str] = set()

        def add_tools(path: Path, *, required: bool = False) -> None:
            if not required and not path.exists() and not path.is_symlink():
                return
            document = cls._read_json(path)
            bindings = document.get("runtimes") if document else None
            valid = (
                document is not None
                and document.get("tool_registry_version") == 1
                and isinstance(bindings, dict)
            )
            values = list(bindings.values()) if isinstance(bindings, dict) else []
            for binding in values:
                if not isinstance(binding, dict) or any(
                    not isinstance(binding.get(key), str)
                    or not binding[key].startswith(prefix)
                    or Path(binding[key]).name != binding[key]
                    for key, prefix in (
                        ("environment_id", "env-tool-"),
                        ("entrypoint_bundle_id", "tool-code-"),
                    )
                ):
                    valid = False
            if not valid:
                protected.update(p.name for p in roots["environments"].glob("env-tool-*"))
                tool_bundles.update(p.name for p in roots["bundles"].glob("tool-code-*"))
                issues.append(
                    {
                        "reason": "missing-or-corrupt-tool-bindings",
                        "path": str(path),
                        "action": "Restore audited tool bindings before collecting isolated tools.",
                    }
                )
                return
            for binding in values:
                protected.add(binding["environment_id"])
                tool_bundles.add(binding["entrypoint_bundle_id"])

        add_tools(roots["state"] / "tool-runtimes.json")
        pointer = roots["state"] / "active-environment"
        # A concurrent activation can change the pointer after this collector starts.
        # Keep its invoked interpreter alive until evidence reads finish.
        candidates: list[str] = [sys.executable]
        if pointer.is_file() and not pointer.is_symlink():
            try:
                candidates.append(pointer.read_text(encoding="utf-8").strip())
            except OSError:
                pass
        job_root = roots["job_workspaces"]
        terminal = {"succeeded", "failed", "cancelled", "timeout", "planned"}
        if job_root.is_dir() and not job_root.is_symlink():
            for child in job_root.iterdir():
                if child.is_symlink() or not child.is_dir() or not child.name.startswith("job-"):
                    continue
                state = cls._read_json(child / "state.json")
                request = cls._read_json(child / "request.json")
                if state and str(state.get("state")) in terminal:
                    continue
                snapshot = cls._reference_snapshot(child)
                if repairs.get("retired_jobs", {}).get(child.name, {}).get("snapshot") == snapshot:
                    retired.add(child.name)
                    continue
                if not request:
                    protected.add("*")
                    issues.append(
                        {
                            "job_id": child.name,
                            "path": str(child),
                            "reason": "missing-or-corrupt-request",
                            "action": f"lf storage repair --retire-job {child.name}",
                        }
                    )
                    continue
                command = request.get("command", ())
                if isinstance(command, list):
                    candidates.extend(str(value) for value in command)
                    add_tools(
                        child / "tool-runtimes.json",
                        required=any(
                            str(a).startswith("LAMBDAFORGE_TOOL_RUNTIMES_PATH=") for a in command
                        ),
                    )
                    # Recovery argv identifies original owned Executions even when the
                    # controller catalog no longer contains their terminal owner Jobs.
                    for argument in command:
                        try:
                            relative = Path(str(argument)).relative_to(job_root)
                        except ValueError:
                            continue
                        if relative.parts and relative.parts[0] != child.name:
                            recovery_jobs.add(relative.parts[0])
                if not state:
                    issues.append(
                        {
                            "job_id": child.name,
                            "path": str(child),
                            "reason": "missing-or-corrupt-state-scoped-to-request",
                            "action": f"lf storage repair --retire-job {child.name}",
                        }
                    )
        for job_id in recovery_jobs | queue_jobs:
            request = cls._read_json(job_root / job_id / "request.json")
            if request is not None and isinstance(request.get("command"), list):
                candidates.extend(str(value) for value in request["command"])
                add_tools(
                    job_root / job_id / "tool-runtimes.json",
                    required=any(
                        str(a).startswith("LAMBDAFORGE_TOOL_RUNTIMES_PATH=")
                        for a in request["command"]
                    ),
                )
            elif job_id in recovery_jobs:
                protected.add("*")
                issues.append(
                    {
                        "job_id": job_id,
                        "reason": "recovery-owner-request-missing",
                        "action": "Restore original recovery ownership evidence before cleanup.",
                    }
                )
        for candidate in candidates:
            try:
                # venv interpreters link outside the invoked environment prefix.
                path = Path(os.path.abspath(Path(candidate).expanduser()))
                relative = path.relative_to(environment_root)
            except (OSError, ValueError):
                continue
            if relative.parts:
                protected.add(relative.parts[0])
        processes = cls._process_references(roots)
        protected.update(processes["environments"])
        for job_id in retired & processes["jobs"]:
            protected.add("*")
            issues.append(
                {
                    "job_id": job_id,
                    "reason": "retired-reference-has-live-process",
                    "action": "Inspect the live owner; retirement is not permission to evict it.",
                }
            )
        if not processes["inspectable"]:
            protected.add("*")
            issues.append(
                {
                    "reason": "process-inspection-incomplete",
                    "action": "Restore same-user process inspection; no death was inferred.",
                }
            )
        return {
            "environments": protected,
            "issues": issues,
            "retired_jobs": retired,
            "protected_jobs": recovery_jobs | queue_jobs,
            "tool_bundles": tool_bundles,
        }

    @classmethod
    def repair_references(
        cls,
        descriptor: Mapping[str, Any],
        selections: Mapping[str, Any],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Audit explicit retirement of abandoned operational references, not scientific Jobs."""
        from contextlib import nullcontext

        from lambdaforge.work.models import atomic_json

        roots = cls._roots(descriptor)
        cache, state = roots["environments"].parent, roots["state"]
        if apply:
            cache.mkdir(parents=True, exist_ok=True)
            state.mkdir(parents=True, exist_ok=True)
        lock = (
            CrossProcessFileLock(
                cache / ".gc.lock", shared=False, timeout_seconds=1, poll_interval_seconds=0.05
            )
            if apply
            else nullcontext()
        )
        with lock:
            path = state / "storage-reference-repairs.json"
            audit = cls._read_json(path) or {"version": 1, "retired_jobs": {}}
            if path.is_symlink() or path.exists() and cls._read_json(path) is None:
                raise ValueError("Corrupt reference-repair audit; no changes made.")
            processes = cls._process_references(roots)
            if not processes["inspectable"]:
                raise ValueError(
                    "Cannot repair references: same-user process inspection is incomplete."
                )
            retired: list[dict[str, Any]] = []
            for job_id in selections.get("retire_jobs", ()):
                if (
                    not isinstance(job_id, str)
                    or not job_id.startswith("job-")
                    or any(value not in "abcdefghijklmnopqrstuvwxyz0123456789-" for value in job_id)
                ):
                    raise ValueError("Invalid exact Job identity for reference repair.")
                job = roots["job_workspaces"] / job_id
                if job.is_symlink() or not job.is_dir():
                    raise ValueError(f"Unsafe/missing Job reference: {job}")
                request = cls._read_json(job / "request.json")
                observed = cls._read_json(job / "state.json")
                if (
                    job_id in processes["jobs"]
                    or observed
                    and observed.get("state")
                    not in {
                        "failed",
                        "succeeded",
                        "cancelled",
                        "timeout",
                        "planned",
                        None,
                    }
                ):
                    raise ValueError(
                        f"Job {job_id} remains active or unresolved; reference retained."
                    )
                if request and request.get("scheduler") not in (None, "direct", "process", "local"):
                    raise ValueError(f"Job {job_id} requires scheduler reconciliation first.")
                if not request and descriptor.get("reconciliation_scheduler") == "slurm":
                    raise ValueError(
                        f"Job {job_id} requires scheduler ownership evidence before repair."
                    )
                snapshot = cls._reference_snapshot(job)
                record = {
                    "job_id": job_id,
                    "path": str(job),
                    "snapshot": snapshot,
                    "reason": "explicit-abandoned-reference-retirement",
                    "checked_at": time.time(),
                }
                audit["retired_jobs"][job_id] = record
                retired.append(record)
            released: list[dict[str, Any]] = []
            for selection in selections.get("release_builds", ()):
                name = str(selection.get("name", ""))
                if (
                    "/" in name
                    or not name.startswith(".environment-build-")
                    or not name.endswith(".lock")
                ):
                    raise ValueError("Invalid exact build marker.")
                marker = cache / name
                owner = cls._read_json(marker / "owner.json" if marker.is_dir() else marker)
                if owner is None or owner != selection.get("owner"):
                    raise ValueError(f"Build {name} has unreadable/changed ownership; retained.")
                proof = selection.get("controller_proof", {})
                dead = cls._orphan_lease(marker) or (
                    proof.get("status") == "dead-or-reused"
                    and proof.get("host") == owner.get("host")
                    and proof.get("pid") == owner.get("pid")
                )
                if not dead or cls._orphan_builder_process_present(marker):
                    raise ValueError(
                        f"Build {name} still has live or unverified owners/descendants."
                    )
                if marker.is_dir() and {value.name for value in marker.iterdir()} != {"owner.json"}:
                    raise ValueError(f"Build {name} has unexpected content; retained.")
                identity = name.removeprefix(".environment-build-").removesuffix(".lock")
                released.append({"name": name, "path": str(marker), "owner": owner})
                audit.setdefault("retired_builds", {})[identity] = {
                    "owner": owner,
                    "retired_at": time.time(),
                    "controller_proof": proof,
                }
            if apply:
                for item in retired:
                    if cls._reference_snapshot(Path(item["path"])) != item["snapshot"]:
                        raise ValueError("Job ownership changed during reference repair.")
                # Audit intent first. A crash or changed marker before removal remains
                # protected: GC only consumes retired-build entries with NO current marker.
                atomic_json(path, audit)
                for item in released:
                    marker, owner = Path(item["path"]), item["owner"]
                    if (
                        cls._read_json(marker / "owner.json" if marker.is_dir() else marker)
                        != owner
                    ):
                        raise ValueError(f"Build {item['name']} changed during repair; retained.")
                    cls._remove_owned(marker, cache)
            return {
                "applied": apply,
                "retired_jobs": retired,
                "released_builds": released,
                "audit_path": str(path),
                "preserved": [
                    "all Job files",
                    "checkpoints",
                    "datasets",
                    "results",
                    "failure history",
                ],
                "note": (
                    "Changed ownership evidence invalidates retirement; GC rechecks live processes."
                ),
            }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file() or path.is_symlink():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        values = tuple(argv if argv is not None else sys.argv[1:])
        if len(values) not in {2, 4}:
            raise SystemExit(
                "Usage: StorageOperations status DESCRIPTOR | "
                "gc DESCRIPTOR REFS APPLY | prune-environments DESCRIPTOR REFS APPLY | "
                "delete-job DESCRIPTOR REFS APPLY | compact-job DESCRIPTOR REFS APPLY | "
                "reconcile DESCRIPTOR REFS APPLY"
            )
        descriptor = json.loads(values[1])
        if values[0] == "status":
            payload = cls.status(descriptor)
        elif values[0] == "reconcile" and len(values) == 4:
            payload = cls.reconcile(descriptor, apply=values[3] == "true")
        elif values[0] == "repair" and len(values) == 4:
            payload = cls.repair_references(
                descriptor, json.loads(values[2]), apply=values[3] == "true"
            )
        elif values[0] == "reference-report" and len(values) == 4:
            payload = cls.reference_report(descriptor)
        elif values[0] == "gc" and len(values) == 4:
            payload = cls.gc(descriptor, json.loads(values[2]), apply=values[3] == "true")
        elif values[0] == "prune-environments" and len(values) == 4:
            references = json.loads(values[2])
            payload = cls.prune_environments(
                descriptor,
                references.get("environments", ()) if isinstance(references, dict) else (),
                apply=values[3] == "true",
            )
        elif values[0] == "delete-job" and len(values) == 4:
            references = json.loads(values[2])
            payload = cls.delete_job(
                descriptor,
                str(references["job_id"]),
                apply=values[3] == "true",
            )
        elif values[0] == "compact-job" and len(values) == 4:
            references = json.loads(values[2])
            payload = cls.compact_job(
                descriptor,
                str(references["job_id"]),
                apply=values[3] == "true",
            )
        else:
            raise SystemExit("Unknown storage operation.")
        print(json.dumps(payload, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(StorageOperations.main())
