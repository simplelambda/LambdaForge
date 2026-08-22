"""Bounded remote-side storage inspection and reference-aware cache deletion."""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock


class StorageOperations:
    """Operate only below exact configured internal roots."""

    @classmethod
    def status(cls, descriptor: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        roots = cls._roots(descriptor)
        return {name: cls._usage(path) for name, path in roots.items()}

    @classmethod
    def gc(
        cls,
        descriptor: Mapping[str, Any],
        references: Mapping[str, Sequence[str]],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Serialize cache collection against another collector on the same cluster."""
        cache_root = Path(str(descriptor["cache_root"])).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            cache_root / ".gc.lock",
            shared=False,
            timeout_seconds=30.0,
            poll_interval_seconds=0.1,
        ):
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
        cache_root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            cache_root / ".gc.lock",
            shared=False,
            timeout_seconds=30.0,
            poll_interval_seconds=0.1,
        ):
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
                        or not child.name.startswith(("env-", ".env-"))
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
                    shutil.rmtree(path)
                    pruned.append(str(item["environment_id"]))
            return {
                "candidates": candidates,
                "pruned": pruned,
                "applied": apply,
                "blocked_reason": None,
            }

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
    def _gc(
        cls,
        descriptor: Mapping[str, Any],
        references: Mapping[str, Sequence[str]],
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        roots = cls._roots(descriptor)
        cache_root = Path(str(descriptor["cache_root"])).expanduser().resolve()
        markers = (
            *cache_root.glob(".environment-build-*.lock"),
            *cache_root.glob(".python-runtime-*.lock"),
        )
        if markers:
            return {
                "candidates": [],
                "reclaimable_bytes": 0,
                "applied": False,
                "blocked_reason": "A managed environment build is currently mutating cache.",
            }
        candidates: list[dict[str, Any]] = []
        candidate_paths: set[str] = set()
        now = time.time()
        maximum_age = descriptor.get("cache_max_age")
        runtime_references = cls._runtime_references(roots, references.get("runtimes", ()))
        for category, reference_key in (
            ("bundles", "bundles"),
            ("environments", "environments"),
            ("runtimes", "runtimes"),
            ("stage_cache", "stage_cache"),
        ):
            root = roots[category]
            protected = (
                runtime_references
                if category == "runtimes"
                else set(str(item) for item in references.get(reference_key, ()))
            )
            if not root.is_dir() or root.is_symlink():
                continue
            for child in sorted(root.iterdir()):
                if (
                    child.is_symlink()
                    or not child.is_dir()
                    or child.name in protected
                    or "*" in protected
                ):
                    continue
                incomplete = ".tmp-" in child.name or not cls._complete(category, child)
                stale = maximum_age is not None and now - child.stat().st_mtime >= float(
                    maximum_age
                )
                if incomplete or stale:
                    usage = cls._usage(child)
                    item = {
                        "category": category,
                        "name": child.name,
                        "path": str(child),
                        "bytes": usage["bytes"],
                        "files": usage["files"],
                        "reason": "incomplete" if incomplete else "stale",
                    }
                    candidates.append(item)
                    candidate_paths.add(str(child.resolve()))
        temporary = roots["temporary"]
        if temporary.is_dir() and not temporary.is_symlink():
            for child in sorted(temporary.iterdir()):
                if child.is_symlink():
                    continue
                usage = cls._usage(child)
                candidates.append(
                    {
                        "category": "temporary",
                        "name": child.name,
                        "path": str(child),
                        "bytes": usage["bytes"],
                        "files": usage["files"],
                        "reason": "temporary",
                    }
                )
                candidate_paths.add(str(child.resolve()))
        maximum_size = descriptor.get("cache_max_size")
        if maximum_size is not None:
            cache_roots = (
                roots["bundles"],
                roots["environments"],
                roots["runtimes"],
                roots["runtime_managers"],
                roots["conda_packages"],
                roots["runtime_packages"],
                roots["package_cache"],
                roots["stage_cache"],
                roots["temporary"],
            )
            cache_bytes = sum(int(cls._usage(path)["bytes"]) for path in cache_roots)
            reclaimable = sum(int(item["bytes"]) for item in candidates)
            quota_candidates: list[tuple[float, str, Path]] = []
            for category, reference_key in (
                ("bundles", "bundles"),
                ("environments", "environments"),
                ("runtimes", "runtimes"),
                ("stage_cache", "stage_cache"),
            ):
                protected = (
                    runtime_references
                    if category == "runtimes"
                    else set(str(item) for item in references.get(reference_key, ()))
                )
                root = roots[category]
                if not root.is_dir() or root.is_symlink():
                    continue
                for child in root.iterdir():
                    resolved = str(child.resolve())
                    if (
                        child.is_dir()
                        and not child.is_symlink()
                        and child.name not in protected
                        and "*" not in protected
                        and resolved not in candidate_paths
                    ):
                        quota_candidates.append((child.stat().st_mtime, category, child))
            for _, category, child in sorted(quota_candidates):
                if cache_bytes - reclaimable <= int(maximum_size):
                    break
                usage = cls._usage(child)
                candidates.append(
                    {
                        "category": category,
                        "name": child.name,
                        "path": str(child),
                        "bytes": usage["bytes"],
                        "files": usage["files"],
                        "reason": "cache-quota",
                    }
                )
                reclaimable += int(usage["bytes"])
        if apply:
            allowed = tuple(path.resolve() for path in roots.values())
            for item in candidates:
                path = Path(str(item["path"])).resolve()
                if not any(path != root and path.is_relative_to(root) for root in allowed):
                    raise RuntimeError(f"GC target escaped configured internal roots: {path}")
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
        return {
            "candidates": candidates,
            "reclaimable_bytes": sum(int(item["bytes"]) for item in candidates),
            "applied": apply,
            "blocked_reason": None,
        }

    @staticmethod
    def _roots(value: Mapping[str, Any]) -> dict[str, Path]:
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
            "package_cache": cache / "pip",
            "stage_cache": cache / "dataset-stages",
            "job_workspaces": run,
            "temporary": cache / "tmp",
            "datasets": Path(str(dataset)).expanduser().resolve()
            if dataset
            else state / "no-dataset-root",
        }

    @staticmethod
    def _usage(path: Path) -> dict[str, Any]:
        if not path.exists() or path.is_symlink():
            return {"path": str(path), "bytes": 0, "files": 0, "exists": False}
        if path.is_file():
            return {"path": str(path), "bytes": path.stat().st_size, "files": 1, "exists": True}
        files = tuple(item for item in path.rglob("*") if item.is_file() and not item.is_symlink())
        return {
            "path": str(path),
            "bytes": sum(item.stat().st_size for item in files),
            "files": len(files),
            "exists": True,
        }

    @staticmethod
    def _complete(category: str, path: Path) -> bool:
        if category == "stage_cache":
            return any(candidate.is_file() for candidate in path.rglob("result.json"))
        if category == "runtimes":
            return (path / ".lambdaforge-python-runtime.json").is_file()
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
        environment_root = roots["environments"].resolve()
        protected: set[str] = set()
        pointer = roots["state"] / "active-environment"
        candidates: list[str] = []
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
                if not state or not request or str(state.get("state")) in terminal:
                    continue
                command = request.get("command", ())
                if isinstance(command, list):
                    candidates.extend(str(value) for value in command)
        for candidate in candidates:
            try:
                path = Path(candidate).expanduser().resolve()
                relative = path.relative_to(environment_root)
            except (OSError, ValueError):
                continue
            if relative.parts:
                protected.add(relative.parts[0])
        return protected

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file() or path.is_symlink():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        values = tuple(argv if argv is not None else sys.argv[1:])
        if len(values) not in {2, 4}:
            raise SystemExit(
                "Usage: StorageOperations status DESCRIPTOR | "
                "gc DESCRIPTOR REFS APPLY | prune-environments DESCRIPTOR REFS APPLY | "
                "delete-job DESCRIPTOR REFS APPLY"
            )
        descriptor = json.loads(values[1])
        if values[0] == "status":
            payload = cls.status(descriptor)
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
        else:
            raise SystemExit("Unknown storage operation.")
        print(json.dumps(payload, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(StorageOperations.main())
