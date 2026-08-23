"""Bound runtime capabilities owned by one executing Work instance."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, TypeVar

from lambdaforge.data.DatasetPublisher import DatasetPublisher
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.work.models import (
    WorkArtifact,
    WorkConfiguration,
    WorkInput,
    WorkResources,
    WorkTrial,
    atomic_json,
)

T = TypeVar("T")
R = TypeVar("R")


def _owned_path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    root = root.resolve()
    raw = Path(value)
    unresolved = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(unresolved))
    if not lexical.is_relative_to(root):
        raise ValueError(f"Managed path escapes its owning root: {value}")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Managed paths cannot traverse symbolic links: {value}")
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Managed path escapes its owning root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Managed path does not exist: {resolved}")
    return resolved


def _fingerprint(path: Path) -> tuple[str, int]:
    if path.is_symlink() or (not path.is_file() and not path.is_dir()):
        raise ValueError(f"Artifacts must be regular files or directories: {path}")
    digest = hashlib.sha256()
    size = 0
    entries = (path,) if path.is_file() else tuple(sorted(path.rglob("*")))
    for item in entries:
        if item.is_symlink():
            raise ValueError(f"Artifact trees cannot contain symbolic links: {item}")
        if not item.is_file():
            continue
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    return digest.hexdigest(), size


class OutputCollection:
    """Register named structured values, artifacts and immutable datasets."""

    def __init__(self, runtime: WorkRuntime) -> None:
        self._runtime = runtime
        self._values: dict[str, Any] = {}
        self._artifacts: dict[str, WorkArtifact] = {}
        self._datasets: dict[str, Mapping[str, Any]] = {}

    def value(self, name: str, value: Any) -> None:
        """Register one immutable JSON-compatible named value."""
        selected = self._new_name(name)
        try:
            encoded = json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(f"Output {selected!r} must be JSON-compatible: {error}") from error
        self._values[selected] = json.loads(encoded)

    def artifact(
        self,
        name: str,
        path: str | Path,
        *,
        role: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Copy or retain one safe path under attempt-owned artifact storage."""
        selected = self._new_name(name)
        raw = Path(path)
        source = (
            raw.resolve(strict=False)
            if raw.is_absolute()
            else (self._runtime.run_dir / raw).resolve(strict=False)
        )
        if not source.exists() or source.is_symlink():
            raise FileNotFoundError(f"Registered artifact is missing or symbolic: {source}")
        if source.is_relative_to(self._runtime.run_dir.resolve()):
            managed = _owned_path(self._runtime.run_dir, source, must_exist=True)
        else:
            destination = _owned_path(
                self._runtime.run_dir,
                Path("artifacts") / self._safe_name(selected),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Managed artifact destination already exists: {destination}")
            if source.is_dir():
                shutil.copytree(source, destination)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unsupported artifact path: {source}")
            managed = destination
        digest, size = _fingerprint(managed)
        self._artifacts[selected] = WorkArtifact(
            selected,
            managed.relative_to(self._runtime.run_dir).as_posix(),
            str(role),
            digest,
            size,
            media_type,
            dict(metadata or {}),
        )
        return managed

    def dataset(
        self,
        *,
        name: str,
        version: str,
        members: Iterable[Mapping[str, Any]],
        output: str = "dataset",
        metadata: Mapping[str, Any] | None = None,
        target_schema: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Stream, verify and atomically publish one independent DatasetVersion."""
        output_name = self._new_name(output)
        cluster = os.environ.get("LAMBDAFORGE_CLUSTER", "local")
        configured_root = os.environ.get("LAMBDAFORGE_DATASET_ROOT")
        if cluster != "local" and configured_root is None:
            raise RuntimeError(
                f"Cluster {cluster!r} has no permanent storage.dataset_root for publication."
            )
        publication_root = Path(
            configured_root or self._runtime.source_dir / ".lambdaforge" / "datasets" / "published"
        )
        registry = DatasetRegistry(
            os.environ.get("LAMBDAFORGE_DATASET_REGISTRY")
            or DatasetRegistry.project_path(self._runtime.source_dir)
        )
        record = DatasetPublisher(registry).publish_members(
            name,
            version,
            members,
            source_root=self._runtime.run_dir,
            publication_root=publication_root,
            build_provenance={
                "work_class": self._runtime.config.work_class,
                "scientific_fingerprint": self._runtime.scientific_fingerprint,
                "execution_id": self._runtime.execution_id,
                "run_id": self._runtime.run_id,
                "attempt_id": self._runtime.attempt_id,
            },
            cluster=cluster,
            metadata=metadata,
            target_schema=target_schema,
        )
        payload = record.to_dict()
        self._datasets[output_name] = payload
        return payload

    @property
    def values(self) -> Mapping[str, Any]:
        return dict(self._values)

    @property
    def artifacts(self) -> tuple[WorkArtifact, ...]:
        return tuple(self._artifacts.values())

    @property
    def datasets(self) -> Mapping[str, Mapping[str, Any]]:
        return dict(self._datasets)

    def _new_name(self, name: str) -> str:
        selected = str(name).strip()
        if not selected:
            raise ValueError("Output names cannot be empty.")
        if selected in self._values or selected in self._artifacts or selected in self._datasets:
            raise ValueError(f"Output {selected!r} is already registered in this attempt.")
        return selected

    @staticmethod
    def _safe_name(value: str) -> str:
        name = "".join(
            character if character.isalnum() or character in "-_." else "-" for character in value
        )
        name = name.strip(".-")
        if not name:
            raise ValueError("Artifact names require at least one portable character.")
        return f"{name[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:10]}"


class MetricCollection:
    """Persist scalar metric history and retain final values for comparisons/HPO."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "metrics.jsonl"
        self._lock = Lock()
        self._latest: dict[str, int | float] = {}
        self._count = 0

    def log(
        self,
        name: str,
        value: int | float,
        *,
        step: int | None = None,
        split: str | None = None,
    ) -> None:
        """Append one durable numeric observation."""
        selected = str(name).strip()
        if not selected:
            raise ValueError("Metric names cannot be empty.")
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise TypeError("Metric values must be finite numeric scalars.")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("Metric step must be a non-negative integer or null.")
        key = f"{split}_{selected}" if split else selected
        record = {"name": selected, "value": value, "step": step, "split": split}
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._latest[key] = value
            self._count += 1

    def log_many(
        self,
        values: Mapping[str, int | float],
        *,
        step: int | None = None,
        split: str | None = None,
    ) -> None:
        """Log several scalar values at the same step."""
        for name, value in values.items():
            self.log(name, value, step=step, split=split)

    @property
    def latest(self) -> Mapping[str, int | float]:
        return dict(self._latest)

    @property
    def count(self) -> int:
        return self._count


class CheckpointCollection:
    """Own safe resumable state for a Run across Attempts."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, *, create_parent: bool = True) -> Path:
        """Return a contained checkpoint path without serializing user state."""
        selected = _owned_path(self._root, name)
        if create_parent:
            selected.parent.mkdir(parents=True, exist_ok=True)
        return selected

    def exists(self, name: str) -> bool:
        """Return whether a safe checkpoint path exists."""
        return self.path(name, create_parent=False).exists()

    def save_json(self, name: str, value: Any) -> Path:
        """Atomically store trusted JSON-shaped checkpoint state."""
        return atomic_json(self.path(name), value)

    def load_json(self, name: str) -> Any:
        """Load existing JSON state; corrupt state raises rather than becoming empty."""
        path = self.path(name, create_parent=False)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return json.loads(path.read_text(encoding="utf-8"))


class CacheCollection:
    """Provide contained reconstructible storage keyed by scientific identity."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, *, create: bool = True) -> Path:
        """Return a safe cache path, creating its parent by default."""
        selected = _owned_path(self._root, name)
        if create:
            selected.parent.mkdir(parents=True, exist_ok=True)
        return selected


class ProgressReporter:
    """Persist optional current progress for job observation and ``lf top``."""

    def __init__(self, run_dir: Path) -> None:
        configured = os.environ.get("LAMBDAFORGE_PROGRESS_PATH")
        self._path = Path(configured).resolve() if configured else run_dir / "progress.json"
        self._lock = Lock()

    def update(
        self,
        *,
        completed: int,
        total: int | None = None,
        message: str | None = None,
    ) -> None:
        """Replace the current bounded progress snapshot atomically."""
        if isinstance(completed, bool) or completed < 0:
            raise ValueError("Progress completed must be a non-negative integer.")
        if total is not None and (isinstance(total, bool) or total < completed):
            raise ValueError("Progress total must be an integer >= completed or null.")
        with self._lock:
            atomic_json(
                self._path,
                {"completed": completed, "total": total, "message": message},
            )


@dataclass(slots=True)
class WorkRuntime:
    """Private one-shot binding installed by WorkRunner."""

    config: WorkConfiguration
    inputs: Mapping[str, WorkInput]
    resources: WorkResources
    run_dir: Path
    temp_dir: Path
    source_dir: Path
    checkpoints: CheckpointCollection
    cache: CacheCollection
    seed: int | None
    trial: WorkTrial | None
    resuming: bool
    scientific_fingerprint: str
    execution_id: str
    run_id: str
    attempt_id: str
    outputs: OutputCollection = field(init=False)
    metrics: MetricCollection = field(init=False)
    progress: ProgressReporter = field(init=False)
    map_counter: int = 0

    def __post_init__(self) -> None:
        self.inputs = MappingProxyType(dict(self.inputs))
        self.outputs = OutputCollection(self)
        self.metrics = MetricCollection(self.run_dir)
        self.progress = ProgressReporter(self.run_dir)

    def map(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        key: Callable[[T], str],
        workers: int = 1,
        executor: str = "thread",
        resume: bool = True,
        name: str | None = None,
    ) -> list[R]:
        """Map inside one Job with stable JSON checkpoints and input-order results."""
        if isinstance(workers, bool) or workers < 1:
            raise ValueError("map workers must be an integer >= 1.")
        if executor not in {"thread", "process"}:
            raise ValueError("map executor must be 'thread' or 'process'.")
        self.map_counter += 1
        operation = name or f"map-{self.map_counter:03d}"
        root = self.checkpoints.path(f"maps/{operation}")
        root.mkdir(parents=True, exist_ok=True)
        materialized = list(items)
        keys = [str(key(item)) for item in materialized]
        if any(not selected for selected in keys):
            raise ValueError("map keys cannot be empty.")
        if len(keys) != len(set(keys)):
            duplicates = sorted({value for value in keys if keys.count(value) > 1})
            raise ValueError(f"map keys must be unique; duplicates: {duplicates}.")
        results: list[Any] = [None] * len(materialized)
        pending: list[tuple[int, T, str]] = []
        for index, (item, selected) in enumerate(zip(materialized, keys, strict=True)):
            path = root / f"{hashlib.sha256(selected.encode()).hexdigest()}.json"
            if resume and path.is_file() and not path.is_symlink():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("key") != selected:
                    raise RuntimeError(f"map checkpoint key collision for {selected!r}.")
                results[index] = record["result"]
            else:
                pending.append((index, item, selected))
        completed = len(materialized) - len(pending)
        self.progress.update(completed=completed, total=len(materialized), message=operation)
        if not pending:
            return results
        if workers == 1:
            for index, item, selected in pending:
                result = function(item)
                self._save_map_result(root, selected, result)
                results[index] = result
                completed += 1
                self.progress.update(
                    completed=completed, total=len(materialized), message=operation
                )
            return results
        pool = (
            ThreadPoolExecutor(max_workers=workers)
            if executor == "thread"
            else ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        )
        with pool:
            futures: dict[Future[R], tuple[int, str]] = {
                pool.submit(function, item): (index, selected) for index, item, selected in pending
            }
            try:
                for future in as_completed(futures):
                    index, selected = futures[future]
                    result = future.result()
                    self._save_map_result(root, selected, result)
                    results[index] = result
                    completed += 1
                    self.progress.update(
                        completed=completed, total=len(materialized), message=operation
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    @staticmethod
    def _save_map_result(root: Path, key: str, value: Any) -> None:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(f"map result for {key!r} must be JSON-compatible: {error}") from error
        atomic_json(
            root / f"{hashlib.sha256(key.encode()).hexdigest()}.json", {"key": key, "result": value}
        )
