"""Bound runtime capabilities owned by one executing Work instance."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, TypeVar, cast

from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility
from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.managed import ManagedFile
from lambdaforge.work.models import (
    WorkConfiguration,
    WorkFidelity,
    WorkInput,
    WorkResources,
    WorkTrial,
    atomic_json,
)
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.paths import WorkPathContext
from lambdaforge.work.tools import ToolService

T = TypeVar("T")
R = TypeVar("R")


class MetricCollection:
    """Persist scalar metric history and retain final values for comparisons/HPO."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "metrics.jsonl"
        self._lock = Lock()
        self._latest: dict[str, int | float] = {}
        self._count = 0
        mirror = os.environ.get("LAMBDAFORGE_HPO_METRICS_PATH")
        self._mirror = Path(mirror).resolve() if mirror else None
        raw_objective = os.environ.get("LAMBDAFORGE_HPO_OBJECTIVE_CONFIG")
        try:
            decoded = json.loads(raw_objective) if raw_objective else None
        except json.JSONDecodeError:
            decoded = None
        self._objective = ObjectiveUtility(decoded) if isinstance(decoded, Mapping) else None
        self._objective_steps: dict[int | None, dict[str, float]] = {}

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
            if self._mirror is not None:
                self._mirror.parent.mkdir(parents=True, exist_ok=True)
                with self._mirror.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if self._objective is not None and self._objective.composite and step is not None:
                    values = self._objective_steps.setdefault(step, {})
                    values[key] = float(value)
                    evaluated = self._objective.evaluate(values)
                    if evaluated is not None:
                        utility_record = {
                            "name": self._objective.metric,
                            "value": evaluated["value"],
                            "step": step,
                            "split": None,
                        }
                        with self._mirror.open("a", encoding="utf-8", newline="\n") as handle:
                            handle.write(
                                json.dumps(
                                    utility_record,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
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


class WorkLog:
    """Emit timestamped researcher messages into both Attempt and scheduler logs."""

    LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})

    def __init__(self) -> None:
        self._lock = Lock()

    def emit(self, message: object, *, level: str = "info") -> None:
        """Write and flush one levelled message through the currently captured stdout."""
        selected = str(level).strip().lower()
        if selected not in self.LEVELS:
            raise ValueError(f"Log level must be one of {tuple(sorted(self.LEVELS))}.")
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = str(message).splitlines() or [""]
        with self._lock:
            for line in lines:
                sys.stdout.write(f"[{timestamp}] [{selected.upper()}] {line}\n")
            sys.stdout.flush()


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
    cache: WorkCache
    seed: int | None
    trial: WorkTrial | None
    fidelity: WorkFidelity | None
    resuming: bool
    scientific_fingerprint: str
    execution_id: str
    run_id: str
    attempt_id: str
    path_context: WorkPathContext
    outputs: OutputCollection = field(init=False)
    metrics: MetricCollection = field(init=False)
    progress: ProgressReporter = field(init=False)
    log: WorkLog = field(init=False)
    tools: ToolService = field(init=False)
    map_counter: int = 0

    def __post_init__(self) -> None:
        self.inputs = MappingProxyType(dict(self.inputs))
        self.outputs = OutputCollection(self)
        self.metrics = MetricCollection(self.run_dir)
        self.progress = ProgressReporter(self.run_dir)
        self.log = WorkLog()
        self.tools = ToolService(self.log)

    @property
    def stop_requested(self) -> bool:
        """Return whether an adaptive controller requested cooperative early stopping."""
        configured = os.environ.get("LAMBDAFORGE_STOP_REQUEST_PATH")
        return bool(configured and Path(configured).is_file())

    def map(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        workers: int = 1,
        executor: str = "thread",
        name: str | None = None,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> list[R]:
        """Apply one function with ordered bounded concurrency and no hidden persistence."""
        self._validate_map_options(workers, executor, retries, retry_backoff)
        self.map_counter += 1
        operation = name or f"map-{self.map_counter:03d}"
        materialized = list(items)
        total = len(materialized)
        self.progress.update(completed=0, total=total, message=operation)
        if workers == 1:
            sequential_results: list[R] = []
            for item in materialized:
                sequential_results.append(
                    _call_with_retries(function, item, retries, float(retry_backoff))
                )
                self.progress.update(
                    completed=len(sequential_results), total=total, message=operation
                )
            return sequential_results
        pool = (
            ThreadPoolExecutor(max_workers=workers)
            if executor == "thread"
            else ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        )
        results: list[Any] = [None] * total
        with pool:
            futures = {
                pool.submit(
                    _call_with_retries,
                    function,
                    item,
                    retries,
                    float(retry_backoff),
                ): index
                for index, item in enumerate(materialized)
            }
            completed = 0
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    completed += 1
                    self.progress.update(completed=completed, total=total, message=operation)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    def resume_map(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        key: Callable[[T], str] | str,
        workers: int = 1,
        executor: str = "thread",
        resume: bool = True,
        name: str | None = None,
        validate: Callable[[R], bool] | None = None,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> list[R]:
        """Map with explicit stable keys and dependency-aware resumable checkpoints."""
        self._validate_map_options(workers, executor, retries, retry_backoff)
        self.map_counter += 1
        # Preserve the historical anonymous checkpoint path for keyed map calls.
        operation = name or f"map-{self.map_counter:03d}"
        root = self.checkpoints.path(f"maps/{operation}")
        root.mkdir(parents=True, exist_ok=True)
        materialized = list(items)
        keys = [self._map_key(item, key) for item in materialized]
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
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, TypeError):
                    pending.append((index, item, selected))
                    continue
                if record.get("key") != selected:
                    raise RuntimeError(f"map checkpoint key collision for {selected!r}.")
                try:
                    dependencies = record.get("dependencies", ())
                    if not isinstance(dependencies, list | tuple):
                        raise TypeError("map dependencies must be a JSON array.")
                    for dependency in dependencies:
                        self._decode_map_value(dependency)
                    restored = self._decode_map_value(record["result"])
                except (_InvalidManagedReference, KeyError, TypeError, ValueError):
                    pending.append((index, item, selected))
                    continue
                if validate is not None:
                    try:
                        accepted = bool(validate(restored))
                    except Exception as error:
                        raise RuntimeError(
                            f"map validator raised for restored item {selected!r}: {error}"
                        ) from error
                    if not accepted:
                        pending.append((index, item, selected))
                        continue
                results[index] = restored
            else:
                pending.append((index, item, selected))
        completed = len(materialized) - len(pending)
        self.progress.update(completed=completed, total=len(materialized), message=operation)
        if not pending:
            return results
        if workers == 1:
            for index, item, selected in pending:
                result, dependencies = self._call_map_item(
                    function, item, retries, float(retry_backoff)
                )
                self._save_map_result(root, selected, result, dependencies=dependencies)
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
            futures: dict[Future[Any], tuple[int, str]] = {}
            for index, item, selected in pending:
                future = (
                    pool.submit(
                        self._call_map_item,
                        function,
                        item,
                        retries,
                        float(retry_backoff),
                    )
                    if executor == "thread"
                    else pool.submit(
                        _call_with_retries,
                        function,
                        item,
                        retries,
                        float(retry_backoff),
                    )
                )
                futures[future] = (index, selected)
            try:
                for future in as_completed(futures):
                    index, selected = futures[future]
                    outcome = future.result()
                    if executor == "thread":
                        result, dependencies = cast(tuple[R, tuple[ManagedFile, ...]], outcome)
                    else:
                        result, dependencies = cast(R, outcome), ()
                    self._save_map_result(
                        root,
                        selected,
                        result,
                        dependencies=dependencies,
                    )
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
    def _validate_map_options(
        workers: int,
        executor: str,
        retries: int,
        retry_backoff: float,
    ) -> None:
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("map workers must be an integer >= 1.")
        if executor not in {"thread", "process"}:
            raise ValueError("map executor must be 'thread' or 'process'.")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("map retries must be a non-negative integer.")
        if (
            isinstance(retry_backoff, bool)
            or not isinstance(retry_backoff, int | float)
            or retry_backoff < 0
        ):
            raise ValueError("map retry_backoff must be a non-negative number.")

    def _call_map_item(
        self,
        function: Callable[[T], R],
        item: T,
        retries: int,
        retry_backoff: float,
    ) -> tuple[R, tuple[ManagedFile, ...]]:
        with self.cache.track_dependencies() as tracked:
            result = _call_with_retries(function, item, retries, retry_backoff)
        return result, tuple(tracked)

    def _save_map_result(
        self,
        root: Path,
        key: str,
        value: Any,
        *,
        dependencies: Iterable[ManagedFile] = (),
    ) -> None:
        encoded = self._encode_map_value(value)
        found = [*dependencies, *self._managed_files(value)]
        unique = {
            (managed.scope, managed.key, managed.sha256, managed.size_bytes): managed
            for managed in found
        }
        encoded_dependencies = [
            self._encode_map_value(managed)
            for managed in sorted(unique.values(), key=lambda item: (item.scope, item.key))
        ]
        try:
            json.dumps({"result": encoded, "dependencies": encoded_dependencies}, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError(f"map result for {key!r} must be JSON-compatible: {error}") from error
        atomic_json(
            root / f"{hashlib.sha256(key.encode()).hexdigest()}.json",
            {
                "map_result_version": 2,
                "key": key,
                "result": encoded,
                "dependencies": encoded_dependencies,
            },
        )

    @staticmethod
    def _map_key(item: T, key: Callable[[T], str] | str) -> str:
        if callable(key):
            return str(key(item))
        if not isinstance(key, str) or not key.strip():
            raise TypeError("map key must be a callable or non-empty field name.")
        field_name = key.strip()
        if isinstance(item, Mapping):
            if field_name not in item:
                raise KeyError(f"map item has no field {field_name!r}.")
            return str(item[field_name])
        if not hasattr(item, field_name):
            raise AttributeError(f"map item has no attribute {field_name!r}.")
        return str(getattr(item, field_name))

    @classmethod
    def _encode_map_value(cls, value: Any) -> Any:
        if isinstance(value, ManagedFile):
            return {"$lambdaforge_managed_file": value.reference()}
        if isinstance(value, Mapping):
            return {str(key): cls._encode_map_value(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [cls._encode_map_value(item) for item in value]
        return value

    def _decode_map_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"$lambdaforge_managed_file"}:
                reference = value["$lambdaforge_managed_file"]
                if not isinstance(reference, Mapping):
                    raise _InvalidManagedReference("Malformed managed-file map reference.")
                scope = reference.get("scope")
                resolver = (
                    self.cache.restore_reference
                    if scope == "cache"
                    else self.checkpoints.restore_reference
                    if scope == "checkpoint"
                    else None
                )
                if resolver is None:
                    raise _InvalidManagedReference(
                        f"Unsupported managed-file scope {scope!r} in Work.map."
                    )
                restored = resolver(
                    str(reference["key"]),
                    sha256=str(reference["sha256"]),
                    size_bytes=int(reference["size_bytes"]),
                )
                if restored is None:
                    raise _InvalidManagedReference(str(reference.get("key")))
                return restored
            return {str(key): self._decode_map_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._decode_map_value(item) for item in value]
        return value

    @classmethod
    def _managed_files(cls, value: Any) -> tuple[ManagedFile, ...]:
        if isinstance(value, ManagedFile):
            return (value,)
        if isinstance(value, Mapping):
            return tuple(managed for item in value.values() for managed in cls._managed_files(item))
        if isinstance(value, tuple | list):
            return tuple(managed for item in value for managed in cls._managed_files(item))
        return ()


class _InvalidManagedReference(RuntimeError):
    pass


def _call_with_retries(
    function: Callable[[T], R],
    item: T,
    retries: int,
    retry_backoff: float,
) -> R:
    for attempt in range(retries + 1):
        try:
            return function(item)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(retry_backoff * (2**attempt))
    raise AssertionError("unreachable")
