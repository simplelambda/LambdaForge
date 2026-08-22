"""Small public runtime API available while a LambdaForge callable is running."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lambdaforge.tasks.artifacts import ArtifactDeclaration


@dataclass(frozen=True, slots=True)
class RunContext:
    """Read-only identity and paths for the active callable attempt."""

    name: str
    run_dir: Path
    source_dir: Path
    attempt_id: str
    config_fingerprint: str
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeCapture:
    """Collect public runtime emissions before TaskRunner seals the result."""

    context: RunContext
    metrics: dict[str, int | float] = field(default_factory=dict)
    metric_count: int = 0
    artifacts: list[ArtifactDeclaration] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)


_ACTIVE: ContextVar[RuntimeCapture | None] = ContextVar("lambdaforge_runtime", default=None)


def current() -> RunContext:
    """Return the active run context, or fail clearly outside LambdaForge execution."""
    capture = _ACTIVE.get()
    if capture is None:
        raise RuntimeError("lf.current() is only available inside an active LambdaForge run.")
    return capture.context


def metric(
    name: str,
    value: int | float,
    *,
    step: int | None = None,
    split: str | None = None,
) -> None:
    """Append one durable metric observation and update its final scalar value."""
    capture = _capture("metric")
    if not str(name).strip():
        raise ValueError("Metric names cannot be empty.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Metric values must be numeric scalars.")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError("Metric step must be a non-negative integer or null.")
    record = {
        "name": str(name),
        "value": value,
        "step": step,
        "split": str(split) if split is not None else None,
    }
    path = capture.context.run_dir / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    key = f"{split}_{name}" if split else str(name)
    capture.metrics[key] = value
    capture.metric_count += 1


def artifact(
    name: str,
    path: str | Path,
    *,
    role: str = "other",
    media_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Declare one already-created run-owned artifact for hashing at completion."""
    capture = _capture("artifact")
    resolved = _run_relative(capture.context.run_dir, path)
    if not resolved.exists():
        raise FileNotFoundError(f"Declared artifact does not exist: {resolved}")
    capture.artifacts.append(
        ArtifactDeclaration(
            resolved.relative_to(capture.context.run_dir).as_posix(),
            name=name,
            kind=role,
            media_type=media_type,
            metadata=metadata or {},
        )
    )
    return resolved


def publish_dataset(
    name: str,
    version: str,
    members: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    target_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream members into an immutable DatasetVersion and return its registry record."""
    capture = _capture("publish_dataset")
    from lambdaforge.data.DatasetPublisher import DatasetPublisher

    cluster = os.environ.get("LAMBDAFORGE_CLUSTER", "local")
    configured_root = os.environ.get("LAMBDAFORGE_DATASET_ROOT")
    if cluster != "local" and configured_root is None:
        raise RuntimeError(
            f"Cluster {cluster!r} has no permanent storage.dataset_root for publication."
        )

    record = DatasetPublisher().publish_members(
        name,
        version,
        members,
        source_root=capture.context.run_dir,
        publication_root=Path(
            os.environ.get(
                "LAMBDAFORGE_DATASET_ROOT",
                str(capture.context.source_dir / "runs" / "datasets" / "published"),
            )
        ),
        build_provenance={
            "task_fingerprint": capture.context.config_fingerprint,
            "attempt_id": capture.context.attempt_id,
            "identity": {
                "task_fingerprint": capture.context.config_fingerprint,
                "parameters": _json_value(capture.context.parameters),
                "seed": capture.context.seed,
            },
        },
        cluster=cluster,
        metadata=metadata,
        target_schema=target_schema,
    )
    payload = record.to_dict()
    capture.datasets.append(payload)
    return payload


def _capture(operation: str) -> RuntimeCapture:
    capture = _ACTIVE.get()
    if capture is None:
        raise RuntimeError(f"lf.{operation}() is only available inside an active LambdaForge run.")
    return capture


def _run_relative(root: Path, value: str | Path) -> Path:
    root = root.resolve()
    path = Path(value)
    unresolved = path if path.is_absolute() else root / path
    lexical = Path(os.path.abspath(unresolved))
    if not lexical.is_relative_to(root):
        raise ValueError(f"Runtime output escapes the run directory: {value!s}")
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Runtime outputs cannot traverse symbolic links: {value!s}")
    candidate = lexical.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"Runtime output escapes the run directory: {value!s}")
    return candidate


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@contextmanager
def activate(capture: RuntimeCapture) -> Iterator[RuntimeCapture]:
    """Install a capture for one task invocation (internal and test-friendly)."""
    token: Token[RuntimeCapture | None] = _ACTIVE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE.reset(token)
