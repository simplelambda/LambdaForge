"""Bounded live study telemetry derived from authoritative Run evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from lambdaforge.work.models import WorkResult, atomic_json


class StudyTelemetry:
    """Publish one compact study index while Runs keep their own logs and metrics."""

    def __init__(self, root: Path, *, progress_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.progress_path = progress_path.resolve() if progress_path is not None else None
        self._lock = Lock()

    @classmethod
    def from_environment(cls) -> StudyTelemetry | None:
        configured = os.environ.get("LAMBDAFORGE_STUDY_PATH")
        if not configured:
            return None
        progress = os.environ.get("LAMBDAFORGE_PROGRESS_PATH")
        return cls(Path(configured), progress_path=Path(progress) if progress else None)

    def initialize(
        self,
        *,
        name: str,
        execution_id: str,
        strategy: str,
        objective: Mapping[str, Any] | None,
        specifications: Sequence[Mapping[str, Any]],
    ) -> None:
        """Create a bounded candidate catalogue before launching any child process."""
        candidates: dict[int, dict[str, Any]] = {}
        for specification in specifications:
            trial = int(specification["trial_index"])
            candidates.setdefault(
                trial,
                {
                    "trial": trial,
                    "parameters": dict(specification.get("trial_parameters", {})),
                    "state": "pending",
                    "runs": [],
                },
            )
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            self.root / "index.json",
            {
                "study_telemetry_version": 1,
                "name": name,
                "execution_id": execution_id,
                "strategy": strategy,
                "objective": dict(objective or {}),
                "planned_runs": len(specifications),
                "candidates": [candidates[key] for key in sorted(candidates)],
                "created_at_utc": _now(),
                "updated_at_utc": _now(),
            },
        )
        self.refresh()

    def schedule(self, specifications: Sequence[Mapping[str, Any]]) -> None:
        """Register the next bounded set of candidate/seed Runs atomically."""
        with self._lock:
            index = self._index()
            candidates = {
                int(value["trial"]): value
                for value in index.get("candidates", ())
                if isinstance(value, dict)
            }
            for specification in specifications:
                trial = int(specification["trial_index"])
                candidate = candidates[trial]
                key = study_run_key(specification)
                runs = candidate.setdefault("runs", [])
                if not any(isinstance(value, dict) and value.get("key") == key for value in runs):
                    runs.append(
                        {
                            "key": key,
                            "seed": specification.get("seed"),
                            "state": "scheduled",
                        }
                    )
                candidate["state"] = "running"
            index["updated_at_utc"] = _now()
            atomic_json(self.root / "index.json", index)
        self.refresh()

    def candidate_states(
        self,
        *,
        active: Sequence[int],
        ranked: Sequence[int],
        finished: bool,
    ) -> None:
        """Persist controller decisions without letting workers mutate shared JSON."""
        active_set, ranked_set = set(active), set(ranked)
        with self._lock:
            index = self._index()
            for candidate in index.get("candidates", ()):
                if not isinstance(candidate, dict):
                    continue
                trial = int(candidate["trial"])
                runs = [value for value in candidate.get("runs", ()) if isinstance(value, dict)]
                states = [
                    self._run_state(str(value["key"])).get("state") for value in runs
                ]
                terminal = bool(states) and all(
                    state in {"succeeded", "failed", "pruned"} for state in states
                )
                if finished and trial in active_set:
                    candidate["state"] = (
                        "completed" if terminal and "succeeded" in states else "failed"
                    )
                elif trial in active_set:
                    candidate["state"] = "promoted" if terminal else "running"
                elif trial in ranked_set:
                    candidate["state"] = "eliminated"
            index["finished"] = finished
            index["updated_at_utc"] = _now()
            atomic_json(self.root / "index.json", index)
        self.refresh()

    def run_started(
        self,
        specification: Mapping[str, Any],
        *,
        run_dir: Path,
        metrics_path: Path,
        training_metrics_path: Path,
    ) -> None:
        """Publish one process-owned state record; no shared-file lock is required."""
        self._write_run(
            study_run_key(specification),
            {
                "state": "running",
                "trial": int(specification["trial_index"]),
                "seed": specification.get("seed"),
                "parameters": dict(specification.get("trial_parameters", {})),
                "run_dir": str(run_dir.resolve()),
                "log_path": str((run_dir / "work.log").resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "training_metrics_path": str(training_metrics_path.resolve()),
                "started_at_utc": _now(),
                "updated_at_utc": _now(),
            },
        )

    def run_finished(self, specification: Mapping[str, Any], result: WorkResult) -> None:
        """Finalize one Run state while retaining only compact scalar summaries."""
        state = "pruned" if result.pruned else result.status
        self._write_run(
            study_run_key(specification),
            {
                "state": state,
                "trial": int(specification["trial_index"]),
                "seed": result.seed,
                "parameters": dict(specification.get("trial_parameters", {})),
                "run_id": result.run_id,
                "attempt_id": result.attempt_id,
                "run_dir": str(result.run_dir.resolve()),
                "log_path": str((result.run_dir / result.logs).resolve()),
                "metrics_path": str((result.run_dir / "metrics.jsonl").resolve()),
                "training_metrics_path": str(
                    (result.run_dir / "training-metrics.jsonl").resolve()
                ),
                "metrics": dict(result.metrics),
                "duration_seconds": result.duration_seconds,
                "finished_at_utc": result.finished_at_utc,
                "failure": (
                    {
                        "type": result.failure.get("type"),
                        "message": result.failure.get("message"),
                    }
                    if isinstance(result.failure, Mapping)
                    else None
                ),
                "prune_reason": result.prune_reason,
                "updated_at_utc": _now(),
            },
        )

    def run_crashed(self, specification: Mapping[str, Any], error: BaseException) -> None:
        """Expose infrastructure/setup failures that occur before a WorkResult exists."""
        self._write_run(
            study_run_key(specification),
            {
                "state": "failed",
                "trial": int(specification["trial_index"]),
                "seed": specification.get("seed"),
                "parameters": dict(specification.get("trial_parameters", {})),
                "failure": {"type": type(error).__name__, "message": str(error)},
                "finished_at_utc": _now(),
                "updated_at_utc": _now(),
            },
        )

    def refresh(self) -> dict[str, Any]:
        """Fold small per-Run records into the single controller-facing snapshot."""
        with self._lock:
            index = self._index()
            completed = active = failed = pruned = 0
            for candidate in index.get("candidates", ()):
                if not isinstance(candidate, dict):
                    continue
                observed_runs: list[dict[str, Any]] = []
                for descriptor in candidate.get("runs", ()):
                    if not isinstance(descriptor, Mapping):
                        continue
                    state = self._run_state(str(descriptor["key"]))
                    merged = {**dict(descriptor), **state}
                    latest, latest_step = self._latest_observation(merged)
                    merged["latest_metrics"] = latest
                    if latest_step is not None:
                        merged["latest_step"] = latest_step
                    observed_runs.append(merged)
                    selected = str(merged.get("state", "scheduled"))
                    completed += selected in {"succeeded", "failed", "pruned"}
                    active += selected == "running"
                    failed += selected == "failed"
                    pruned += selected == "pruned"
                candidate["runs"] = observed_runs
                candidate["latest_metrics"] = self._candidate_metrics(observed_runs)
            snapshot = {
                **index,
                "counts": {
                    "candidates": len(index.get("candidates", ())),
                    "scheduled_runs": sum(
                        len(value.get("runs", ()))
                        for value in index.get("candidates", ())
                        if isinstance(value, Mapping)
                    ),
                    "completed_runs": completed,
                    "active_runs": active,
                    "failed_runs": failed,
                    "pruned_runs": pruned,
                },
                "updated_at_utc": _now(),
            }
            atomic_json(self.root / "summary.json", snapshot)
            if self.progress_path is not None:
                scheduled = int(snapshot["counts"]["scheduled_runs"])
                atomic_json(
                    self.progress_path,
                    {
                        "completed": completed,
                        "total": (
                            scheduled
                            if bool(index.get("finished"))
                            else int(index.get("planned_runs", completed))
                        ),
                        "message": (
                            f"{active} active; {failed} failed; {pruned} pruned"
                        ),
                    },
                )
            return snapshot

    def _write_run(self, key: str, updates: Mapping[str, Any]) -> None:
        path = self.root / "runs" / f"{key}.json"
        current = self._read(path)
        atomic_json(path, {**current, **dict(updates), "key": key})

    def _run_state(self, key: str) -> dict[str, Any]:
        return self._read(self.root / "runs" / f"{key}.json")

    def _index(self) -> dict[str, Any]:
        return self._read(self.root / "index.json")

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        if not path.is_file() or path.is_symlink():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _latest_observation(
        run: Mapping[str, Any],
    ) -> tuple[dict[str, float], int | None]:
        latest: dict[str, float] = {
            str(key): float(value)
            for key, value in dict(run.get("metrics", {})).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        latest_step: int | None = None
        for field in ("metrics_path", "training_metrics_path"):
            raw = run.get(field)
            if not isinstance(raw, str):
                continue
            path = Path(raw)
            if not path.is_file() or path.is_symlink():
                continue
            try:
                # Metric telemetry is intentionally line-oriented. Reading at most the
                # recent 256 KiB keeps refresh cost bounded even for very long studies.
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 256 * 1024))
                    text = handle.read().decode("utf-8", errors="ignore")
                for line in text.splitlines()[1 if size > 256 * 1024 else 0 :]:
                    value = json.loads(line)
                    name = str(value["name"])
                    key = f"{value['split']}_{name}" if value.get("split") else name
                    metric = value.get("value")
                    if isinstance(metric, int | float) and not isinstance(metric, bool):
                        latest[key] = float(metric)
                    step = value.get("step")
                    if isinstance(step, int) and not isinstance(step, bool):
                        latest_step = max(latest_step or step, step)
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return latest, latest_step

    @staticmethod
    def _candidate_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for run in runs:
            if run.get("state") not in {"succeeded", "running"}:
                continue
            for name, value in dict(run.get("latest_metrics", {})).items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    grouped.setdefault(str(name), []).append(float(value))
        return {name: sum(values) / len(values) for name, values in grouped.items() if values}


def study_run_key(specification: Mapping[str, Any]) -> str:
    """Return a stable path-safe key for one candidate/seed member."""
    trial = int(specification["trial_index"])
    seed = specification.get("seed")
    seed_token = "none" if seed is None else f"n{abs(int(seed))}" if int(seed) < 0 else str(seed)
    return f"trial-{trial:05d}-seed-{seed_token}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["StudyTelemetry", "study_run_key"]
