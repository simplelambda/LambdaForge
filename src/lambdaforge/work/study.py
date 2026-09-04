"""Bounded live study telemetry derived from authoritative Run evidence."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility, aggregate_constraint, pareto_front
from lambdaforge.hpo.StudyInsights import StudyInsightAnalyzer
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
        planned_runs: int | None = None,
        planned_candidates: int | None = None,
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
                "planned_runs": len(specifications) if planned_runs is None else planned_runs,
                "planned_candidates": (
                    len(candidates) if planned_candidates is None else planned_candidates
                ),
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
                candidate = candidates.setdefault(
                    trial,
                    {
                        "trial": trial,
                        "parameters": dict(specification.get("trial_parameters", {})),
                        "state": "pending",
                        "runs": [],
                    },
                )
                key = study_run_key(specification)
                runs = candidate.setdefault("runs", [])
                if not any(isinstance(value, dict) and value.get("key") == key for value in runs):
                    runs.append(
                        {
                            "key": key,
                            "seed": specification.get("seed"),
                            "phase": specification.get("hpo_phase", "search"),
                            "fidelity": dict(specification.get("hpo_fidelity", {})),
                            "state": "scheduled",
                        }
                    )
                candidate["state"] = "running"
            index["candidates"] = [candidates[key] for key in sorted(candidates)]
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
                states = [self._run_state(str(value["key"])).get("state") for value in runs]
                terminal = bool(states) and all(
                    state in {"succeeded", "failed", "pruned", "cancelled"} for state in states
                )
                if finished and trial in active_set:
                    candidate["state"] = (
                        self._terminal_candidate_state(states, successful="completed")
                        if terminal
                        else "running"
                    )
                elif trial in active_set:
                    candidate["state"] = "promoted" if terminal else "running"
                elif trial in ranked_set:
                    candidate["state"] = "eliminated"
            index["finished"] = finished
            index["updated_at_utc"] = _now()
            atomic_json(self.root / "index.json", index)
        snapshot = self.refresh()
        if finished:
            # Analysis cannot alter the controller decision or scientific outcome.
            try:
                from lambdaforge.analysis.StudyAnalysis import StudyAnalysis

                StudyAnalysis.persist(
                    snapshot,
                    self.root / "analysis.json",
                    objective=snapshot.get("objective"),
                    status="final",
                )
            except Exception as error:
                atomic_json(
                    self.root / "analysis-error.json",
                    {"type": type(error).__name__, "message": str(error), "at_utc": _now()},
                )

    def candidates_observed(self, trials: Sequence[int]) -> None:
        """Mark proposed candidates whose initial evidence is now terminal."""
        selected = set(trials)
        with self._lock:
            index = self._index()
            for candidate in index.get("candidates", ()):
                if not isinstance(candidate, dict) or int(candidate["trial"]) not in selected:
                    continue
                runs = [value for value in candidate.get("runs", ()) if isinstance(value, dict)]
                states = [self._run_state(str(value["key"])).get("state") for value in runs]
                if states and all(
                    state in {"succeeded", "failed", "pruned", "cancelled"} for state in states
                ):
                    candidate["state"] = self._terminal_candidate_state(
                        states, successful="observed"
                    )
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
                "phase": specification.get("hpo_phase", "search"),
                "fidelity": dict(specification.get("hpo_fidelity", {})),
                "parameters": dict(specification.get("trial_parameters", {})),
                "run_dir": str(run_dir.resolve()),
                "log_path": str((run_dir / "work.log").resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "training_metrics_path": str(training_metrics_path.resolve()),
                "hpo_metrics_path": str(specification.get("hpo_metrics_path", "")),
                "gpu_index": specification.get("gpu_index"),
                "gpu_token": specification.get("gpu_slot"),
                "failure": None,
                "finished_at_utc": None,
                "prune_reason": None,
                "started_at_utc": _now(),
                "updated_at_utc": _now(),
            },
        )

    def run_finished(self, specification: Mapping[str, Any], result: WorkResult) -> None:
        """Finalize one Run state while retaining only compact scalar summaries."""
        state = (
            "pruned"
            if result.termination_type == "performance_pruned"
            else "paused"
            if result.termination_type == "scheduler_preempted"
            else "aborted"
            if result.termination_type == "aborted"
            else result.status
        )
        self._write_run(
            study_run_key(specification),
            {
                "state": state,
                "trial": int(specification["trial_index"]),
                "seed": result.seed,
                "phase": result.study_phase or specification.get("hpo_phase", "search"),
                "fidelity": dict(result.fidelity or {}),
                "parameters": dict(specification.get("trial_parameters", {})),
                "run_id": result.run_id,
                "attempt_id": result.attempt_id,
                "run_dir": str(result.run_dir.resolve()),
                "log_path": str((result.run_dir / result.logs).resolve()),
                "metrics_path": str((result.run_dir / "metrics.jsonl").resolve()),
                "training_metrics_path": str((result.run_dir / "training-metrics.jsonl").resolve()),
                "metrics": dict(result.metrics),
                "objective_observation": (
                    dict(result.objective_observation)
                    if result.objective_observation is not None
                    else None
                ),
                "duration_seconds": result.duration_seconds,
                "gpu_index": result.gpu_index,
                "gpu_token": result.gpu_token,
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
                "termination_type": result.termination_type,
                "termination": dict(result.termination),
                "updated_at_utc": _now(),
            },
        )

    def run_retrying(
        self,
        specification: Mapping[str, Any],
        *,
        reason: str,
        retry: int,
    ) -> None:
        """Expose a bounded automatic retry while preserving prior Attempt evidence."""
        self._write_run(
            study_run_key(specification),
            {
                "state": "retrying",
                "trial": int(specification["trial_index"]),
                "seed": specification.get("seed"),
                "phase": specification.get("hpo_phase", "search"),
                "fidelity": dict(specification.get("hpo_fidelity", {})),
                "parameters": dict(specification.get("trial_parameters", {})),
                "gpu_index": specification.get("gpu_index"),
                "gpu_token": specification.get("gpu_slot"),
                "retry": retry,
                "retry_reason": reason,
                "failure": None,
                "finished_at_utc": None,
                "updated_at_utc": _now(),
            },
        )

    def queued_action_cancelled(self, specification: Mapping[str, Any], *, reason: str) -> None:
        """Mark a planned Run that was invalidated before consuming scientific compute."""
        self._write_run(
            study_run_key(specification),
            {
                "state": "cancelled",
                "trial": int(specification["trial_index"]),
                "seed": specification.get("seed"),
                "phase": specification.get("hpo_phase", "search"),
                "fidelity": dict(specification.get("hpo_fidelity", {})),
                "parameters": dict(specification.get("trial_parameters", {})),
                "termination_type": "not_started_cancelled",
                "termination": {"type": "not_started_cancelled", "reason": reason},
                "finished_at_utc": _now(),
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
                "phase": specification.get("hpo_phase", "search"),
                "fidelity": dict(specification.get("hpo_fidelity", {})),
                "parameters": dict(specification.get("trial_parameters", {})),
                "failure": {"type": type(error).__name__, "message": str(error)},
                "finished_at_utc": _now(),
                "updated_at_utc": _now(),
            },
        )

    def controller_decision(self, event: Mapping[str, Any]) -> None:
        """Publish a bounded controller-action tail for live human/machine explanations."""
        with self._lock:
            current = self._read(self.root / "controller.json")
            recent = [value for value in current.get("recent", ()) if isinstance(value, dict)]
            persisted_event = dict(event)
            self._append_controller_event(persisted_event)
            recent.append(persisted_event)
            belief = event.get("surrogate_belief")
            if not isinstance(belief, Mapping):
                belief = current.get("surrogate_belief")
            atomic_json(
                self.root / "controller.json",
                {
                    "controller_telemetry_version": 1,
                    "last": dict(event),
                    "recent": recent[-25:],
                    "history_count": int(current.get("history_count", 0)) + 1,
                    "surrogate_belief": dict(belief) if isinstance(belief, Mapping) else None,
                    "updated_at_utc": _now(),
                },
            )

    def _append_controller_event(self, event: Mapping[str, Any]) -> None:
        """Append one complete decision without making the live summary unbounded."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "controller-history.jsonl"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            payload = (
                json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with os.fdopen(descriptor, "ab", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)

    def admission_state(self, diagnostics: Mapping[str, Any]) -> None:
        """Publish the current bounded resource-admission explanation."""
        with self._lock:
            current = self._read(self.root / "admission.json")
            recent = [value for value in current.get("recent", ()) if isinstance(value, dict)]
            event = {**dict(diagnostics), "observed_at_utc": _now()}
            recent.append(event)
            atomic_json(
                self.root / "admission.json",
                {
                    "admission_version": 1,
                    "current": event,
                    "recent": recent[-25:],
                    "updated_at_utc": _now(),
                },
            )

    def refresh(self) -> dict[str, Any]:
        """Fold small per-Run records into the single controller-facing snapshot."""
        with self._lock:
            index = self._index()
            objective = index.get("objective", {})
            objective = objective if isinstance(objective, Mapping) else {}
            objective_metric = str(objective.get("metric", ""))
            objective_mode = str(objective.get("mode", "max"))
            completed = active = queued = paused = failed = pruned = cancelled = 0
            terminations: dict[str, int] = {}
            total_wall_seconds = 0.0
            total_gpu_seconds = 0.0
            for candidate in index.get("candidates", ()):
                if not isinstance(candidate, dict):
                    continue
                observed_runs: list[dict[str, Any]] = []
                for descriptor in candidate.get("runs", ()):
                    if not isinstance(descriptor, Mapping):
                        continue
                    key = str(descriptor["key"])
                    state = self._run_state(key)
                    observation = self._observation_state(key)
                    merged = {**dict(descriptor), **state, **observation}
                    latest, latest_step, best_step, best_objective, live_objective, status = (
                        self._observation_summary(
                            merged,
                            objective=objective,
                        )
                    )
                    merged["latest_metrics"] = latest
                    if live_objective is not None:
                        merged["objective_observation"] = live_objective
                    merged["objective_status"] = status
                    merged["current_observed_objective"] = (
                        live_objective.get("current") if live_objective is not None else None
                    )
                    merged["best_observed_objective"] = best_objective
                    terminal_comparable = selected_state_is_final(merged)
                    merged["final_objective"] = best_objective if terminal_comparable else None
                    if merged.get("state") == "pruned":
                        merged["objective_censoring"] = {
                            "censored": True,
                            "reason": "performance-pruned",
                        }
                    if latest_step is not None:
                        merged["latest_step"] = latest_step
                    if best_step is not None and best_objective is not None:
                        merged["best_step"] = best_step
                        merged["best_objective"] = best_objective
                        if (
                            observation.get("best_step") != best_step
                            or observation.get("best_objective") != best_objective
                        ):
                            # Preserve the all-time optimum incrementally. Refresh reads only a
                            # bounded tail of large metric streams, so an early best epoch must
                            # survive later refreshes and controller restarts.
                            self._write_observation(
                                key,
                                {"best_step": best_step, "best_objective": best_objective},
                            )
                    observed_runs.append(merged)
                    selected = str(merged.get("state", "scheduled"))
                    completed += selected in {"succeeded", "failed", "pruned", "cancelled"}
                    active += selected in {"running", "retrying"}
                    queued += selected == "scheduled"
                    paused += selected == "paused"
                    failed += selected == "failed"
                    pruned += selected == "pruned"
                    cancelled += selected == "cancelled"
                    termination = str(merged.get("termination_type", ""))
                    if termination:
                        terminations[termination] = terminations.get(termination, 0) + 1
                candidate["runs"] = observed_runs
                candidate_wall = sum(
                    float(run.get("duration_seconds", 0.0) or 0.0) for run in observed_runs
                )
                candidate_gpu = sum(
                    float(run.get("duration_seconds", 0.0) or 0.0)
                    for run in observed_runs
                    if run.get("gpu_index") is not None or run.get("gpu_token") is not None
                )
                candidate["cost"] = {
                    "wall_seconds": candidate_wall,
                    "gpu_seconds": candidate_gpu,
                    "runs_started": sum(bool(run.get("started_at_utc")) for run in observed_runs),
                }
                total_wall_seconds += candidate_wall
                total_gpu_seconds += candidate_gpu
                candidate["latest_metrics"] = self._candidate_metrics(observed_runs)
                for stale in (
                    "selection_objective",
                    "selection_seed_count",
                    "selection_standard_error",
                    "raw_metrics",
                    "objective_components",
                ):
                    candidate.pop(stale, None)
                objective_summary = self._candidate_objective_summary(
                    observed_runs,
                    objective_metric=objective_metric,
                    objective_mode=objective_mode,
                    objective_constraints=(
                        objective.get("constraints", {})
                        if isinstance(objective.get("constraints", {}), Mapping)
                        else {}
                    ),
                )
                candidate.update(objective_summary)
                feasibility = objective_summary.get("feasibility", {})
                feasibility = feasibility if isinstance(feasibility, Mapping) else {}
                run_states = {str(run.get("state", "")) for run in observed_runs}
                if (
                    feasibility.get("feasible") is False
                    and "succeeded" in run_states
                    and run_states <= {"succeeded", "failed", "pruned"}
                ):
                    candidate["state"] = "infeasible"
            candidates = [value for value in index.get("candidates", ()) if isinstance(value, dict)]
            for candidate in candidates:
                candidate["pareto_optimal"] = False
            for trial in pareto_front(candidates, objective):
                for candidate in candidates:
                    if candidate.get("trial") == trial:
                        candidate["pareto_optimal"] = True
                        break
            controller = self._read(self.root / "controller.json")
            admission = self._read(self.root / "admission.json")
            recent_decisions = [
                value for value in controller.get("recent", ()) if isinstance(value, Mapping)
            ]
            initialization = next(
                (
                    value
                    for value in reversed(recent_decisions)
                    if value.get("action") == "INITIALIZE"
                ),
                {},
            )
            retained_belief = controller.get("surrogate_belief")
            surrogate_belief = (
                dict(retained_belief)
                if isinstance(retained_belief, Mapping)
                else next(
                    (
                        dict(value["surrogate_belief"])
                        for value in reversed(recent_decisions)
                        if isinstance(value.get("surrogate_belief"), Mapping)
                    ),
                    None,
                )
            )
            raw_slots = initialization.get("slots_total")
            slots_total = int(raw_slots) if isinstance(raw_slots, int) and raw_slots > 0 else None
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
                    "queued_runs": queued,
                    "paused_runs": paused,
                    "failed_runs": failed,
                    "pruned_runs": pruned,
                    "cancelled_before_start": cancelled,
                    "terminations": terminations,
                },
                "controller": {
                    **controller,
                    "scheduler": {
                        "slots_total": slots_total,
                        "slots_active": active,
                        "slots_available": (
                            max(0, slots_total - active) if slots_total is not None else None
                        ),
                        "running": active,
                        "queued": queued,
                        "paused": paused,
                        "preempted": terminations.get("scheduler_preempted", 0),
                    },
                },
                "admission": admission,
                "cost": {
                    "wall_seconds": total_wall_seconds,
                    "gpu_seconds": total_gpu_seconds,
                },
                "hpo_analysis": (
                    StudyInsightAnalyzer.analyze(
                        candidates,
                        objective,
                    )
                    if str(index.get("strategy", "")) == "adaptive"
                    else None
                ),
                "surrogate_belief": surrogate_belief,
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
                        "message": (f"{active} active; {failed} failed; {pruned} pruned"),
                    },
                )
            return snapshot

    def _write_run(self, key: str, updates: Mapping[str, Any]) -> None:
        path = self.root / "runs" / f"{key}.json"
        current = self._read(path)
        atomic_json(path, {**current, **dict(updates), "key": key})

    def _run_state(self, key: str) -> dict[str, Any]:
        return self._read(self.root / "runs" / f"{key}.json")

    def _write_observation(self, key: str, updates: Mapping[str, Any]) -> None:
        """Persist controller-owned aggregates without rewriting worker-owned state."""
        path = self.root / "observations" / f"{key}.json"
        current = self._read(path)
        atomic_json(path, {**current, **dict(updates), "key": key})

    def _observation_state(self, key: str) -> dict[str, Any]:
        return self._read(self.root / "observations" / f"{key}.json")

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
    def _observation_summary(
        run: Mapping[str, Any],
        *,
        objective: Mapping[str, Any],
    ) -> tuple[
        dict[str, float],
        int | None,
        int | None,
        float | None,
        Mapping[str, Any] | None,
        Mapping[str, Any],
    ]:
        objective_metric = str(objective.get("metric", ""))
        objective_mode = str(objective.get("mode", "max"))
        latest: dict[str, float] = {
            str(key): float(value)
            for key, value in dict(run.get("metrics", {})).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        latest_metric_steps: dict[str, int] = {}
        raw_latest_step = run.get("latest_step")
        latest_step = raw_latest_step if isinstance(raw_latest_step, int) else None
        raw_best_step = run.get("best_step")
        best_step = raw_best_step if isinstance(raw_best_step, int) else None
        raw_best = run.get("best_objective")
        best_objective = (
            float(raw_best)
            if isinstance(raw_best, int | float) and not isinstance(raw_best, bool)
            else None
        )
        records: list[Mapping[str, Any]] = []
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
                    if value.get("kind") == "chart-filter":
                        continue
                    if isinstance(value, Mapping):
                        records.append(value)
                    name = str(value["name"])
                    key = f"{value['split']}_{name}" if value.get("split") else name
                    metric = value.get("value")
                    numeric: float | None = None
                    step = value.get("step")
                    if isinstance(metric, int | float) and not isinstance(metric, bool):
                        numeric = float(metric)
                        # Streams can overlap when worker telemetry is adopted. Current means
                        # the greatest scientific step, not the last duplicate read from disk.
                        if isinstance(step, int) and not isinstance(step, bool):
                            previous = latest_metric_steps.get(key)
                            if previous is None or step >= previous:
                                latest[key] = numeric
                                latest_metric_steps[key] = step
                        elif key not in latest_metric_steps:
                            latest[key] = numeric
                    if isinstance(step, int) and not isinstance(step, bool):
                        latest_step = max(latest_step or step, step)
                        if (
                            numeric is not None
                            and key == objective_metric
                            and (
                                best_objective is None
                                or (objective_mode == "min" and numeric < best_objective)
                                or (objective_mode != "min" and numeric > best_objective)
                            )
                        ):
                            best_step, best_objective = step, numeric
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        # A terminal result computes these fields from the complete ordered streams. Reapply it
        # after the bounded live-tail scan so duplicated callback/adoption records cannot turn
        # current into best or otherwise change the terminal meaning.
        persisted = run.get("objective_observation")
        persisted = persisted if isinstance(persisted, Mapping) else None
        evaluator = ObjectiveUtility(objective) if objective_metric else None
        live = evaluator.observation(records, fallback=latest) if evaluator is not None else None
        status = (
            evaluator.status(records, fallback=latest)
            if evaluator is not None
            else {
                "status": "unavailable",
                "missing_components": [],
                "latest_step": latest_step,
                "latest_complete_step": None,
            }
        )
        selected_source = persisted if persisted is not None else live
        selected = dict(selected_source) if selected_source is not None else None
        if selected is not None and selected.get("metric") == objective_metric:
            raw_current = selected.get("current")
            if isinstance(raw_current, int | float) and not isinstance(raw_current, bool):
                latest[objective_metric] = float(raw_current)
            raw_current_step = selected.get("current_step")
            if isinstance(raw_current_step, int) and not isinstance(raw_current_step, bool):
                latest_step = raw_current_step
            raw_objective_best = selected.get("best")
            if isinstance(raw_objective_best, int | float) and not isinstance(
                raw_objective_best, bool
            ):
                observed_best = float(raw_objective_best)
                observed_best_step = selected.get("best_step")
                replaces_history = (
                    persisted is not None
                    or best_objective is None
                    or (objective_mode == "min" and observed_best < best_objective)
                    or (objective_mode != "min" and observed_best > best_objective)
                )
                if replaces_history:
                    best_objective = observed_best
                    if isinstance(observed_best_step, int) and not isinstance(
                        observed_best_step, bool
                    ):
                        best_step = observed_best_step
                else:
                    # The bounded live tail cannot disprove an earlier all-time optimum.
                    # Keep the structured observation internally consistent for TUI/API users.
                    selected["best"] = best_objective
                    selected["best_step"] = best_step
        if selected is not None:
            status = {
                "status": "complete",
                "missing_components": [],
                "latest_step": selected.get("current_step", latest_step),
                "latest_complete_step": selected.get("current_step", latest_step),
            }
        return latest, latest_step, best_step, best_objective, selected, status

    @staticmethod
    def _terminal_candidate_state(states: Sequence[Any], *, successful: str) -> str:
        """Distinguish scientific pruning from infrastructure/consumer failure."""
        normalized = {str(state) for state in states}
        if "failed" in normalized:
            return "failed"
        if "pruned" in normalized:
            return "pruned"
        if "succeeded" in normalized:
            return successful
        if normalized and normalized <= {"pruned", "cancelled"}:
            return "pruned"
        return "failed"

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

    @staticmethod
    def _candidate_objective_summary(
        runs: Sequence[Mapping[str, Any]],
        *,
        objective_metric: str,
        objective_mode: str,
        objective_constraints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expose display values separately from the seed-mean used by adaptive HPO."""
        if not objective_metric:
            return {}
        current_values: list[float] = []
        selectable: list[tuple[float, Mapping[str, Any]]] = []
        provisional: list[tuple[float, Mapping[str, Any]]] = []
        censored = any(run.get("state") == "pruned" for run in runs)
        for run in runs:
            latest = run.get("latest_metrics", {})
            latest = latest if isinstance(latest, Mapping) else {}
            current = latest.get(objective_metric)
            if isinstance(current, int | float) and not isinstance(current, bool):
                current_values.append(float(current))
            best = run.get("best_objective")
            if not isinstance(best, int | float) or isinstance(best, bool):
                continue
            pair = (float(best), run)
            if run.get("state") == "succeeded":
                selectable.append(pair)
            elif run.get("state") in {"running", "retrying"}:
                provisional.append(pair)
        evidence = selectable or provisional
        constraints = StudyTelemetry._candidate_constraint_summary(
            runs, objective_constraints or {}
        )
        output: dict[str, Any] = {"feasibility": constraints, "partially_censored": censored}
        if current_values:
            output["current_objective"] = sum(current_values) / len(current_values)
        if selectable and constraints["feasible"] and not censored:
            selection_values = [value for value, _run in selectable]
            output["selection_objective"] = sum(selection_values) / len(selection_values)
            output["selection_seed_count"] = len(selectable)
            output["selection_standard_error"] = (
                statistics.stdev(selection_values) / math.sqrt(len(selection_values))
                if len(selection_values) > 1
                else None
            )
        if evidence:
            component_groups: dict[str, dict[str, list[float]]] = {}
            for _value, run in evidence:
                observation = run.get("objective_observation", {})
                observation = observation if isinstance(observation, Mapping) else {}
                components = observation.get(
                    "components" if run.get("state") == "succeeded" else "current_components",
                    {},
                )
                components = components if isinstance(components, Mapping) else {}
                for name, raw_detail in components.items():
                    detail = raw_detail if isinstance(raw_detail, Mapping) else {}
                    grouped = component_groups.setdefault(str(name), {})
                    for field in ("raw", "normalized", "weight", "contribution"):
                        raw = detail.get(field)
                        if isinstance(raw, int | float) and not isinstance(raw, bool):
                            grouped.setdefault(field, []).append(float(raw))
            if component_groups:
                output["raw_metrics"] = {
                    name: sum(fields["raw"]) / len(fields["raw"])
                    for name, fields in component_groups.items()
                    if fields.get("raw")
                }
                output["objective_components"] = {
                    name: {
                        field: sum(values) / len(values)
                        for field, values in fields.items()
                        if values
                    }
                    for name, fields in component_groups.items()
                }
            best_value, best_run = (min if objective_mode == "min" else max)(
                evidence, key=lambda item: item[0]
            )
            output.update(
                {
                    "best_objective": best_value,
                    "best_seed": best_run.get("seed"),
                    "best_step": best_run.get("best_step"),
                    "best_is_provisional": not bool(selectable),
                }
            )
        return output

    @staticmethod
    def _candidate_constraint_summary(
        runs: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fold best-epoch guardrail values across completed seeds for the read model."""
        if not rules:
            return {"feasible": True, "constraints": {}}
        completed = [run for run in runs if run.get("state") == "succeeded"]
        summary: dict[str, Any] = {}
        for name, raw_rule in rules.items():
            rule = raw_rule if isinstance(raw_rule, Mapping) else {}
            values: list[float] = []
            missing = 0
            for run in completed:
                observation = run.get("objective_observation", {})
                observation = observation if isinstance(observation, Mapping) else {}
                observed = observation.get("constraints", {})
                observed = observed if isinstance(observed, Mapping) else {}
                detail = observed.get(str(name), {})
                detail = detail if isinstance(detail, Mapping) else {}
                value = detail.get("value")
                if isinstance(value, int | float) and not isinstance(value, bool):
                    values.append(float(value))
                else:
                    missing += 1
            summary[str(name)] = aggregate_constraint(values, rule, missing=missing)
        return {
            "feasible": bool(completed)
            and all(bool(value["satisfied"]) for value in summary.values()),
            "constraints": summary,
        }


def study_run_key(specification: Mapping[str, Any]) -> str:
    """Return a stable path-safe key for one candidate/seed member."""
    trial = int(specification["trial_index"])
    seed = specification.get("seed")
    seed_token = "none" if seed is None else f"n{abs(int(seed))}" if int(seed) < 0 else str(seed)
    return f"trial-{trial:05d}-seed-{seed_token}"


def selected_state_is_final(run: Mapping[str, Any]) -> bool:
    """Return whether one Run has terminal, comparable scientific evidence."""
    if run.get("state") != "succeeded" or run.get("termination_type", "completed") != "completed":
        return False
    fidelity = run.get("fidelity")
    if not isinstance(fidelity, Mapping) or not fidelity:
        return True
    target = fidelity.get("target")
    maximum = fidelity.get("maximum")
    return (
        isinstance(target, int)
        and not isinstance(target, bool)
        and isinstance(maximum, int)
        and not isinstance(maximum, bool)
        and target >= maximum
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["StudyTelemetry", "selected_state_is_final", "study_run_key"]
