"""Pure aggregate lifecycle projection over existing Study obligations and latest Runs.

This is not another state machine: scheduling, scientific convergence and retry remain owned by
the native planner. No file access, fitting, retry or mutation occurs in these functions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def required_evidence(snapshot: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the existing required-evidence contract once for live and terminal views."""
    requirements = {
        str(value.get("key")): value
        for value in (design.get("evidence") or {}).get("requirements", ())
        if isinstance(value, Mapping) and value.get("required") is True
    }
    candidates = [value for value in snapshot.get("candidates", ()) if isinstance(value, Mapping)]
    if design.get("type") == "adaptive":
        proposed = {int(value["trial"]) for value in candidates if value.get("trial") is not None}
        requirements = {
            key: value for key, value in requirements.items() if value.get("candidate") in proposed
        }
    states: dict[str, str] = {}
    for candidate in candidates:
        for run in candidate.get("runs", ()):
            if not isinstance(run, Mapping):
                continue
            requirement = run.get("evidence_requirement")
            if isinstance(requirement, Mapping) and requirement.get("key"):
                key = str(requirement["key"])
                if requirement.get("required") is True:
                    requirements.setdefault(key, dict(requirement))
                states[key] = str(run.get("state", "scheduled"))
    missing = sorted(key for key in requirements if states.get(key) not in {"succeeded", "pruned"})
    return {
        "required_runs": len(requirements),
        "required_completed": sum(states.get(key) == "succeeded" for key in requirements),
        "required_pruned": sum(states.get(key) == "pruned" for key in requirements),
        "required_failed": sum(states.get(key) in {"failed", "infeasible"} for key in requirements),
        "required_missing": len(missing),
        "missing_requirement_keys": missing,
    }


def execution_evidence(
    runs: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> dict[str, Any]:
    """Adapt native final Run cells to the same required-evidence authority as telemetry."""
    observed: dict[tuple[int, Any, str], dict[int, Mapping[str, Any]]] = {}
    candidates: dict[int, dict[str, Any]] = {}
    for run in runs:
        # A repeated one-candidate design has no parameterized WorkTrial in native results.
        trial = int((run.get("trial") or {}).get("index", 1))
        phase = str(run.get("study_phase") or "search")
        if phase == "sweep" and design.get("type") in {"sweep", "repeated"}:
            phase = "search"  # Fixed dispatcher phase; the native design owns its evidence key.
        target = int((run.get("fidelity") or {}).get("target", 0))
        rungs = observed.setdefault((trial, run.get("seed"), phase), {})
        previous = rungs.get(target)
        if previous is None or int(run.get("attempt_number", 1)) >= int(
            previous.get("attempt_number", 1)
        ):
            rungs[target] = run
        candidates.setdefault(trial, {"trial": trial, "runs": []})
    for requirement in (design.get("evidence") or {}).get("requirements", ()):
        if not isinstance(requirement, Mapping):
            continue
        trial = int(requirement["candidate"])
        candidate = candidates.get(trial)
        if candidate is None:
            continue
        phase = str(requirement.get("phase") or "search")
        requirement_target = requirement.get("fidelity")
        rungs = observed.get((trial, requirement.get("seed"), phase), {})
        # A later rung can satisfy an unbounded design obligation, but never confirmation
        # with the same seed or an explicitly different cumulative fidelity requirement.
        outcome = (
            (rungs[max(rungs)] if rungs else None)
            if requirement_target is None
            else rungs.get(int(requirement_target))
        )
        if outcome is None:
            continue
        state = (
            "pruned"
            if outcome.get("termination_type") == "performance_pruned" or outcome.get("pruned")
            else "succeeded"
            if outcome.get("status") == "succeeded"
            and outcome.get("termination_type", "completed") == "completed"
            else "failed"
        )
        candidate["runs"].append({"state": state, "evidence_requirement": requirement})
    return required_evidence({"candidates": list(candidates.values())}, design)


@dataclass(frozen=True, slots=True)
class StudyState:
    """Operational activity, evidence completeness and health are independent dimensions."""

    operational: str
    evidence: str
    health: str
    prior_attempt_failures: int
    censored_runs: int
    required_missing: int
    physical_history_complete: bool

    @property
    def final_status(self) -> str:
        """Project the legacy terminal label from evidence, not physical Attempt failures."""
        return (
            "succeeded"
            if self.operational == "stopped" and self.evidence == "complete"
            else "failed"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_state_version": 1,
            "operational": self.operational,
            "evidence": self.evidence,
            "health": self.health,
            "prior_attempt_failures": self.prior_attempt_failures,
            "censored_runs": self.censored_runs,
            "required_missing": self.required_missing,
            "physical_history_complete": self.physical_history_complete,
        }

    @classmethod
    def from_execution(
        cls, summary: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], *, status: str
    ) -> StudyState:
        """Read current/legacy Execution envelopes without opening physical Attempt files.

        Earlier physical failures are history, not current logical outcomes. Missing older
        Attempts remain unknown history rather than guessed failures or refunded cost.
        """
        grouped: dict[tuple[Any, Any], dict[int, Mapping[str, Any]]] = {}
        for run in runs:
            grouped.setdefault((run.get("name"), run.get("run_id")), {})[
                int(run.get("attempt_number", 1))
            ] = run
        logical = []
        for attempts in grouped.values():
            ordinal = max(attempts)
            latest = attempts[ordinal]
            termination = latest.get("termination_type")
            logical.append(
                {
                    "state": (
                        "pruned"
                        if termination == "performance_pruned" or latest.get("pruned")
                        else "paused"
                        if termination == "scheduler_preempted"
                        else str(latest.get("status", "unknown"))
                    ),
                    "attempt_statistics": {
                        "failed": sum(
                            attempt.get("status") == "failed" for attempt in attempts.values()
                        ),
                        "history_complete": len(attempts) == ordinal and min(attempts) == 1,
                    },
                }
            )
        evidence = summary.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        return cls.from_snapshot(
            {
                **dict(evidence),
                "status": status,
                "finished": status
                in {
                    "succeeded",
                    "failed",
                    "cancelled",
                    "completed",
                    "completed_with_failures",
                    "incomplete",
                    "aborted",
                    "timeout",
                    "interrupted",
                },
                "candidates": [{"runs": logical}],
            }
        )

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> StudyState:
        """Read latest logical outcomes; never let earlier failed Attempts poison success."""
        runs = [
            run
            for candidate in snapshot.get("candidates", ())
            if isinstance(candidate, Mapping)
            for run in candidate.get("runs", ())
            if isinstance(run, Mapping)
        ]
        states = Counter(str(run.get("state", "scheduled")) for run in runs)
        failures = states["failed"] + states["infeasible"]
        recovering = states["retrying"]
        recovered = sum(
            max(
                0,
                int((run.get("attempt_statistics") or {}).get("failed", 0))
                - int(run.get("state") == "failed"),
            )
            for run in runs
        )
        missing = int(snapshot.get("required_missing", 0) or 0)
        finished = snapshot.get("finished") is True
        legacy_incomplete = snapshot.get("design_status") == "incomplete"
        operational = (
            "cancelling"
            if snapshot.get("status") == "cancelling"
            else "stopped"
            if finished
            else "running"
            if states["running"]
            else "paused"
            if states["paused"] and not (states["scheduled"] or recovering)
            else "waiting_resources"
            if states["scheduled"] or recovering
            else "stopped"
            if failures
            else "preparing"
        )
        evidence = (
            "pending_recovery"
            if recovering or states["paused"]
            else "incomplete"
            if finished and (missing or failures or legacy_incomplete or not runs)
            else "complete"
            if finished and runs
            else "partial"
            if states["succeeded"] or states["pruned"]
            else "pending"
        )
        health = (
            "terminal_failure"
            if finished and failures
            else "blocked"
            if failures and not states["running"] and not recovering
            else "degraded"
            if failures or recovering or recovered
            else "healthy"
        )
        return cls(
            operational,
            evidence,
            health,
            recovered,
            states["pruned"],
            missing,
            all(
                (run.get("attempt_statistics") or {}).get("history_complete") is True
                for run in runs
            ),
        )
