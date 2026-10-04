"""Execution-only boundary shared by fixed, sequential-sweep and adaptive Study planners."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.work.models import WorkResult
from lambdaforge.work.study import StudyTelemetry

Specifications = Sequence[Mapping[str, Any]]
ResultCallback = Callable[[WorkResult, Specifications, Specifications], Sequence[dict[str, Any]]]
FrontierCallback = Callable[[Specifications, Specifications], Sequence[dict[str, Any]]]
InfeasibleCallback = Callable[[Specifications, str], Sequence[dict[str, Any]]]


class StudyDispatcher(Protocol):
    """Execute exact planner proposals; callbacks and scientific policy stay at the owner.

    A dispatcher returns only after collecting its admitted work. It must never allocate a
    seed, propose a candidate, fit an optimizer or turn an unknown owner into a lost Attempt.
    This is an internal dependency-injection boundary, not another Work runner.
    """

    def __call__(
        self,
        specifications: Sequence[dict[str, Any]],
        *,
        resources: ResourceRequest,
        policy: AdaptiveSearchPolicy,
        objective_metric: str,
        objective_mode: str,
        objective: Mapping[str, Any],
        historical_results: Sequence[WorkResult],
        parallelism: int,
        telemetry: StudyTelemetry | None = None,
        on_result: ResultCallback | None = None,
        on_resource_blocked: FrontierCallback | None = None,
        on_resource_infeasible: InfeasibleCallback | None = None,
        on_queued_cancel: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> tuple[WorkResult, ...]:
        """Dispatch the current queue and return exact results to the original planner."""
        ...
