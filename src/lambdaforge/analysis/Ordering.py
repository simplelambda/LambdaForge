"""Central objective ordering semantics for post-study analysis."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def better(left: float, right: float, mode: str) -> bool:
    """Return whether ``left`` is scientifically better than ``right``."""
    _validate_mode(mode)
    return left < right if mode == "min" else left > right


def best(values: Iterable[T], *, key: Any, mode: str) -> T:
    """Select the best value using one explicit objective direction."""
    _validate_mode(mode)
    return (min if mode == "min" else max)(values, key=key)


def signed_improvement(candidate: float, reference: float, mode: str) -> float:
    """Return a positive number when candidate improves over reference."""
    _validate_mode(mode)
    return (candidate - reference) * (1.0 if mode == "max" else -1.0)


def supported_region(
    points: Sequence[Mapping[str, Any]], *, selected: Mapping[str, Any] | None, mode: str
) -> list[float] | None:
    """Return supported x values within one uncertainty unit of the predicted optimum."""
    if selected is None:
        return None
    optimum = float(selected["predicted_objective"])
    uncertainty = max(float(selected.get("uncertainty", 0.0)), 1e-12)
    included = [
        float(point["x"])
        for point in points
        if point.get("support_count", 0) > 0
        and signed_improvement(float(point["predicted_objective"]), optimum, mode) >= -uncertainty
    ]
    return [min(included), max(included)] if included else None


def _validate_mode(mode: str) -> None:
    if mode not in {"min", "max"}:
        raise ValueError(f"Objective mode must be 'min' or 'max', received {mode!r}.")


__all__ = ["best", "better", "signed_improvement", "supported_region"]
