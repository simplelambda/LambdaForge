"""Conservative learning-curve prediction and retrospective pruning audit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompletedCurve:
    """One completed seed curve used only for retrospective policy calibration."""

    trial: int
    seed: int | None
    points: tuple[tuple[int, float], ...]
    duration_seconds: float = 0.0
    used_gpu: bool = False


def predict_curve(
    history: Sequence[tuple[int, float]],
    *,
    target_step: int,
    prior_deviation: float,
) -> tuple[float, float]:
    """Return a local-linear prediction whose uncertainty expands with its horizon."""
    recent = tuple(history[-5:])
    if not recent:
        raise ValueError("Curve prediction requires at least one observation.")
    if len(recent) < 2:
        return float(recent[-1][1]), float(prior_deviation)
    steps = [float(item[0]) for item in recent]
    values = [float(item[1]) for item in recent]
    step_mean = statistics.fmean(steps)
    value_mean = statistics.fmean(values)
    denominator = sum((step - step_mean) ** 2 for step in steps)
    slope = (
        sum(
            (step - step_mean) * (value - value_mean)
            for step, value in zip(steps, values, strict=True)
        )
        / denominator
        if denominator > 0
        else 0.0
    )
    intercept = value_mean - slope * step_mean
    prediction = intercept + slope * target_step
    residuals = [
        value - (intercept + slope * step) for step, value in zip(steps, values, strict=True)
    ]
    residual_deviation = statistics.stdev(residuals) if len(residuals) > 2 else prior_deviation
    horizon = max(0.0, target_step - steps[-1]) / max(1.0, steps[-1] - steps[0])
    deviation = max(residual_deviation, prior_deviation * 0.1) * (1.0 + 0.25 * horizon)
    return prediction, deviation


def audit_pruner(
    curves: Sequence[CompletedCurve],
    *,
    mode: str,
    min_step: int,
    confirmations: int,
    probability_threshold: float,
    margin: float,
) -> dict[str, Any]:
    """Backtest candidate pruning over completed curves without fabricating objectives.

    The report is diagnostic: it replays candidate-level decisions on prefixes and checks them
    against the complete curves that were deliberately retained for calibration.
    """
    grouped: dict[int, list[CompletedCurve]] = {}
    for curve in curves:
        if curve.points:
            grouped.setdefault(curve.trial, []).append(curve)
    settings: dict[str, Any] = {
        "probability_threshold": probability_threshold,
        "confirmations": confirmations,
        "equivalence_margin": margin,
        "curve_model": "conservative-local-linear-last-5",
    }
    if len(grouped) < 2:
        return {
            **settings,
            "completed_candidates": len(grouped),
            "calibration_candidates": 0,
            "simulated_performance_prunes": 0,
            "runs_saved": 0,
            "epochs_saved": 0,
            "gpu_seconds_saved": 0.0,
            "false_prune_rate": None,
            "false_pruned_candidates": [],
            "regret": None,
            "probability_brier_score": None,
            "curve_rmse": None,
            "interval_coverage_90": None,
        }
    sign = 1.0 if mode == "max" else -1.0
    final_scores = {
        trial: statistics.fmean(curve.points[-1][1] for curve in values)
        for trial, values in grouped.items()
    }
    best_final = max(final_scores, key=lambda trial: sign * final_scores[trial])
    competitive = {
        trial: sign * (score - final_scores[best_final]) >= -margin
        for trial, score in final_scores.items()
    }
    all_values = [value for curve in curves for _step, value in curve.points]
    prior = max(statistics.pstdev(all_values) if len(all_values) > 1 else 0.0, 1e-6)

    def estimate(trial: int, boundary: int) -> tuple[float, float] | None:
        predictions: list[float] = []
        variances: list[float] = []
        for curve in grouped[trial]:
            prefix = [point for point in curve.points if point[0] <= boundary]
            if not prefix or prefix[-1][0] < min_step:
                continue
            mean, deviation = predict_curve(
                prefix,
                target_step=curve.points[-1][0],
                prior_deviation=prior,
            )
            predictions.append(mean)
            variances.append(deviation**2)
        if not predictions:
            return None
        between = statistics.variance(predictions) if len(predictions) > 1 else 0.0
        return (
            statistics.fmean(predictions),
            math.sqrt(max(1e-12, between + statistics.fmean(variances)) / len(predictions)),
        )

    pruned_at: dict[int, int] = {}
    last_probabilities: dict[int, float] = {}
    for trial, trial_curves in grouped.items():
        boundaries = sorted(
            {
                step
                for curve in trial_curves
                for step, _value in curve.points
                if step >= min_step and step < curve.points[-1][0]
            }
        )
        below = 0
        for boundary in boundaries:
            candidate = estimate(trial, boundary)
            references = {
                other: value
                for other in grouped
                if other != trial
                for value in [estimate(other, boundary)]
                if value is not None
            }
            if candidate is None or not references:
                continue
            reference_trial = max(references, key=lambda item: sign * references[item][0])
            reference = references[reference_trial]
            difference = sign * (candidate[0] - reference[0])
            error = math.sqrt(candidate[1] ** 2 + reference[1] ** 2)
            probability = (
                1.0
                if error <= 0 and difference >= -margin
                else 0.0
                if error <= 0
                else 1.0 - statistics.NormalDist(difference, error).cdf(-margin)
            )
            last_probabilities[trial] = probability
            below = below + 1 if probability < probability_threshold else 0
            if below >= confirmations:
                pruned_at[trial] = boundary
                break

    errors: list[float] = []
    covered = 0
    for curve in curves:
        if len(curve.points) < max(3, min_step + 1):
            continue
        boundary = max(min_step, curve.points[len(curve.points) // 2][0])
        prefix = [point for point in curve.points if point[0] <= boundary]
        if len(prefix) < 2 or prefix[-1][0] >= curve.points[-1][0]:
            continue
        predicted, deviation = predict_curve(
            prefix,
            target_step=curve.points[-1][0],
            prior_deviation=prior,
        )
        error = curve.points[-1][1] - predicted
        errors.append(error)
        covered += abs(error) <= 1.645 * deviation

    runs_saved = epochs_saved = 0
    gpu_seconds_saved = 0.0
    for trial, boundary in pruned_at.items():
        for curve in grouped[trial]:
            remaining = max(0, curve.points[-1][0] - boundary)
            if not remaining:
                continue
            runs_saved += 1
            epochs_saved += remaining
            span = max(1, curve.points[-1][0] - curve.points[0][0] + 1)
            if curve.used_gpu:
                gpu_seconds_saved += curve.duration_seconds * remaining / span
    false_pruned = sorted(trial for trial in pruned_at if competitive[trial])
    survivors = [trial for trial in grouped if trial not in pruned_at]
    survivor_best = max(
        (sign * final_scores[trial] for trial in survivors),
        default=sign * final_scores[best_final],
    )
    regret = max(0.0, sign * final_scores[best_final] - survivor_best)
    brier = [
        (probability - float(competitive[trial])) ** 2
        for trial, probability in last_probabilities.items()
    ]
    return {
        **settings,
        "completed_candidates": len(grouped),
        "calibration_candidates": len(last_probabilities),
        "simulated_performance_prunes": len(pruned_at),
        "runs_saved": runs_saved,
        "epochs_saved": epochs_saved,
        "gpu_seconds_saved": gpu_seconds_saved,
        "false_prune_rate": len(false_pruned) / len(pruned_at) if pruned_at else 0.0,
        "false_pruned_candidates": false_pruned,
        "regret": regret,
        "probability_brier_score": statistics.fmean(brier) if brier else None,
        "curve_rmse": (
            math.sqrt(statistics.fmean(error * error for error in errors)) if errors else None
        ),
        "interval_coverage_90": covered / len(errors) if errors else None,
    }


__all__ = ["CompletedCurve", "audit_pruner", "predict_curve"]
