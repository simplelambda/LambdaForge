"""Feature-aware conservative GPU-memory predictor."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveMemoryObservation import AdaptiveMemoryObservation
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.PredictiveEstimate import PredictiveEstimate


class FeatureAwareMemoryModel:
    """Predict ``M(x, z)`` with conservative quantiles and censored OOM evidence.

    Numeric hyperparameters and consumer-declared resource features form a log-linear model with
    non-negative slopes. OOMs enter as conservative pseudo-observations above their known lower
    bound. Distance outside the observed feature box inflates uncertainty. Before enough evidence
    exists, a declared logical budget or an optional parameter-state estimate is used; that
    structural estimate intentionally excludes activations and library workspaces.
    """

    def __init__(
        self,
        *,
        cold_start_bytes: int,
        headroom_bytes: int = 0,
        safety_quantile: float = 0.99,
        min_observations: int = 3,
        parameter_count_feature: str = "parameter_count",
        bytes_per_parameter: int = 4,
        gradient_copies: float = 1.0,
        optimizer_copies: float = 2.0,
        buffer_bytes: int = 0,
        censored_multiplier: float = 1.1,
    ) -> None:
        if (
            cold_start_bytes < 0
            or headroom_bytes < 0
            or min_observations < 1
            or bytes_per_parameter < 1
            or gradient_copies < 0
            or optimizer_copies < 0
            or buffer_bytes < 0
            or censored_multiplier <= 1
        ):
            raise ValueError("Invalid feature-aware memory-model configuration.")
        if not 0.5 <= safety_quantile < 1.0:
            raise ValueError("Memory safety quantile must be in [0.5, 1).")
        self.cold_start_bytes = int(cold_start_bytes)
        self.headroom_bytes = int(headroom_bytes)
        self.safety_quantile = float(safety_quantile)
        self.min_observations = int(min_observations)
        self.parameter_count_feature = str(parameter_count_feature)
        self.bytes_per_parameter = int(bytes_per_parameter)
        self.gradient_copies = float(gradient_copies)
        self.optimizer_copies = float(optimizer_copies)
        self.buffer_bytes = int(buffer_bytes)
        self.censored_multiplier = float(censored_multiplier)

    def predict(self, action: AdaptiveAction, state: AdaptiveOptimizerState) -> PredictiveEstimate:
        """Return a high-quantile reservation for the concrete candidate features."""
        evidence = self._evidence(state)
        structural = self._structural_estimate(action.resource_features)
        if len(evidence) < self.min_observations:
            mean = float(max(self.cold_start_bytes, structural))
            estimate = PredictiveEstimate.normal(
                mean,
                0.0,
                quantile=self.safety_quantile,
                source=(
                    "structural_cold_start_without_activations"
                    if structural > self.cold_start_bytes
                    else "logical_budget_cold_start"
                ),
            )
            return PredictiveEstimate(
                estimate.mean,
                estimate.standard_deviation,
                estimate.upper + self.headroom_bytes,
                estimate.source,
            )

        feature_keys = sorted(
            {
                key
                for item in evidence
                for key in self._numeric_features(item.parameters, item.resource_features)
            }
            | set(self._numeric_features(action.parameters, action.resource_features))
        )
        train_x = np.asarray(
            [
                self._vector(item.parameters, item.resource_features, feature_keys)
                for item in evidence
            ],
            dtype=np.float64,
        )
        targets = np.asarray(
            [math.log(max(1.0, self._target(item))) for item in evidence],
            dtype=np.float64,
        )
        design = np.column_stack((np.ones(len(train_x)), train_x))
        ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-3
        ridge[0, 0] = 1e-9
        coefficients = np.linalg.pinv(design.T @ design + ridge) @ design.T @ targets
        coefficients[1:] = np.maximum(coefficients[1:], 0.0)
        candidate = np.asarray(
            self._vector(action.parameters, action.resource_features, feature_keys),
            dtype=np.float64,
        )
        log_mean = float(np.r_[1.0, candidate] @ coefficients)
        fitted = design @ coefficients
        residual = float(np.std(targets - fitted, ddof=1)) if len(targets) > 1 else 0.0
        minima = train_x.min(axis=0)
        maxima = train_x.max(axis=0)
        spans = np.maximum(maxima - minima, 0.25)
        outside = np.maximum(minima - candidate, 0.0) + np.maximum(candidate - maxima, 0.0)
        ood_distance = float(np.linalg.norm(outside / spans))
        log_deviation = max(0.05, residual, 0.20 * ood_distance)
        mean = max(float(math.exp(min(log_mean, 700.0))), float(structural))
        same_lower_bounds = [
            item.lower_bound_bytes
            for item in evidence
            if item.censored
            and item.lower_bound_bytes is not None
            and self._same_signature(item, action)
        ]
        if same_lower_bounds:
            mean = max(mean, max(same_lower_bounds) * self.censored_multiplier)
        deviation = mean * math.sqrt(max(0.0, math.exp(log_deviation**2) - 1.0))
        estimate = PredictiveEstimate.normal(
            mean,
            deviation,
            quantile=self.safety_quantile,
            source="feature_aware_censored_log_linear",
        )
        return PredictiveEstimate(
            estimate.mean,
            estimate.standard_deviation,
            estimate.upper + self.headroom_bytes,
            estimate.source,
        )

    def requires_probe(self, state: AdaptiveOptimizerState) -> bool:
        """Return whether the predictor still has cold-start evidence."""
        return len(self._evidence(state)) < self.min_observations

    def _structural_estimate(self, features: Mapping[str, object]) -> int:
        value = features.get(self.parameter_count_feature)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return 0
        copies = 1.0 + self.gradient_copies + self.optimizer_copies
        return int(value * self.bytes_per_parameter * copies + self.buffer_bytes)

    @staticmethod
    def _numeric_features(
        parameters: Mapping[str, object], resource_features: Mapping[str, object]
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        for prefix, values in (("parameter", parameters), ("resource", resource_features)):
            for key, value in values.items():
                if isinstance(value, bool):
                    output[f"{prefix}:{key}"] = float(value)
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    output[f"{prefix}:{key}"] = math.log1p(max(0.0, float(value)))
        return output

    @classmethod
    def _vector(
        cls,
        parameters: Mapping[str, object],
        resource_features: Mapping[str, object],
        keys: list[str],
    ) -> list[float]:
        values = cls._numeric_features(parameters, resource_features)
        return [values.get(key, 0.0) for key in keys]

    def _target(self, observation: AdaptiveMemoryObservation) -> float:
        if observation.censored:
            assert observation.lower_bound_bytes is not None
            return observation.lower_bound_bytes * self.censored_multiplier
        return float(observation.peak_bytes or observation.lower_bound_bytes or 0)

    @staticmethod
    def _same_signature(observation: AdaptiveMemoryObservation, action: AdaptiveAction) -> bool:
        return dict(observation.parameters) == dict(action.parameters) and dict(
            observation.resource_features
        ) == dict(action.resource_features)

    @staticmethod
    def _evidence(state: AdaptiveOptimizerState) -> list[AdaptiveMemoryObservation]:
        output: list[AdaptiveMemoryObservation] = list(state.memory_observations)
        completed = {action.action_id: action for action in state.completed_actions}
        for observation in state.observations:
            action = completed.get(observation.action_id)
            if observation.oom and observation.memory_limit_bytes is not None:
                output.append(
                    AdaptiveMemoryObservation(
                        observation.config_id,
                        observation.parameters,
                        action.resource_features if action is not None else {},
                        lower_bound_bytes=observation.memory_limit_bytes,
                        censored=True,
                        source="training_oom",
                    )
                )
            elif observation.peak_reserved_bytes > 0:
                output.append(
                    AdaptiveMemoryObservation(
                        observation.config_id,
                        observation.parameters,
                        action.resource_features if action is not None else {},
                        peak_bytes=observation.peak_reserved_bytes,
                        source="training",
                    )
                )
        unique: list[AdaptiveMemoryObservation] = []
        for item in output:
            if any(
                existing.config_id == item.config_id
                and dict(existing.parameters) == dict(item.parameters)
                and dict(existing.resource_features) == dict(item.resource_features)
                and existing.peak_bytes == item.peak_bytes
                and existing.lower_bound_bytes == item.lower_bound_bytes
                and existing.censored == item.censored
                for existing in unique
            ):
                continue
            unique.append(item)
        return unique
