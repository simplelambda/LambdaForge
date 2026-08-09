"""Persistent action-centric adaptive optimization controller."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveActionSelector import AdaptiveActionSelector
from lambdaforge.hpo.AdaptiveEventLog import AdaptiveEventLog
from lambdaforge.hpo.AdaptiveFidelityPolicy import AdaptiveFidelityPolicy
from lambdaforge.hpo.AdaptiveMemoryObservation import AdaptiveMemoryObservation
from lambdaforge.hpo.AdaptiveObservation import AdaptiveObservation
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptivePhase import AdaptivePhase
from lambdaforge.hpo.AdaptiveSeedRacer import AdaptiveSeedRacer
from lambdaforge.hpo.BoTorchSearcher import BoTorchSearcher
from lambdaforge.hpo.EmpiricalCostModel import EmpiricalCostModel
from lambdaforge.hpo.FeatureAwareMemoryModel import FeatureAwareMemoryModel
from lambdaforge.hpo.FixedBudgetFidelityPolicy import FixedBudgetFidelityPolicy
from lambdaforge.hpo.FixedSeedPolicy import FixedSeedPolicy
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.RandomAdaptiveSearcher import RandomAdaptiveSearcher
from lambdaforge.hpo.ResourceAdmissionController import ResourceAdmissionController
from lambdaforge.hpo.SobolSearcher import SobolSearcher
from lambdaforge.hpo.SuccessiveHalvingFidelityPolicy import (
    SuccessiveHalvingFidelityPolicy,
)


class AdaptiveExperimentController:
    """Choose the next feasible new/resume/seed/confirmation action from durable history."""

    def __init__(
        self,
        config: AdaptiveOptimizerConfig,
        state: AdaptiveOptimizerState,
        *,
        state_path: str | Path,
        event_log: AdaptiveEventLog,
    ) -> None:
        self.config = config
        self.state = state
        self.state_path = Path(state_path)
        self.event_log = event_log
        self.learning_model = LearningCurveModel(exploration_weight=config.exploration_weight)
        self.cost_model = EmpiricalCostModel()
        self.memory_model = FeatureAwareMemoryModel(
            cold_start_bytes=config.memory_per_job_bytes,
            headroom_bytes=config.memory_headroom_bytes,
            safety_quantile=config.memory_safety_quantile,
            min_observations=config.memory_min_observations,
            parameter_count_feature=config.memory_parameter_count_feature,
            bytes_per_parameter=config.memory_bytes_per_parameter,
            gradient_copies=config.memory_gradient_copies,
            optimizer_copies=config.memory_optimizer_copies,
            buffer_bytes=config.memory_buffer_bytes,
        )
        self.admission = ResourceAdmissionController(
            logical_limit_bytes=config.memory_per_job_bytes
        )
        self.selector = AdaptiveActionSelector(
            direction=config.direction,
            max_budget=config.max_budget,
            exploration_weight=config.exploration_weight,
            risk_type=config.risk_type,
            risk_lambda=config.risk_lambda,
        )
        self.fidelity = self._fidelity_policy()
        self.seed_policy = (
            AdaptiveSeedRacer(config)
            if config.seed_strategy == "adaptive_racing"
            else FixedSeedPolicy(config)
        )
        self.sobol = SobolSearcher(seed=config.controller_seed)
        self.random = RandomAdaptiveSearcher(seed=config.controller_seed)
        self.bayesian = BoTorchSearcher(
            seed=config.controller_seed,
            direction=config.direction,
            acquisition=(
                "knowledge_gradient"
                if "knowledge_gradient" in config.acquisition_strategy
                else "expected_improvement"
            ),
            raw_samples=config.candidate_pool_size,
            max_budget=config.max_budget,
            min_budget=config.min_budget,
            budget_step=config.budget_step,
            refresh_interval=config.surrogate_refresh_interval,
        )
        self.custom_searcher: Any | None = None
        self._apply_component_overrides()

    def select_next(
        self, *, available_bytes: int | MemoryCapacity | None = None
    ) -> AdaptiveAction | None:
        """Select and persist one action; callers may request again for another free resource."""
        self._apply_conservative_drops()
        if (
            self._search_budget_exhausted()
            and self.state.phase is AdaptivePhase.SEARCH
            and not self.state.pending_actions
        ):
            self._enter_confirmation()
        candidates = self._candidate_actions()
        if (
            not candidates
            and self.state.phase is AdaptivePhase.SEARCH
            and not self.state.pending_actions
        ):
            self._enter_confirmation()
            candidates = self._confirmation_candidates()
        if not candidates:
            if self.state.phase is AdaptivePhase.CONFIRMATION and not self.state.pending_actions:
                self.state.phase = AdaptivePhase.FINISHED
                self.event_log.append("HPO_FINISHED", self.summary())
                self.state.save(self.state_path)
            return None
        ranked = self.selector.rank(
            candidates,
            self.state,
            learning_model=self.learning_model,
            cost_model=self.cost_model,
            memory_model=self.memory_model,
            admission=self.admission,
            available_bytes=available_bytes,
        )
        if self.config.max_gpu_seconds is not None:
            committed_seconds = self.state.total_gpu_seconds + sum(
                action.predicted_cost for action in self.state.pending_actions.values()
            )
            ranked = tuple(
                action
                for action in ranked
                if committed_seconds + action.predicted_cost <= self.config.max_gpu_seconds
            )
        alternatives = [action.to_dict() for action in ranked[:10]]
        if not ranked:
            self.event_log.append(
                "ACTION_REJECTED",
                {"reason": "budget_or_resource_admission", "candidates": len(candidates)},
            )
            return None
        selected = ranked[0]
        self.state.next_action_id()
        self.state.register_pending(selected)
        self.state.save(self.state_path)
        self.event_log.append(
            "DECISION_SELECTED",
            {
                "decision_index": self.state.decision_index,
                "selected": selected.to_dict(),
                "alternatives": alternatives,
            },
        )
        event = {
            AdaptiveActionKind.START_NEW: "TRIAL_STARTED",
            AdaptiveActionKind.RESUME: "TRIAL_RESUMED",
            AdaptiveActionKind.ADD_SEED: "SEED_ADDED",
            AdaptiveActionKind.CONFIRM: "CONFIRMATION_STARTED",
        }.get(selected.kind, "ACTION_SELECTED")
        self.event_log.append(event, selected.to_dict())
        return selected

    def observe(self, observation: AdaptiveObservation) -> None:
        """Incorporate one asynchronous result and atomically checkpoint controller knowledge."""
        action = self.state.complete(observation)
        event = {
            "oom_gpu": "TRIAL_OOM",
            "paused": "TRIAL_PAUSED",
            "early_stopped": "TRIAL_EARLY_STOPPED",
            "failed": "TRIAL_FAILED",
            "cancelled": "TRIAL_CANCELLED",
        }.get(observation.status.value, "TRIAL_COMPLETED")
        self.event_log.append(
            event,
            {"action": action.to_dict(), "observation": observation.to_dict()},
        )
        if observation.oom:
            self.state.dropped_configurations.add(observation.config_id)
            self.event_log.append(
                "CONFIG_DROPPED",
                {"config_id": observation.config_id, "reason": "oom_gpu"},
            )
        if observation.oom and observation.memory_limit_bytes is not None:
            self.state.memory_observations.append(
                AdaptiveMemoryObservation(
                    observation.config_id,
                    observation.parameters,
                    action.resource_features,
                    lower_bound_bytes=observation.memory_limit_bytes,
                    censored=True,
                    source="training_oom",
                )
            )
        elif observation.peak_reserved_bytes > 0:
            self.state.memory_observations.append(
                AdaptiveMemoryObservation(
                    observation.config_id,
                    observation.parameters,
                    action.resource_features,
                    peak_bytes=observation.peak_reserved_bytes,
                    source="training",
                )
            )
        self.state.save(self.state_path)

    def observe_memory(self, observation: AdaptiveMemoryObservation) -> None:
        """Persist candidate preflight evidence without creating a scientific score."""
        self.state.memory_observations.append(observation)
        self.event_log.append("MEMORY_OBSERVATION", observation.to_dict())
        self.state.save(self.state_path)

    def recover_observation(self, observation: AdaptiveObservation) -> None:
        """Reconcile a completed persisted run left pending by controller interruption."""
        if observation.action_id in self.state.pending_actions:
            self.observe(observation)

    def summary(self) -> dict[str, Any]:
        """Return factual progress without inventing compute savings."""
        completed = [item for item in self.state.observations if item.score is not None]
        scored_configurations = {
            item.config_id for item in self.state.observations if item.score is not None
        }
        ranked = sorted(
            {
                config_id: self.learning_model.predict_configuration(
                    self.state, config_id, max_budget=self.config.max_budget
                )
                for config_id in scored_configurations
            }.items(),
            key=lambda item: self._objective_value(item[1].mean, item[1].standard_deviation),
            reverse=True,
        )
        status = (
            "running"
            if self.state.phase is not AdaptivePhase.FINISHED
            else "ok"
            if completed
            else "failed"
        )
        return {
            "status": status,
            "phase": self.state.phase.value,
            "best_configuration": ranked[0][0] if ranked else None,
            "best_posterior": ranked[0][1].to_dict() if ranked else None,
            "objective_risk": {
                "type": self.config.risk_type,
                "lambda": self.config.risk_lambda,
            },
            "configurations": len(self.state.configurations),
            "completed_actions": len(self.state.completed_actions),
            "pending_actions": len(self.state.pending_actions),
            "observations": len(completed),
            "total_epochs": self.state.total_epochs,
            "full_training_equivalents": self.state.total_epochs / self.config.max_budget,
            "total_gpu_seconds": self.state.total_gpu_seconds,
            "dropped_configurations": sorted(self.state.dropped_configurations),
            "oom_observations": sum(item.oom for item in self.state.observations),
            "failed_observations": sum(
                item.status.value == "failed" for item in self.state.observations
            ),
            "cancelled_observations": sum(
                item.status.value == "cancelled" for item in self.state.observations
            ),
            "confirmation_actions": sum(
                action.kind is AdaptiveActionKind.CONFIRM for action in self.state.completed_actions
            ),
            "seed_usage": self._seed_usage(),
            "confirmation_statistics": self._confirmation_statistics(),
            "learning_curves": self._learning_curves(),
            "memory_observations": self._memory_observations(),
            "fallback_count": self.state.fallback_count,
        }

    def _seed_usage(self) -> dict[str, dict[str, list[int]]]:
        confirmation_ids = {
            action.action_id
            for action in self.state.completed_actions
            if action.kind is AdaptiveActionKind.CONFIRM
        }
        output: dict[str, dict[str, set[int]]] = {}
        for observation in self.state.observations:
            if observation.score is None:
                continue
            phase = "confirmation" if observation.action_id in confirmation_ids else "search"
            output.setdefault(observation.config_id, {"search": set(), "confirmation": set()})[
                phase
            ].add(observation.seed)
        return {
            config_id: {phase: sorted(seeds) for phase, seeds in phases.items()}
            for config_id, phases in sorted(output.items())
        }

    def _confirmation_statistics(self) -> dict[str, Any]:
        confirmation_ids = {
            action.action_id
            for action in self.state.completed_actions
            if action.kind is AdaptiveActionKind.CONFIRM
        }
        by_config: dict[str, dict[int, float]] = {}
        for observation in self.state.observations:
            if observation.action_id in confirmation_ids and observation.score is not None:
                by_config.setdefault(observation.config_id, {})[observation.seed] = (
                    observation.score
                )
        configurations = {
            config_id: self._sample_statistics(scores)
            for config_id, scores in sorted(by_config.items())
        }
        paired: list[dict[str, Any]] = []
        config_ids = sorted(by_config)
        for left_index, left in enumerate(config_ids):
            for right in config_ids[left_index + 1 :]:
                shared = sorted(set(by_config[left]) & set(by_config[right]))
                differences = [by_config[left][seed] - by_config[right][seed] for seed in shared]
                if differences:
                    paired.append(
                        {
                            "left": left,
                            "right": right,
                            "seeds": shared,
                            "difference": "left_minus_right",
                            **self._statistics(differences),
                        }
                    )
        return {"configurations": configurations, "paired_differences": paired}

    def _sample_statistics(self, scores: dict[int, float]) -> dict[str, Any]:
        return {"seeds": sorted(scores), **self._statistics(list(scores.values()))}

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, Any]:
        count = len(values)
        mean = fmean(values)
        deviation = stdev(values) if count > 1 else None
        standard_error = deviation / math.sqrt(count) if deviation is not None else None
        interval = (
            [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
            if standard_error is not None
            else None
        )
        return {
            "count": count,
            "mean": mean,
            "sample_standard_deviation": deviation,
            "standard_error": standard_error,
            "normal_95_confidence_interval": interval,
        }

    def _learning_curves(self) -> dict[str, dict[str, list[list[float | int]]]]:
        output: dict[str, dict[str, dict[int, float]]] = {}
        for observation in self.state.observations:
            curve = output.setdefault(observation.config_id, {}).setdefault(
                str(observation.seed), {}
            )
            curve.update(observation.curve)
            if observation.score is not None:
                curve[observation.budget] = observation.score
        return {
            config_id: {
                seed: [[budget, score] for budget, score in sorted(curve.items())]
                for seed, curve in sorted(seeds.items(), key=lambda item: int(item[0]))
            }
            for config_id, seeds in sorted(output.items())
        }

    def _memory_observations(self) -> list[dict[str, Any]]:
        training = [
            {
                "config_id": observation.config_id,
                "seed": observation.seed,
                "budget": observation.budget,
                "peak_allocated_bytes": observation.peak_allocated_bytes,
                "peak_reserved_bytes": observation.peak_reserved_bytes,
                "oom": observation.oom,
            }
            for observation in self.state.observations
            if observation.peak_allocated_bytes > 0
            or observation.peak_reserved_bytes > 0
            or observation.oom
        ]
        return (
            [item.to_dict() for item in self.state.memory_observations]
            if self.state.memory_observations
            else training
        )

    def _candidate_actions(self) -> tuple[AdaptiveAction, ...]:
        if self.state.phase is AdaptivePhase.CONFIRMATION:
            return self._confirmation_candidates()
        if self.state.phase is AdaptivePhase.FINISHED:
            return ()
        if self._search_budget_exhausted():
            return ()
        output: list[AdaptiveAction] = []
        if len(self.state.configurations) < self.config.initial_trials:
            parameters = self._initial_searcher().propose(self.config.space, self.state, count=1)
            output.extend(self._new_actions(parameters))
            if output:
                return tuple(self._with_resource_features(action) for action in output)
        else:
            try:
                parameters = self._searcher().propose(self.config.space, self.state, count=1)
            except Exception as error:
                self.state.fallback_count += 1
                self.event_log.append(
                    "HPO_SURROGATE_FALLBACK",
                    {
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "fallback": "sobol",
                    },
                )
                parameters = self.sobol.propose(self.config.space, self.state, count=1)
            output.extend(self._new_actions(parameters))
        output.extend(self.fidelity.resume_candidates(self.state))
        output.extend(self.seed_policy.candidates(self.state, self.learning_model))
        pending_keys = {
            (action.config_id, action.seed) for action in self.state.pending_actions.values()
        }
        committed_epochs = self.state.total_epochs + sum(
            action.target_budget - action.current_budget
            for action in self.state.pending_actions.values()
        )
        remaining_epochs = self.config.max_total_epochs - committed_epochs
        return tuple(
            self._with_resource_features(action)
            for action in output
            if (action.config_id, action.seed) not in pending_keys
            and action.target_budget - action.current_budget <= remaining_epochs
        )

    def _with_resource_features(self, action: AdaptiveAction) -> AdaptiveAction:
        if action.resource_features:
            return action
        return replace(
            action,
            resource_features=self.config.resolve_resource_features(action.parameters),
        )

    def _new_actions(self, candidates: tuple[dict[str, object], ...]) -> tuple[AdaptiveAction, ...]:
        target = (
            self.config.max_budget
            if self.config.fidelity_strategy == "fixed"
            else self.config.min_budget
        )
        return tuple(
            AdaptiveAction(
                (
                    f"candidate-{self.state.decision_index + 1:06d}-new-"
                    f"{self.config.space.identifier(parameters)}"
                ),
                AdaptiveActionKind.START_NEW,
                self.config.space.identifier(parameters),
                parameters,
                self.config.search_seeds[0],
                0,
                target,
            )
            for parameters in candidates
        )

    def _confirmation_candidates(self) -> tuple[AdaptiveAction, ...]:
        if not self.config.confirmation_seeds:
            return ()
        scored = {
            item.config_id
            for item in self.state.observations
            if item.score is not None and item.config_id not in self.state.dropped_configurations
        }
        ranked = sorted(
            scored,
            key=self._prediction_objective,
            reverse=True,
        )[: self.config.confirmation_top_k]
        output: list[AdaptiveAction] = []
        pending = {
            (action.config_id, action.seed) for action in self.state.pending_actions.values()
        }
        for config_id in ranked:
            observed = {item.seed for item in self.state.observations_for(config_id)}
            for seed in self.config.confirmation_seeds:
                if seed not in observed and (config_id, seed) not in pending:
                    output.append(
                        AdaptiveAction(
                            (
                                f"candidate-{self.state.decision_index + 1:06d}-confirm-"
                                f"{config_id}-{seed}"
                            ),
                            AdaptiveActionKind.CONFIRM,
                            config_id,
                            self.state.configurations[config_id],
                            seed,
                            0,
                            self.config.max_budget,
                            AdaptivePhase.CONFIRMATION,
                        )
                    )
        committed_epochs = self.state.total_epochs + sum(
            action.target_budget - action.current_budget
            for action in self.state.pending_actions.values()
        )
        remaining_epochs = self.config.max_total_epochs - committed_epochs
        return tuple(
            self._with_resource_features(action)
            for action in output
            if action.target_budget - action.current_budget <= remaining_epochs
        )

    def _apply_conservative_drops(self) -> None:
        dominated = getattr(self.fidelity, "dominated", None)
        if dominated is None:
            return
        for config_id, probability in dominated(self.state, self.learning_model):
            self.state.dropped_configurations.add(config_id)
            self.event_log.append(
                "CONFIG_DROPPED",
                {"config_id": config_id, "competitive_probability": probability},
            )

    def _search_budget_exhausted(self) -> bool:
        return (
            len(self.state.completed_actions) + len(self.state.pending_actions)
            >= self.config.max_actions
            or self.state.total_epochs >= self.config.max_total_epochs
            or (
                self.config.max_gpu_seconds is not None
                and self.state.total_gpu_seconds >= self.config.max_gpu_seconds
            )
        )

    def _enter_confirmation(self) -> None:
        self.state.phase = AdaptivePhase.CONFIRMATION
        self.event_log.append("SEARCH_FINISHED", self.summary())
        self.state.save(self.state_path)

    def _initial_searcher(self) -> SobolSearcher | RandomAdaptiveSearcher:
        return self.sobol if self.config.initialization_strategy == "sobol" else self.random

    def _searcher(self) -> BoTorchSearcher | SobolSearcher | RandomAdaptiveSearcher:
        if self.custom_searcher is not None:
            return self.custom_searcher
        if self.config.search_strategy == "bayesian":
            return self.bayesian
        if self.config.search_strategy == "sobol":
            return self.sobol
        return self.random

    def _fidelity_policy(
        self,
    ) -> AdaptiveFidelityPolicy | FixedBudgetFidelityPolicy | SuccessiveHalvingFidelityPolicy:
        if self.config.fidelity_strategy == "fixed":
            return FixedBudgetFidelityPolicy(self.config)
        if self.config.fidelity_strategy == "successive_halving":
            return SuccessiveHalvingFidelityPolicy(self.config)
        return AdaptiveFidelityPolicy(self.config)

    def _objective_value(self, mean: float, standard_deviation: float) -> float:
        oriented = mean if self.config.direction == "maximize" else -mean
        if self.config.risk_type == "mean_minus_std":
            oriented -= self.config.risk_lambda * standard_deviation
        return oriented

    def _prediction_objective(self, config_id: str) -> float:
        prediction = self.learning_model.predict_configuration(
            self.state, config_id, max_budget=self.config.max_budget
        )
        return self._objective_value(prediction.mean, prediction.standard_deviation)

    def _apply_component_overrides(self) -> None:
        """Replace policy boundaries from trusted importable YAML specs when requested."""
        assignments = {
            "searcher": "custom_searcher",
            "fidelity_policy": "fidelity",
            "seed_policy": "seed_policy",
            "learning_curve_model": "learning_model",
            "cost_model": "cost_model",
            "memory_model": "memory_model",
            "admission_controller": "admission",
            "action_selector": "selector",
        }
        for configured, attribute in assignments.items():
            spec = self.config.components.get(configured)
            if spec is not None:
                setattr(self, attribute, ObjectFactory.build(spec))
