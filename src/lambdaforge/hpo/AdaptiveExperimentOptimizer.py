"""High-level asynchronous adaptive experiment optimizer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch

from lambdaforge.experiments.ExecutionConfig import ExecutionConfig
from lambdaforge.experiments.ExecutionMode import ExecutionMode
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.experiments.retention.ArtifactRetentionManager import (
    ArtifactRetentionManager,
)
from lambdaforge.experiments.retention.ArtifactRetentionPolicy import (
    ArtifactRetentionPolicy,
)
from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveEventLog import AdaptiveEventLog
from lambdaforge.hpo.AdaptiveExperimentController import AdaptiveExperimentController
from lambdaforge.hpo.AdaptiveExperimentPlan import AdaptiveExperimentPlan
from lambdaforge.hpo.AdaptiveExperimentResult import AdaptiveExperimentResult
from lambdaforge.hpo.AdaptiveExperimentWorker import AdaptiveExperimentWorker
from lambdaforge.hpo.AdaptiveMemoryObservation import AdaptiveMemoryObservation
from lambdaforge.hpo.AdaptiveObservation import AdaptiveObservation
from lambdaforge.hpo.AdaptiveObservationReader import AdaptiveObservationReader
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveResource import AdaptiveResource
from lambdaforge.hpo.AdaptiveRunMaterializer import AdaptiveRunMaterializer
from lambdaforge.hpo.AdaptiveTrialStatus import AdaptiveTrialStatus
from lambdaforge.hpo.MemoryCapacity import MemoryCapacity
from lambdaforge.hpo.MemoryProbePolicy import MemoryProbePolicy
from lambdaforge.hpo.TorchMemoryPreflight import TorchMemoryPreflight
from lambdaforge.hpo.UtilityAwareScheduler import UtilityAwareScheduler
from lambdaforge.training.orchestration.TrainingJob import TrainingJob
from lambdaforge.training.orchestration.TrainingOrchestrator import TrainingOrchestrator


class AdaptiveExperimentOptimizer:
    """Drive action selection, isolated training and durable asynchronous feedback."""

    def __init__(self, config: ExperimentConfig | Mapping[str, Any]) -> None:
        self.experiment = (
            config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        )
        self.base = self.experiment.as_dict()
        self.config = AdaptiveOptimizerConfig.from_experiment(self.base)
        if not self.config.enabled:
            raise ValueError("AdaptiveExperimentOptimizer requires hpo.enabled: true.")
        fingerprint = self.config.study_fingerprint(self.base)
        self.study_dir = (
            self.experiment.suite_dir
            / ".lambdaforge"
            / "adaptive"
            / fingerprint.removeprefix("sha256:")[:16]
        )
        self.state_path = self.study_dir / "state.json"
        self.event_path = self.study_dir / "events.jsonl"
        self.summary_path = self.study_dir / "summary.json"
        self.event_log = AdaptiveEventLog(self.event_path)
        self.state = self._load_state(fingerprint)
        self.controller = AdaptiveExperimentController(
            self.config,
            self.state,
            state_path=self.state_path,
            event_log=self.event_log,
        )
        self.materializer = AdaptiveRunMaterializer()
        self.reader = AdaptiveObservationReader()
        self.runner = ExperimentRunner()
        self._configs: dict[str, dict[str, Any]] = {}
        self._actions: dict[str, AdaptiveAction] = {}
        self._reservations: dict[int, int] = {}
        self._requeue: list[AdaptiveAction] = []
        self.scheduler = UtilityAwareScheduler()
        self.probe_policy = MemoryProbePolicy(
            mode=self.config.memory_probe_mode,
            relative_uncertainty_threshold=self.config.memory_probe_relative_uncertainty,
            near_limit_fraction=self.config.memory_probe_near_limit_fraction,
            oom_probability_threshold=self.config.memory_probe_oom_probability,
        )

    def inspect(self) -> AdaptiveExperimentPlan:
        """Resolve an informative plan without writing state or probing CUDA."""
        execution = ExecutionConfig.from_mapping(self.base)
        return AdaptiveExperimentPlan(
            {
                "mode": "adaptive_hpo",
                "objective": {
                    "metric": self.config.metric,
                    "direction": self.config.direction,
                    "risk": self.config.risk_type,
                },
                "search_space": self.config.space.to_dict(),
                "initialization": {
                    "strategy": self.config.initialization_strategy,
                    "trials": self.config.initial_trials,
                },
                "search": {
                    "strategy": self.config.search_strategy,
                    "optional_provider": "botorch"
                    if self.config.search_strategy == "bayesian"
                    else None,
                },
                "fidelity": {
                    "strategy": self.config.fidelity_strategy,
                    "unit": self.config.fidelity_unit,
                    "min": self.config.min_budget,
                    "max": self.config.max_budget,
                    "step": self.config.budget_step,
                },
                "seeds": {
                    "strategy": self.config.seed_strategy,
                    "search": list(self.config.search_seeds),
                    "confirmation": list(self.config.confirmation_seeds),
                },
                "budget": {
                    "max_actions": self.config.max_actions,
                    "max_total_epochs": self.config.max_total_epochs,
                    "max_gpu_seconds": self.config.max_gpu_seconds,
                },
                "resources": {
                    "gpus": execution.gpus,
                    "max_concurrency": self.config.max_concurrency,
                    "logical_memory_budget_bytes": self.config.memory_per_job_bytes,
                    "explicit_capacities": list(self.config.device_capacities),
                    "unknown_capacity_policy": self.config.unknown_memory_policy,
                    "resource_features": dict(self.config.resource_feature_paths),
                    "probe_policy": self.config.memory_probe_mode,
                },
                "study_dir": str(self.study_dir),
            }
        )

    def run(self) -> AdaptiveExperimentResult:
        """Run or resume the study until its search and confirmation phases finish."""
        execution = ExecutionConfig.from_mapping(self.base)
        slots, capacities = self._resources(execution)
        manager = ArtifactRetentionManager()
        policy = ArtifactRetentionPolicy.from_config(self.experiment)
        lock = manager.activity_lock(self.experiment, policy, shared=False)
        lock.acquire()
        try:
            manager.invalidate_receipt(self.experiment)
            self._recover_pending(execution)
            orchestrator = TrainingOrchestrator(
                grace_seconds=execution.grace_seconds,
                cpu_threads_per_job=execution.cpu_threads_per_job,
                cpu_interop_threads_per_job=execution.cpu_interop_threads_per_job,
                cpu_cores_per_job=execution.cpu_cores_per_job,
            )

            def supply(slot_index: int, slot: tuple[int, ...] | None) -> TrainingJob | None:
                device_index = slot[0] if slot else None
                capacity = capacities.get(device_index, MemoryCapacity.unknown())
                active = sum(
                    value
                    for index, value in self._reservations.items()
                    if self._slot_device(slots[index]) == device_index
                )
                available = capacity.remaining(active)
                while True:
                    action = self._next_action(available)
                    if action is None:
                        return None
                    run_config = self.materializer.materialize(self.base, action, self.config)
                    if execution.mode is not ExecutionMode.SEQUENTIAL:
                        run_config = execution.patch_run(run_config)
                    probe = self._probe_action(
                        action, run_config, device_index, available, execution
                    )
                    if probe is not None and probe.get("status") == "oom":
                        self.controller.observe(
                            AdaptiveObservation(
                                action.action_id,
                                action.config_id,
                                action.parameters,
                                action.seed,
                                action.current_budget,
                                None,
                                (),
                                AdaptiveTrialStatus.OOM_GPU,
                                peak_allocated_bytes=int(probe.get("peak_allocated_bytes", 0)),
                                peak_reserved_bytes=int(probe.get("peak_reserved_bytes", 0)),
                                oom=True,
                                error=str(probe.get("error", "memory preflight OOM")),
                                memory_limit_bytes=int(
                                    probe.get("lower_bound_bytes", self.config.memory_per_job_bytes)
                                ),
                            )
                        )
                        continue
                    self._configs[action.action_id] = run_config
                    self._actions[action.action_id] = action
                    self._reservations[slot_index] = action.memory_reservation_bytes
                    memory_budget = (
                        self.config.memory_per_job_bytes if self.config.allocator_cap else 0
                    )
                    return TrainingJob(
                        action.action_id,
                        AdaptiveExperimentWorker(run_config, memory_budget),
                    )

            def finished(name: str, exit_code: int | None, slot_index: int) -> None:
                action = self._actions.pop(name)
                run_config = self._configs.pop(name)
                self._reservations.pop(slot_index, None)
                run_dir = self.runner.experiment_run_dir(run_config)
                observation = self.reader.read(action, run_dir, self.config, exit_code=exit_code)
                self.controller.observe(observation)

            orchestrator.run_dynamic(slots, supply, finished)
            if self.state.phase.value != "finished":
                self.state.save(self.state_path)
                raise RuntimeError(
                    "Adaptive optimization cannot admit or propose another action. Review "
                    "events.jsonl, memory capacity/budget and search-space exhaustion."
                )
        finally:
            lock.release()
        summary = self.controller.summary()
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return AdaptiveExperimentResult(self.study_dir, self.state_path, self.event_path, summary)

    def _load_state(self, fingerprint: str) -> AdaptiveOptimizerState:
        if not self.state_path.exists():
            return AdaptiveOptimizerState(
                study_fingerprint=fingerprint,
                controller_seed=self.config.controller_seed,
            )
        state = AdaptiveOptimizerState.load(self.state_path)
        if state.study_fingerprint != fingerprint:
            raise ValueError("Adaptive optimizer state does not match this study configuration.")
        return state

    def _recover_pending(self, execution: ExecutionConfig) -> None:
        for action in tuple(self.state.pending_actions.values()):
            config = self.materializer.materialize(self.base, action, self.config)
            if execution.mode is not ExecutionMode.SEQUENTIAL:
                config = execution.patch_run(config)
            run_dir = self.runner.experiment_run_dir(config)
            if (run_dir / "result.json").exists():
                observation = self.reader.read(action, run_dir, self.config, exit_code=0)
                self.controller.recover_observation(observation)
            else:
                self._requeue.append(action)

    def _next_action(self, available_bytes: MemoryCapacity) -> AdaptiveAction | None:
        assignments = self.scheduler.pack(
            self._requeue,
            [
                AdaptiveResource(
                    "free-slot",
                    None,
                    memory_capacity_bytes=available_bytes.bytes,
                    memory_capacity_kind=available_bytes.kind,
                )
            ],
        )
        if assignments:
            selected = assignments[0].action
            self._requeue.remove(selected)
            self.event_log.append("ACTION_REQUEUED", selected.to_dict())
            return selected
        if self._requeue:
            return None
        return self.controller.select_next(available_bytes=available_bytes)

    def _probe_action(
        self,
        action: AdaptiveAction,
        run_config: dict[str, Any],
        device: int | None,
        capacity: MemoryCapacity,
        execution: ExecutionConfig,
    ) -> dict[str, Any] | None:
        if self.config.memory_probe_mode == "never":
            return None
        if device is None:
            raise ValueError("CUDA memory probes require an adaptive GPU resource.")
        if not self.probe_policy.should_probe(
            action,
            self.state,
            self.controller.memory_model,
            capacity,
        ):
            self.event_log.append(
                "MEMORY_PREFLIGHT_SKIPPED",
                {"action_id": action.action_id, "reason": "predictor_confident"},
            )
            return None
        assert isinstance(self.config.memory_probe_spec, Mapping)
        results = TorchMemoryPreflight().run(
            dict(self.config.memory_probe_spec),
            devices=[device],
            budget_bytes=self.config.memory_per_job_bytes,
            output_dir=self.study_dir / "preflight" / action.action_id,
            grace_seconds=execution.grace_seconds,
            configuration=run_config,
            resource_context={
                "capacity": capacity.kind.value,
                "capacity_bytes": capacity.bytes,
                "memory_budget_bytes": self.config.memory_per_job_bytes,
                "resource_features": dict(action.resource_features),
            },
            label=action.action_id,
        )
        result = results[device]
        self.event_log.append(
            "MEMORY_PREFLIGHT_COMPLETED",
            {"action_id": action.action_id, "device": device, "result": result},
        )
        if result.get("status") == "ok":
            peak = int(result.get("peak_reserved_bytes", 0))
            if peak > 0:
                self.controller.observe_memory(
                    AdaptiveMemoryObservation(
                        action.config_id,
                        action.parameters,
                        action.resource_features,
                        peak_bytes=peak,
                        source="candidate_preflight",
                    )
                )
        return result

    def _resources(
        self, execution: ExecutionConfig
    ) -> tuple[list[list[int]], dict[int | None, MemoryCapacity]]:
        gpus = list(execution.gpus or ())
        if not gpus:
            return (
                [[] for _ in range(self.config.max_concurrency)],
                {None: MemoryCapacity.unbounded()},
            )
        slots = [[gpus[index % len(gpus)]] for index in range(self.config.max_concurrency)]
        capacities: dict[int | None, MemoryCapacity] = {}
        for position, device in enumerate(gpus):
            if position < len(self.config.device_capacities):
                capacities[device] = MemoryCapacity.known(self.config.device_capacities[position])
            elif torch.cuda.is_available():
                capacities[device] = MemoryCapacity.known(
                    int(torch.cuda.get_device_properties(device).total_memory)
                )
            elif (
                self.config.unknown_memory_policy == "declared_budget"
                and self.config.memory_per_job_bytes > 0
            ):
                capacities[device] = MemoryCapacity.known(self.config.memory_per_job_bytes)
            else:
                capacities[device] = MemoryCapacity.unknown()
        return slots, capacities

    @staticmethod
    def _slot_device(slot: list[int]) -> int | None:
        return slot[0] if slot else None
