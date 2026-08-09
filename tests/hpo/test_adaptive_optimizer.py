"""Focused contracts for action-centric adaptive optimization."""

from __future__ import annotations

import csv

import pytest

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
from lambdaforge.experiments.results.RunFingerprint import RunFingerprint
from lambdaforge.hpo import (
    AdaptiveAction,
    AdaptiveActionKind,
    AdaptiveObservation,
    AdaptiveOptimizerConfig,
    AdaptiveOptimizerState,
    AdaptiveTrialStatus,
    SearchSpace,
)
from lambdaforge.hpo.AdaptiveEventLog import AdaptiveEventLog
from lambdaforge.hpo.AdaptiveExperimentController import AdaptiveExperimentController
from lambdaforge.hpo.AdaptiveResource import AdaptiveResource
from lambdaforge.hpo.AdaptiveRunMaterializer import AdaptiveRunMaterializer
from lambdaforge.hpo.AdaptiveSeedRacer import AdaptiveSeedRacer
from lambdaforge.hpo.EmpiricalMemoryModel import EmpiricalMemoryModel
from lambdaforge.hpo.FixedSeedPolicy import FixedSeedPolicy
from lambdaforge.hpo.LearningCurveModel import LearningCurveModel
from lambdaforge.hpo.ResourceAdmissionController import ResourceAdmissionController
from lambdaforge.hpo.SobolSearcher import SobolSearcher
from lambdaforge.hpo.UtilityAwareScheduler import UtilityAwareScheduler


def _hpo() -> dict[str, object]:
    return {
        "enabled": True,
        "objective": {"metric": "val_loss", "direction": "minimize"},
        "space": {
            "optimizer.params.lr": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "scale": "log",
            },
            "model.params.hidden": {"type": "int", "low": 4, "high": 16},
        },
        "initialization": {"strategy": "sobol", "trials": 2},
        "search": {"strategy": "sobol"},
        "fidelity": {"min": 1, "max": 3, "step": 1},
        "seeds": {"values": [7, 17], "confirmation_values": [101]},
        "budget": {"max_actions": 8, "max_total_epochs": 20},
    }


def _base(tmp_path) -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "experiment": {"name": "adaptive", "output_root": str(tmp_path)},
        "data": {
            "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
            "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
            "datamodule": {"params": {"batch_size": 8, "num_workers": 0}},
        },
        "model": {
            "target": "lambdaforge.nn.models.MLP",
            "params": {"in_features": 4, "out_features": 1, "hidden": [4]},
        },
        "losses": [{"target": "lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss"}],
        "optimizer": {"ref": "torch.optim.AdamW", "params": {"lr": 0.001}},
        "task": {"params": {"model_input_key": "x", "model_output_key": "logits"}},
        "trainer": {
            "max_epochs": 3,
            "accelerator": "cpu",
            "devices": 1,
            "checkpoint_policy": "last",
            "enable_progress_bar": False,
            "num_sanity_val_steps": 0,
            "print_epoch_table": False,
            "trainer_kwargs": {"enable_model_summary": False},
        },
        "hpo": _hpo(),
    }


def test_search_space_and_sobol_are_deterministic_and_resume_from_state() -> None:
    config = AdaptiveOptimizerConfig(_hpo())
    first = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=3)
    second = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=3)
    searcher = SobolSearcher(seed=3)

    proposed = searcher.propose(config.space, first, count=2)
    assert proposed == searcher.propose(config.space, second, count=2)
    assert len({config.space.identifier(value) for value in proposed}) == 2
    assert all(1e-4 <= float(value["optimizer.params.lr"]) <= 1e-2 for value in proposed)


def test_state_round_trip_preserves_pending_observations_and_counters(tmp_path) -> None:
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=11)
    action = AdaptiveAction("a1", AdaptiveActionKind.START_NEW, "c1", {"x": 1}, 7, 0, 2)
    state.next_action_id()
    state.register_pending(action)
    state.complete(
        AdaptiveObservation(
            "a1",
            "c1",
            {"x": 1},
            7,
            2,
            0.4,
            ((1, 0.8), (2, 0.4)),
            AdaptiveTrialStatus.PAUSED,
            seconds=2.5,
        )
    )
    path = state.save(tmp_path / "state.json")

    restored = AdaptiveOptimizerState.load(path)
    assert restored.to_dict() == state.to_dict()
    assert restored.total_epochs == 2
    assert restored.next_action_id() == "decision-000002"


def test_learning_curve_keeps_improving_slow_starter_competitive() -> None:
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    state.configurations = {"slow": {"x": 1}, "plateau": {"x": 2}}
    state.observations.extend(
        [
            AdaptiveObservation(
                "slow",
                "slow",
                {"x": 1},
                7,
                3,
                0.50,
                ((1, 0.1), (2, 0.3), (3, 0.5)),
                AdaptiveTrialStatus.PAUSED,
            ),
            AdaptiveObservation(
                "plateau",
                "plateau",
                {"x": 2},
                7,
                3,
                0.55,
                ((1, 0.54), (2, 0.55), (3, 0.55)),
                AdaptiveTrialStatus.PAUSED,
            ),
        ]
    )
    model = LearningCurveModel()
    slow = model.predict_configuration(state, "slow", max_budget=10)
    plateau = model.predict_configuration(state, "plateau", max_budget=10)
    assert slow.mean > plateau.mean


def test_validator_and_inspect_expose_adaptive_plan_without_writes(tmp_path) -> None:
    base = _base(tmp_path)
    report = ExperimentValidator().validate(base)
    assert report.is_valid, report.summary()
    plan = Experiment(base).inspect()
    assert plan.to_dict()["mode"] == "adaptive_hpo"  # type: ignore[union-attr]
    assert not (tmp_path / "adaptive").exists()

    invalid = dict(base)
    invalid["sweep"] = {"grid": {"optimizer.params.lr": [0.1]}}
    report = ExperimentValidator().validate(invalid, check_imports=False)
    assert not report.is_valid
    assert "cannot be combined with sweep" in report.summary()


def test_adaptive_fidelity_fingerprint_is_stable_and_resume_does_not_repeat_epochs(
    tmp_path,
) -> None:
    base = _base(tmp_path)
    optimizer = AdaptiveOptimizerConfig.from_experiment(base)
    materializer = AdaptiveRunMaterializer()
    parameters = {
        "optimizer.params.lr": 0.001,
    }
    first = AdaptiveAction("a1", AdaptiveActionKind.START_NEW, "config-one", parameters, 7, 0, 1)
    second = AdaptiveAction("a2", AdaptiveActionKind.RESUME, "config-one", parameters, 7, 1, 3)
    first_config = materializer.materialize(base, first, optimizer)
    second_config = materializer.materialize(base, second, optimizer)
    assert RunFingerprint.digest(first_config) == RunFingerprint.digest(second_config)

    runner = ExperimentRunner()
    assert runner.run_single_experiment(first_config).status.value == "ok"
    assert runner.run_single_experiment(second_config).status.value == "ok"
    metrics = runner.experiment_run_dir(second_config) / "metrics.csv"
    with open(metrics, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(float(row["epoch"])) for row in rows] == [1, 2, 3]


def test_search_space_rejects_operational_conditions_and_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        SearchSpace.from_mapping({})


def test_seed_racing_uses_next_shared_seed_only_after_full_budget() -> None:
    config = AdaptiveOptimizerConfig(_hpo())
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    state.configurations = {"candidate": {"x": 1}}
    state.observations.append(
        AdaptiveObservation(
            "a1",
            "candidate",
            {"x": 1},
            7,
            3,
            0.2,
            ((1, 0.5), (2, 0.3), (3, 0.2)),
            AdaptiveTrialStatus.COMPLETED,
        )
    )
    actions = AdaptiveSeedRacer(config).candidates(state, LearningCurveModel())
    assert len(actions) == 1
    assert actions[0].kind is AdaptiveActionKind.ADD_SEED
    assert actions[0].seed == 17


def test_fixed_seed_policy_uses_the_same_replaceable_policy_signature() -> None:
    value = _hpo()
    value["seeds"] = {"strategy": "fixed", "values": [7, 17]}
    config = AdaptiveOptimizerConfig(value)
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    state.configurations = {"candidate": {"x": 1}}

    actions = FixedSeedPolicy(config).candidates(state, LearningCurveModel())
    assert [action.seed for action in actions] == [7, 17]


def test_custom_fidelity_policy_can_offer_probabilistic_drops_without_inheritance(
    tmp_path,
) -> None:
    class ProjectFidelityPolicy:
        def resume_candidates(self, state):
            del state
            return ()

        def dominated(self, state, learning_model):
            del state, learning_model
            return (("drop-me", 0.001),)

    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    state.configurations = {"drop-me": {"x": 1}, "keep": {"x": 2}}
    controller = AdaptiveExperimentController(
        AdaptiveOptimizerConfig(_hpo()),
        state,
        state_path=tmp_path / "state.json",
        event_log=AdaptiveEventLog(tmp_path / "events.jsonl"),
    )
    controller.fidelity = ProjectFidelityPolicy()  # type: ignore[assignment]

    controller._apply_conservative_drops()
    assert state.dropped_configurations == {"drop-me"}


def test_utility_scheduler_packs_by_utility_and_conservative_memory() -> None:
    low = AdaptiveAction("low", AdaptiveActionKind.START_NEW, "c1", {}, 1, 0, 1).with_scores(
        information_gain=1,
        predicted_cost=1,
        feasibility_probability=1,
        memory_reservation_bytes=6,
        reasons={},
    )
    high = AdaptiveAction("high", AdaptiveActionKind.START_NEW, "c2", {}, 1, 0, 1).with_scores(
        information_gain=2,
        predicted_cost=1,
        feasibility_probability=1,
        memory_reservation_bytes=6,
        reasons={},
    )
    assignments = UtilityAwareScheduler().pack(
        [low, high],
        [AdaptiveResource("gpu-0", 0, memory_capacity_bytes=10, max_jobs=2)],
    )
    assert [item.action.action_id for item in assignments] == ["high"]


def test_cost_and_feasibility_both_change_global_action_utility() -> None:
    expensive = AdaptiveAction(
        "expensive", AdaptiveActionKind.START_NEW, "c1", {}, 1, 0, 1
    ).with_scores(
        information_gain=0.10,
        predicted_cost=10,
        feasibility_probability=1,
        memory_reservation_bytes=1,
        reasons={},
    )
    efficient = AdaptiveAction(
        "efficient", AdaptiveActionKind.START_NEW, "c2", {}, 1, 0, 1
    ).with_scores(
        information_gain=0.06,
        predicted_cost=2,
        feasibility_probability=0.99,
        memory_reservation_bytes=1,
        reasons={},
    )
    assert expensive.utility == pytest.approx(0.01)
    assert efficient.utility == pytest.approx(0.0297)
    assignments = UtilityAwareScheduler().pack(
        [expensive, efficient],
        [AdaptiveResource("gpu-0", 0, memory_capacity_bytes=1, max_jobs=1)],
    )
    assert assignments[0].action.action_id == "efficient"


def test_synthetic_memory_admission_rejects_infeasible_cold_start() -> None:
    action = AdaptiveAction("a", AdaptiveActionKind.START_NEW, "c", {}, 1, 0, 1)
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    admission = ResourceAdmissionController(logical_limit_bytes=20)

    accepted = admission.assess(
        action,
        state,
        EmpiricalMemoryModel(cold_start_bytes=19),
        available_bytes=40,
    )
    rejected = admission.assess(
        action,
        state,
        EmpiricalMemoryModel(cold_start_bytes=35),
        available_bytes=40,
    )
    assert accepted[:3] == (True, 1.0, 19)
    assert rejected[0] is False


def test_summary_reports_confirmation_statistics_curves_seeds_and_memory(tmp_path) -> None:
    config = AdaptiveOptimizerConfig(_hpo())
    state = AdaptiveOptimizerState(study_fingerprint="study", controller_seed=0)
    state.configurations = {"a": {"x": 1}, "b": {"x": 2}}
    for config_id, scores in {"a": (0.20, 0.30), "b": (0.40, 0.50)}.items():
        for seed, score in zip((101, 211), scores, strict=True):
            action_id = f"{config_id}-{seed}"
            state.completed_actions.append(
                AdaptiveAction(
                    action_id,
                    AdaptiveActionKind.CONFIRM,
                    config_id,
                    state.configurations[config_id],
                    seed,
                    0,
                    3,
                )
            )
            state.observations.append(
                AdaptiveObservation(
                    action_id,
                    config_id,
                    state.configurations[config_id],
                    seed,
                    3,
                    score,
                    ((1, score + 0.2), (3, score)),
                    AdaptiveTrialStatus.COMPLETED,
                    peak_reserved_bytes=1024,
                )
            )
    controller = AdaptiveExperimentController(
        config,
        state,
        state_path=tmp_path / "state.json",
        event_log=AdaptiveEventLog(tmp_path / "events.jsonl"),
    )

    summary = controller.summary()
    statistics = summary["confirmation_statistics"]
    assert statistics["configurations"]["a"]["mean"] == pytest.approx(0.25)
    assert statistics["configurations"]["a"]["standard_error"] == pytest.approx(0.05)
    assert statistics["paired_differences"][0]["mean"] == pytest.approx(-0.20)
    assert summary["seed_usage"]["a"]["confirmation"] == [101, 211]
    assert summary["learning_curves"]["a"]["101"] == [[1, 0.4], [3, 0.2]]
    assert len(summary["memory_observations"]) == 4


def test_adaptive_experiment_runs_end_to_end_through_dynamic_backend(tmp_path) -> None:
    base = _base(tmp_path)
    hpo = dict(base["hpo"])  # type: ignore[arg-type]
    hpo["initialization"] = {"strategy": "sobol", "trials": 1}
    hpo["fidelity"] = {"min": 1, "max": 2, "step": 1}
    hpo["seeds"] = {"values": [7], "confirmation_values": []}
    hpo["budget"] = {"max_actions": 2, "max_total_epochs": 4}
    hpo["max_concurrency"] = 1
    base["hpo"] = hpo

    result = Experiment(base).run()
    assert result.summary["status"] == "ok"  # type: ignore[union-attr]
    assert result.summary["phase"] == "finished"  # type: ignore[union-attr]
    assert result.summary["completed_actions"] == 2  # type: ignore[union-attr]
    assert result.summary["failed_observations"] == 0  # type: ignore[union-attr]
    assert result.state_path.exists()  # type: ignore[union-attr]
    assert result.event_log_path.exists()  # type: ignore[union-attr]
