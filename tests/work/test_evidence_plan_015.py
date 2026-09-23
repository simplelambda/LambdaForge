from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.hpo.ScientificDesign import ScientificQuestionAnalyzer, SeedNoiseEstimate
from lambdaforge.work import runner
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.runner import WorkRunner
from lambdaforge.work.study import StudyTelemetry, study_run_key


def _sweep_mapping(*, values: int = 8, seeds: int = 10) -> dict[str, Any]:
    return {
        "name": "fixed-evidence",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": list(range(seeds)),
        "sweep": {
            "space": {"quality": [float(value) for value in range(values)]},
            "reference": {"quality": 0.0},
        },
        "objective": {"metric": "score", "mode": "max"},
        "execution": {"runs_per_gpu": "auto", "max_parallel": 2},
        "resources": {"cpu": 2},
    }


def test_wisdom_shaped_sweep_has_eighty_required_balanced_identities(tmp_path: Path) -> None:
    config = WorkConfig.from_mapping(_sweep_mapping(), source=tmp_path / "sweep.yaml")
    definition = config.levels[0].runs[0]
    design = definition.study_design

    assert design is not None
    assert design.kind == "sweep"
    assert definition.run_count == config.planned_runs == 80
    assert len(design.evidence.required) == 80
    expanded = WorkRunner._expanded(definition)
    assert {(trial, seed) for trial, _variant, seed in expanded} == {
        (trial, seed) for trial in range(1, 9) for seed in range(10)
    }
    assert [seed for _trial, _variant, seed in expanded[:8]] == [0] * 8
    assert [trial for trial, _variant, _seed in expanded[:8]] == list(range(1, 9))
    assert [trial for trial, _variant, _seed in expanded[8:16]] == [*range(2, 9), 1]


def test_fixed_and_legacy_exhaustive_use_the_common_resource_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[int, bool, int | None]] = []

    def fake_dispatch(specifications: Any, **kwargs: Any) -> tuple[Any, ...]:
        policy = kwargs["policy"]
        captured.append((len(specifications), policy.early_stopping, policy.max_parallel))
        return ()

    monkeypatch.delenv("LAMBDAFORGE_STUDY_PATH", raising=False)
    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", fake_dispatch)
    sweep = WorkConfig.from_mapping(_sweep_mapping(), source=tmp_path / "sweep.yaml")
    fixed = sweep.levels[0].runs[0]
    specifications = [
        {
            "definition": {
                "name": fixed.name,
                "work_class": fixed.work_class,
                "resources": fixed.resources.to_dict(),
                "objective": dict(fixed.objective or {}),
                "execution_policy": fixed.execution_policy.to_dict(),
                "study_design": fixed.study_design.to_dict() if fixed.study_design else None,
            },
            "parameters": dict(variant),
            "trial_parameters": dict(variant),
            "trial_index": trial,
            "seed": seed,
            "execution_id": "execution-fixed",
            "execution_dir": tmp_path,
            "source": tmp_path / "sweep.yaml",
        }
        for trial, variant, seed in WorkRunner._expanded(fixed)
    ]
    runner._execute_fixed_evidence_group(specifications)

    legacy = WorkConfig.from_mapping(
        {
            "name": "legacy",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [1, 2],
            "search": {"strategy": "exhaustive", "quality": [1.0, 2.0]},
            "execution": {"max_parallel": 3},
        },
        source=tmp_path / "legacy.yaml",
    ).levels[0].runs[0]
    assert legacy.study_design is not None and legacy.study_design.kind == "sweep"
    assert legacy.search_policy is None
    legacy_specifications = [
        {
            "definition": {
                "name": legacy.name,
                "work_class": legacy.work_class,
                "resources": legacy.resources.to_dict(),
                "objective": dict(legacy.objective or {}),
                "execution_policy": legacy.execution_policy.to_dict(),
                "study_design": legacy.study_design.to_dict(),
            },
            "parameters": dict(variant),
            "trial_parameters": dict(variant),
            "trial_index": trial,
            "seed": seed,
            "execution_id": "execution-legacy",
            "execution_dir": tmp_path,
            "source": tmp_path / "legacy.yaml",
        }
        for trial, variant, seed in WorkRunner._expanded(legacy)
    ]
    runner._execute_fixed_evidence_group(legacy_specifications)
    assert captured == [(80, False, 2), (4, False, 3)]


def test_structured_and_legacy_adaptive_yaml_normalize_identically(tmp_path: Path) -> None:
    shared = {
        "name": "adaptive",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "seeds": [1, 2, 3, 4],
        "objective": {"metric": "score", "mode": "max", "practical_margin": 0.01},
        "resources": {"cpu": 4},
    }
    modern = WorkConfig.from_mapping(
        {
            **shared,
            "search": {
                "budget": {"candidates": 3, "runs": 12},
                "space": {"quality": [0.1, 0.5, 0.9]},
                "replication": {"minimum": 4, "confirmation": []},
                "pruning": {"enabled": False},
            },
            "execution": {"runs_per_gpu": "auto", "max_parallel": 4},
        },
        source=tmp_path / "modern.yaml",
    ).levels[0].runs[0]
    legacy = WorkConfig.from_mapping(
        {
            **shared,
            "search": {
                "trials": 3,
                "max_runs": 12,
                "min_seeds": 4,
                "confirmation_seeds": [],
                "early_stopping": {"enabled": False},
                "runs_per_gpu": "auto",
                "max_parallel": 4,
                "quality": [0.1, 0.5, 0.9],
            },
        },
        source=tmp_path / "legacy.yaml",
    ).levels[0].runs[0]

    assert modern.search_policy is not None and legacy.search_policy is not None
    assert modern.search_policy.to_dict() == legacy.search_policy.to_dict()
    assert modern.execution_policy == legacy.execution_policy
    assert modern.search_policy.scientific_margin == pytest.approx(0.01)

    alias_only = WorkConfig.from_mapping(
        {
            **{key: value for key, value in shared.items() if key != "objective"},
            "objective": {"metric": "score", "mode": "max"},
            "search": {
                "trials": 1,
                "confirmation_seeds": [],
                "seed_racing": {"equivalence_margin": 0.02},
                "quality": [0.5],
            },
        },
        source=tmp_path / "legacy-margin.yaml",
    ).levels[0].runs[0]
    assert alias_only.objective is not None
    assert alias_only.objective["practical_margin"] == pytest.approx(0.02)


def test_candidate_budget_does_not_cancel_required_minimum_replication(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "minimum-replication",
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [11, 12, 13, 14],
            "search": {
                "budget": {"candidates": 3, "runs": 12},
                "space": {"quality": [0.1, 0.5, 0.9]},
                "replication": {"minimum": 4, "confirmation": []},
                "pruning": {"enabled": False},
            },
            "execution": {"max_parallel": 3},
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 3},
        },
        source=tmp_path / "minimum-replication.yaml",
    )

    result = WorkRunner().run(config)

    assert result.status == "succeeded"
    assert len(result.runs) == 12
    assert {
        (int(run.trial["index"]), run.seed)
        for run in result.runs
        if run.trial is not None
    } == {(trial, seed) for trial in range(1, 4) for seed in (11, 12, 13, 14)}
    assert result.summary["evidence"]["required_missing"] == 0


def test_structured_space_parameter_names_do_not_collide_with_policy_names(
    tmp_path: Path,
) -> None:
    config = WorkConfig.from_mapping(
        {
            "name": "collision-free",
            "run": "tests.work_cases.PolicyNamedParameterWork",
            "search": {
                "budget": {"candidates": 2},
                "space": {
                    "trials": [16],
                    "max_parallel": [2, 4],
                },
                "replication": {"minimum": 1, "confirmation": []},
            },
            "objective": {"metric": "score", "mode": "max"},
        },
        source=tmp_path / "collision-free.yaml",
    )
    definition = config.levels[0].runs[0]

    assert definition.search_policy is not None
    assert definition.search_policy.candidate_budget == 2
    assert {tuple(sorted(value.items())) for value in definition.variants} == {
        (("max_parallel", 2), ("trials", 16)),
        (("max_parallel", 4), ("trials", 16)),
    }


def test_sweep_grid_and_budget_validation_are_strict(tmp_path: Path) -> None:
    mapping = _sweep_mapping(values=2, seeds=2)
    mapping["sweep"] = {
        "space": {"quality": {"range": [1e-5, 1e-3], "points": 3, "scale": "log"}}
    }
    definition = WorkConfig.from_mapping(mapping, source=tmp_path / "grid.yaml").levels[0].runs[0]
    assert [variant["quality"] for variant in definition.variants] == pytest.approx(
        [1e-5, 1e-4, 1e-3]
    )

    invalid = _sweep_mapping(values=8, seeds=10)
    invalid["execution"] = {"max_runs": 40}
    with pytest.raises(ValueError, match="requires 80 Runs"):
        WorkConfig.from_mapping(invalid, source=tmp_path / "invalid.yaml")

    with pytest.raises(ValueError, match="adaptive candidate/seed controls"):
        WorkConfig.from_mapping(
            {
                **_sweep_mapping(values=2, seeds=2),
                "sweep": {"space": {"quality": [1, 2]}, "pruning": False},
            },
            source=tmp_path / "invalid-control.yaml",
        )


def test_terminal_telemetry_reconciles_queued_required_evidence(tmp_path: Path) -> None:
    config = WorkConfig.from_mapping(
        _sweep_mapping(values=2, seeds=2), source=tmp_path / "sweep.yaml"
    )
    definition = config.levels[0].runs[0]
    assert definition.study_design is not None
    requirements = {
        (value.candidate, value.seed): value.to_dict()
        for value in definition.study_design.evidence.required
    }
    specifications = [
        {
            "trial_index": trial,
            "seed": seed,
            "trial_parameters": dict(variant),
            "evidence_requirement": requirements[(trial, seed)],
        }
        for trial, variant, seed in WorkRunner._expanded(definition)
    ]
    telemetry = StudyTelemetry(tmp_path / "study")
    telemetry.initialize(
        name="fixed",
        execution_id="execution-fixed",
        strategy="sweep",
        objective={"metric": "score", "mode": "max"},
        specifications=specifications,
        design=definition.study_design.to_dict(),
    )
    telemetry.schedule(specifications)
    first = specifications[0]
    telemetry._write_run(
        study_run_key(first),
        {"state": "succeeded", "termination_type": "completed"},
    )
    telemetry.candidate_states(
        active=(1, 2),
        ranked=(1, 2),
        finished=True,
        finish_reason="time-budget-exhausted",
    )
    snapshot = telemetry.refresh()

    assert snapshot["finished"] is True
    assert snapshot["status"] == "incomplete"
    assert snapshot["design_status"] == "incomplete"
    assert snapshot["scientific_status"] == "unresolved"
    assert snapshot["finish_reason"] == "time-budget-exhausted"
    assert snapshot["required_completed"] == 1
    assert snapshot["required_missing"] == 3
    assert snapshot["counts"]["queued_runs"] == 0


def _analysis_sweep(
    values: dict[int, dict[int, float]], *, margin: float | None = None
) -> dict[str, Any]:
    requirements = [
        {
            "key": f"candidate-{trial}:seed-{seed}:search:fidelity-none",
            "candidate": trial,
            "seed": seed,
            "phase": "search",
            "required": True,
        }
        for trial in values
        for seed in (1, 2, 3, 4)
    ]
    objective: dict[str, Any] = {"metric": "score", "mode": "max"}
    if margin is not None:
        objective["practical_margin"] = margin
    return StudyAnalysis.compute(
        {
            "execution_id": "execution-sweep",
            "finished": True,
            "objective": objective,
            "design": {
                "type": "sweep",
                "space": {"profile": {"values": ["a", "b", "c"]}},
                "reference": {"profile": "a"},
                "evidence": {"requirements": requirements},
            },
            "candidates": [
                {
                    "trial": trial,
                    "parameters": {"profile": chr(96 + trial)},
                    "runs": [
                        {
                            "seed": seed,
                            "state": "succeeded",
                            "phase": "search",
                            "final_objective": score,
                            "best_observed_objective": score,
                            "termination_type": "completed",
                        }
                        for seed, score in by_seed.items()
                    ],
                }
                for trial, by_seed in values.items()
            ],
        },
        status="final",
    )


def test_sweep_analysis_uses_shared_seed_blocks_without_imputation() -> None:
    analysis = _analysis_sweep(
        {
            1: {1: 0.9, 2: 0.88, 3: 0.91, 4: 0.89},
            2: {1: 0.5, 2: 0.52, 3: 0.49, 4: 0.51},
            3: {1: 0.6, 2: 0.61, 4: 0.59},
        }
    )
    sweep = analysis["sweep_analysis"]

    assert analysis["scientific_status"] == "resolved"
    assert sweep["exact_conclusion"]["kind"] == "PREFERRED"
    assert sweep["exact_conclusion"]["trials"] == [1]
    comparison = next(value for value in sweep["comparisons"] if value["left_trial"] == 3)
    assert comparison["paired_seeds"] == [1, 2, 4]
    assert comparison["paired_support"] == 3
    assert 3 in comparison["missing_seed_cells"]
    assert sweep["missing_cells_are_imputed"] is False
    assert sweep["evidence_matrix"]["seeds"] == [1, 2, 3, 4]
    third_row = next(
        row for row in sweep["evidence_matrix"]["rows"] if row["trial"] == 3
    )
    assert next(cell for cell in third_row["cells"] if cell["seed"] == 3)["state"] == (
        "missing"
    )
    assert analysis["summary"]["observed_candidate_count"] == 3
    assert analysis["summary"]["evidence_complete_candidate_count"] == 2


def test_complete_noisy_sweep_can_be_unresolved_or_practically_equivalent() -> None:
    noisy = _analysis_sweep(
        {
            1: {1: 0.51, 2: 0.49, 3: 0.51, 4: 0.49},
            2: {1: 0.49, 2: 0.51, 3: 0.49, 4: 0.51},
        }
    )
    assert noisy["design_status"] == "complete"
    assert noisy["scientific_status"] == "unresolved"
    assert noisy["sweep_analysis"]["exact_conclusion"]["kind"] == "NO_CLEAR_PREFERENCE"

    equivalent = _analysis_sweep(
        {
            1: {1: 0.51, 2: 0.50, 3: 0.52, 4: 0.49},
            2: {1: 0.50, 2: 0.51, 3: 0.50, 4: 0.50},
        },
        margin=0.05,
    )
    conclusion = equivalent["sweep_analysis"]["exact_conclusion"]
    assert conclusion["kind"] == "PRACTICALLY_EQUIVALENT"
    assert conclusion["confidence"] == pytest.approx(1.0)


def test_shared_seed_realizations_never_substitute_a_different_seed() -> None:
    outcomes = {
        1: {1: 11.0, 2: 12.0, 3: 13.0},
        2: {1: 21.0, 2: 22.0, 3: 23.0},
        3: {1: 31.0, 2: 32.0},
    }
    realizations = ScientificQuestionAnalyzer._realizations(
        {trial: {"profile": trial} for trial in outcomes},
        outcomes,
        {trial: sum(values.values()) / len(values) for trial, values in outcomes.items()},
        noise=SeedNoiseEstimate(None, {}, 0, 0, 0),
        fallback_sigma=1.0,
        count=100,
        rng=random.Random(9),
    )

    assert realizations
    for realization in realizations:
        assert len({int(value) % 10 for value in realization.values()}) <= 1


def test_exact_conclusion_does_not_turn_argmax_or_missing_margin_into_a_claim() -> None:
    unclear = ScientificQuestionAnalyzer._exact_conclusion(
        {'PREFERRED:"a"': 0.45, 'PREFERRED:"b"': 0.40, "UNRESOLVED": 0.15},
        levels=("a", "b"),
        practical_margin=None,
    )
    equivalence_without_margin = ScientificQuestionAnalyzer._exact_conclusion(
        {"PRACTICALLY_EQUIVALENT": 0.9, 'PREFERRED:"a"': 0.1},
        levels=("a", "b"),
        practical_margin=None,
    )

    assert unclear.kind == "NO_CLEAR_PREFERENCE"
    assert equivalence_without_margin.kind == "NO_CLEAR_PREFERENCE"
