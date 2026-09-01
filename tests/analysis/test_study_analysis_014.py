from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.StudyAnalysis import StudyAnalysis
from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility


def _run(
    value: float,
    *,
    seed: int = 1,
    state: str = "succeeded",
    phase: str = "search",
    step: int = 10,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "state": state,
        "phase": phase,
        "termination_type": "performance_pruned" if state == "pruned" else "completed",
        "fidelity": {"current": step, "target": step, "maximum": 10},
        "objective_observation": {
            "metric": "score",
            "mode": "max",
            "current": value - 0.01,
            "current_step": step,
            "best": value,
            "best_step": max(1, step - 1),
        },
        "duration_seconds": 10 + seed,
        "gpu_index": 0,
    }


def _study(rows: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    return {
        "execution_id": "execution-synthetic",
        "finished": True,
        "objective": {"metric": "score", "mode": "max"},
        "candidates": [
            {
                "trial": index,
                "parameters": parameters,
                "runs": [_run(value, seed=seed) for seed in (1, 2, 3)],
            }
            for index, (parameters, value) in enumerate(rows, 1)
        ],
    }


def test_dominant_parameter_has_larger_global_importance() -> None:
    rows = [
        ({"x1": x / 5, "x2": y / 5}, x / 5 + (y % 2) * 0.001) for x in range(6) for y in range(6)
    ]
    analysis = StudyAnalysis.compute(
        _study(rows),
        authored_space={"x1": {"range": [0, 1]}, "x2": {"range": [0, 1]}},
    )
    importance = analysis["parameter_importance"]
    assert importance["x1"]["importance"] > importance["x2"]["importance"]
    assert analysis["surrogate"]["observations"] == 36


def test_interaction_and_boundary_diagnostics_respond_to_synthetic_truth() -> None:
    interaction_rows = [
        (
            {"x1": x / 4, "x2": y / 4},
            (x / 4 - 0.5) * (y / 4 - 0.5),
        )
        for x in range(5)
        for y in range(5)
    ]
    interaction = StudyAnalysis.compute(
        _study(interaction_rows),
        authored_space={"x1": {"range": [0, 1]}, "x2": {"range": [0, 1]}},
    )
    assert interaction["interactions"]["pairs"][0]["importance"] > 0

    boundary = StudyAnalysis.compute(
        _study([({"x": index / 19}, index / 19) for index in range(20)]),
        authored_space={"x": {"range": [0, 1]}},
    )
    assert boundary["boundaries"]["x"]["status"] in {"possible", "likely"}
    assert boundary["boundaries"]["x"]["boundary"] == "upper"

    interior = StudyAnalysis.compute(
        _study([({"x": index / 19}, -(((index / 19) - 0.5) ** 2)) for index in range(20)]),
        authored_space={"x": {"range": [0, 1]}},
    )
    assert interior["boundaries"]["x"]["status"] == "none"


def test_pruned_evidence_is_visible_but_never_a_final_seed() -> None:
    source = {
        "execution_id": "execution-pruned",
        "finished": True,
        "objective": {"metric": "score", "mode": "max"},
        "candidates": [
            {
                "trial": 1,
                "parameters": {"width": 64},
                "runs": [_run(0.65, state="pruned", step=5)],
            }
        ],
    }
    analysis = StudyAnalysis.compute(source)
    candidate = analysis["candidates"][0]
    run = candidate["runs"][0]
    assert run["best_observed_objective"] == pytest.approx(0.65)
    assert run["final_objective"] is None
    assert run["censored"] is True
    assert candidate["mean"] is None
    assert candidate["best_observed_objective"] == pytest.approx(0.65)


def test_confirmation_is_separate_and_analysis_is_deterministic() -> None:
    source = {
        "execution_id": "execution-confirmation",
        "finished": True,
        "objective": {"metric": "score", "mode": "max"},
        "candidates": [
            {
                "trial": 1,
                "parameters": {"width": 64},
                "runs": [
                    *[_run(value, seed=seed) for seed, value in enumerate((0.8, 0.82, 0.81), 1)],
                    *[
                        _run(value, seed=seed, phase="confirmation")
                        for seed, value in enumerate((0.7, 0.71, 0.69), 101)
                    ],
                ],
            },
            {
                "trial": 2,
                "parameters": {"width": 128},
                "runs": [
                    _run(value, seed=seed) for seed, value in enumerate((0.75, 0.76, 0.74), 1)
                ],
            },
        ],
    }
    first = StudyAnalysis.compute(source)
    second = StudyAnalysis.compute(source)
    assert first["winner"]["confirmation_status"] == "regressed"
    assert first["candidates"][0]["screening"]["mean"] == pytest.approx(0.81)
    assert first["candidates"][0]["confirmation"]["mean"] == pytest.approx(0.7)
    assert first["source"]["evidence_fingerprint"] == second["source"]["evidence_fingerprint"]
    assert first["candidates"] == second["candidates"]
    assert first["findings"] == second["findings"]


def test_categorical_and_conditional_parameters_keep_inactivity_explicit() -> None:
    rows = []
    for index in range(24):
        family = "attention" if index % 2 else "linear"
        parameters: dict[str, Any] = {"family": family}
        if family == "attention":
            parameters["heads"] = 2 if index % 4 == 1 else 8
        value = 0.8 if family == "attention" else 0.4
        value += 0.05 if parameters.get("heads") == 8 else 0.0
        rows.append((parameters, value))
    analysis = StudyAnalysis.compute(
        _study(rows),
        authored_space={
            "family": {"values": ["linear", "attention"]},
            "heads": {"values": [2, 8], "when": {"family": "attention"}},
        },
    )
    family = analysis["top_region_importance"]["family"]
    heads = analysis["coverage"]["marginal"]["heads"]
    assert family["importance"] > 0
    assert heads["inactive_fraction"] == pytest.approx(0.5)
    assert heads["full_fidelity_count"] == 12
    assert heads["observed_range"] == [2.0, 8.0]


def test_composite_objective_uses_objective_utility_and_resource_pareto() -> None:
    objective = ObjectiveUtility.normalize(
        {
            "metrics": {
                "quality": {"mode": "max", "weight": 0.75, "range": [0, 1]},
                "latency": {"mode": "min", "weight": 0.25, "range": [0, 10]},
            }
        }
    )
    evaluator = ObjectiveUtility(objective)
    candidates = []
    for trial, (quality, latency, duration) in enumerate(((0.8, 4.0, 80), (0.76, 3.0, 20)), 1):
        utility = evaluator.evaluate({"quality": quality, "latency": latency})
        assert utility is not None
        run = _run(float(utility["value"]), seed=trial)
        run["duration_seconds"] = duration
        run["objective_observation"].update(
            {
                "metric": "__lambdaforge_utility__",
                "components": {
                    "quality": {"raw": quality},
                    "latency": {"raw": latency},
                },
            }
        )
        candidates.append({"trial": trial, "parameters": {"width": trial}, "runs": [run]})
    analysis = StudyAnalysis.compute(
        {
            "execution_id": "composite",
            "finished": True,
            "objective": objective,
            "candidates": candidates,
        }
    )
    assert analysis["objective"] == objective
    assert analysis["pareto"]["objective_components"] == ["quality", "latency"]
    assert analysis["resources"]["pareto"]["gpu_seconds"]


def test_persistence_is_idempotent_for_the_same_evidence(tmp_path: Path) -> None:
    source = _study([({"x": index / 7}, index / 7) for index in range(8)])
    path = tmp_path / "analysis.json"
    first = StudyAnalysis.persist(source, path)
    original_bytes = path.read_bytes()
    second = StudyAnalysis.persist(source, path)
    assert first == second
    assert path.read_bytes() == original_bytes
    assert second["analysis_version"] == 1


def test_optional_report_fails_only_at_export_when_plotly_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original_import = builtins.__import__

    def without_plotly(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("plotly"):
            raise ImportError("optional provider absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)
    with pytest.raises(RuntimeError, match=r"lambdaforge\[analysis-report\]"):
        write_html({"source": {"status": "final"}}, tmp_path / "report.html")
    assert not (tmp_path / "report.html").exists()
