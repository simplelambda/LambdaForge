from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lambdaforge.analysis.Effects import reference_set
from lambdaforge.analysis.Report import (
    write_html,
    write_metric_html,
    write_parameter_html,
    write_resource_html,
)
from lambdaforge.analysis.SearchSpace import build_space, valid_point
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
    assert second["analysis_version"] == 3


def test_empirical_seed_stability_requires_repeated_leading_seeds() -> None:
    source = {
        "execution_id": "single-seed",
        "finished": True,
        "objective": {"metric": "score", "mode": "max"},
        "candidates": [
            {"trial": trial, "parameters": {"x": trial}, "runs": [_run(value, seed=1)]}
            for trial, value in ((1, 0.8), (2, 0.7), (3, 0.6))
        ],
    }
    insufficient = StudyAnalysis.compute(source)["seed_analysis"]
    assert insufficient["status"] == "insufficient"
    assert insufficient["reason"] == "insufficient_repeated_seed_evidence"
    assert insufficient["winner_rank_one_fraction"] is None
    assert insufficient["empirical"] is None

    for candidate in source["candidates"]:
        candidate["runs"].append(
            _run(float(candidate["runs"][0]["objective_observation"]["best"]) - 0.01, seed=2)
        )
    available = StudyAnalysis.compute(source)["seed_analysis"]
    assert available["status"] == "available"
    assert 0 <= available["winner_rank_one_fraction"] <= 1
    assert available["empirical"]["method"] == "candidate-seed-bootstrap"


def test_minimization_reverses_response_optimum_and_preserves_mode() -> None:
    rows = [({"x": index / 19}, index / 19) for index in range(20)]
    maximum = StudyAnalysis.compute(_study(rows), authored_space={"x": {"range": [0, 1]}})
    minimum_source = _study(rows)
    minimum_source["objective"] = {"metric": "score", "mode": "min"}
    minimum = StudyAnalysis.compute(minimum_source, authored_space={"x": {"range": [0, 1]}})
    max_point = maximum["response_curves"]["x"]["best_supported_point"]
    min_point = minimum["response_curves"]["x"]["best_supported_point"]
    assert max_point["x"] > min_point["x"]
    assert maximum["winner"]["screening_winner"]["trial"] == 20
    assert minimum["winner"]["screening_winner"]["trial"] == 1


def test_conditional_reference_domain_preserves_two_dependency_levels() -> None:
    authored = {
        "family": {"values": ["linear", "attention"]},
        "heads": {"values": [2, 8], "when": {"family": "attention"}},
        "dropout": {"range": [0.0, 0.5], "when": {"heads": 8}},
    }
    candidates = [
        {"parameters": {"family": "linear"}},
        {"parameters": {"family": "attention", "heads": 2}},
        {"parameters": {"family": "attention", "heads": 8, "dropout": 0.2}},
    ]
    space = build_space(candidates, authored)
    points = reference_set(space, count=200, fingerprint="conditional-domain")
    assert points
    assert all(valid_point(point, space) for point in points)
    assert all("heads" not in point for point in points if point["family"] == "linear")
    assert all("dropout" not in point for point in points if point.get("heads") != 8)
    assert all("dropout" in point for point in points if point.get("heads") == 8)


def test_discrete_numeric_values_are_not_expanded_into_a_continuous_domain() -> None:
    authored = {"width": {"values": [64, 128, 256]}}
    candidates = [{"parameters": {"width": 64}}, {"parameters": {"width": 256}}]
    space = build_space(candidates, authored)
    points = reference_set(space, count=100, fingerprint="numeric-values")
    assert {point["width"] for point in points} <= {64, 128, 256}
    assert valid_point({"width": 128}, space)
    assert not valid_point({"width": 100}, space)


def test_surrogate_metadata_matches_leave_one_candidate_out_computation() -> None:
    analysis = StudyAnalysis.compute(
        _study([({"x": index / 29}, (index % 7) / 7) for index in range(30)]),
        authored_space={"x": {"range": [0, 1]}},
    )
    surrogate = analysis["surrogate"]
    assert surrogate["validation_method"] == "leave-one-candidate-out"
    assert surrogate["validation"] == "leave-one-candidate-out"
    assert surrogate["fold_count"] == surrogate["observations"] == 30
    if surrogate["quality"] == "poor":
        assert analysis["parameter_importance"]["x"]["reliability"] == "low"


def test_resource_cost_separates_intrinsic_run_cost_from_controller_spend() -> None:
    source = _study([({"x": 0}, 0.8), ({"x": 1}, 0.7)])
    first = source["candidates"][0]
    for run in first["runs"]:
        run["duration_seconds"] = 10
    analysis = StudyAnalysis.compute(source)
    candidate = analysis["candidates"][0]["resource_cost"]
    assert candidate["intrinsic_per_comparable_run"]["gpu_seconds"] == 10
    assert candidate["controller_spend"]["gpu_seconds"] == 30
    assert candidate["controller_spend"]["seeds_purchased"] == 3


def test_confirmation_reuses_persisted_equivalence_margin() -> None:
    source = {
        "execution_id": "margin",
        "finished": True,
        "objective": {"metric": "score", "mode": "max"},
        "controller": {
            "recent": [
                {
                    "action": "INITIALIZE",
                    "policy": {
                        "proposal_pool_size": 100,
                        "seed_racing": {"equivalence_margin": 0.15},
                    },
                }
            ]
        },
        "candidates": [
            {
                "trial": 1,
                "parameters": {"x": 1},
                "runs": [
                    *[_run(value, seed=seed) for seed, value in enumerate((0.8, 0.81, 0.79), 1)],
                    *[
                        _run(value, seed=seed, phase="confirmation")
                        for seed, value in enumerate((0.7, 0.71, 0.69), 11)
                    ],
                ],
            }
        ],
    }
    analysis = StudyAnalysis.compute(source)
    assert analysis["winner"]["confirmation_status"] == "confirmed_equivalent"
    assert analysis["winner"]["confirmation"]["equivalence_margin"] == 0.15
    resolution = analysis["candidate_pool_resolution"]
    assert resolution["kind"] == "observed-candidate-resolution"
    assert resolution["proposal_pool_size"] == 100
    assert resolution["proposal_pool_points_persisted"] is False


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


def test_interactive_curve_and_parameter_exports_use_optional_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_objects = ModuleType("plotly.graph_objects")

    class Figure:
        def __init__(self, data: Any = None) -> None:
            self.data = list(data or ())
            self.layout: dict[str, Any] = {}

        def add_trace(self, trace: Any) -> None:
            self.data.append(trace)

        def update_layout(self, **kwargs: Any) -> None:
            self.layout.update(kwargs)

    def trace(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    graph_objects.Figure = Figure  # type: ignore[attr-defined]
    graph_objects.Scatter = trace  # type: ignore[attr-defined]
    graph_objects.Heatmap = trace  # type: ignore[attr-defined]
    graph_objects.Surface = trace  # type: ignore[attr-defined]
    plotly = ModuleType("plotly")
    plotly.graph_objects = graph_objects  # type: ignore[attr-defined]
    offline = ModuleType("plotly.offline")
    offline.plot = lambda figure, **_kwargs: (  # type: ignore[attr-defined]
        f"<div data-traces='{len(figure.data)}'>{figure.layout.get('title', '')}</div>"
    )
    monkeypatch.setitem(sys.modules, "plotly", plotly)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", graph_objects)
    monkeypatch.setitem(sys.modules, "plotly.offline", offline)

    curves = write_metric_html(
        {"val_score": [{"step": 1, "value": 0.5}, {"step": 2, "value": 0.7}]},
        ["val_score"],
        tmp_path / "curves.html",
        display_names={"val_score": "Validation score"},
    )
    parameter = write_parameter_html(
        {
            "response_curves": {
                "width": {
                    "points": [
                        {
                            "x": 64,
                            "predicted_objective": 0.6,
                            "uncertainty": 0.05,
                            "support_count": 2,
                        },
                        {
                            "x": 128,
                            "predicted_objective": 0.7,
                            "uncertainty": 0.04,
                            "support_count": 3,
                        },
                    ]
                }
            },
            "interactions": {
                "surfaces": {
                    "width::depth": {
                        "x": [64, 128],
                        "y": [1, 2],
                        "cells": [
                            {"x": 64, "y": 1, "predicted_objective": 0.5},
                            {"x": 128, "y": 1, "predicted_objective": 0.6},
                            {"x": 64, "y": 2, "predicted_objective": 0.65},
                            {"x": 128, "y": 2, "predicted_objective": 0.7},
                        ],
                    }
                }
            },
        },
        "width",
        tmp_path / "width.html",
    )
    resources = write_resource_html(
        "gpu-cluster",
        {
            "total_cpu": [(-60.0, 40.0), (0.0, 55.0)],
            "mine_cpu": [(-60.0, 10.0), (0.0, 20.0)],
            "total_ram": [(-60.0, 30.0), (0.0, 32.0)],
            "total_gpu": [(-60.0, 70.0), (0.0, 80.0)],
        },
        tmp_path / "resources.html",
    )

    curve_html = curves.read_text(encoding="utf-8")
    assert "LambdaForge learning curves" in curve_html
    assert "Learning curves" in curve_html
    exported = parameter.read_text(encoding="utf-8")
    assert "Response: width" in exported
    assert "Pairwise response: width::depth" in exported
    assert "3D pairwise response: width::depth" in exported
    resource_html = resources.read_text(encoding="utf-8")
    assert "LambdaForge resources · gpu-cluster" in resource_html
    assert "GPU memory" in resource_html
