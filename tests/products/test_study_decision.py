"""Native scientific selections survive operational failures without fabricated conclusions."""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.analysis.Evidence import evidence_fingerprint, normalize_evidence
from lambdaforge.analysis.StudyAnalysis import ANALYSIS_VERSION, StudyAnalysis
from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.products import ProductBundle, ProductRegistry, build_study_decision


def source() -> dict[str, Any]:
    candidate = {
        "trial": 1,
        "parameters": {"width": 64},
        "value": 0.7,
        "runs": ["run-1"],
        "seeds": [4],
        "feasibility": {"feasible": True},
        "partially_censored": False,
        "search_uncertainty": {"samples": 1, "mean": 0.7, "standard_error": None},
        "confirmation_complete": False,
    }
    return {
        "execution_result_version": 1,
        "execution_id": "execution-decision",
        "name": "selection",
        "status": "failed",
        "scientific_fingerprint": "original-code",
        "summary": {
            "objective": {"metric": "score", "mode": "max"},
            "study_design": {"kind": "repeated", "space": {}, "evidence": {}},
            "candidates": [candidate],
            "best": {
                **candidate,
                "selection_score": 0.7,
                "selection_basis": "conservative-search-bound",
            },
        },
        "runs": [
            {
                "run_id": "run-1",
                "attempt_id": "attempt-0002",
                "attempt_number": 2,
                "work_class": "example.Training",
                "name": "training",
                "status": "succeeded",
                "seed": 4,
                "trial": {"index": 1, "parameters": {"width": 64}},
                "parameters": {"width": 64, "pooling": "max"},
                "metrics": {"score": 0.7},
                "run_dir": "/operational/attempt",
                "inputs": [
                    {
                        "name": "dataset",
                        "kind": "dataset",
                        "path": "/operational/dataset",
                        "content_id": "sha256:" + "a" * 64,
                    }
                ],
            },
            {
                "run_id": "run-auxiliary",
                "attempt_number": 1,
                "name": "visualization",
                "status": "failed",
                "seed": 2,
                "failure": {"type": "RuntimeError"},
            },
        ],
    }


def analysis(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "execution_id": value["execution_id"],
            "status": "final",
            "evidence_fingerprint": StudyAnalysis.evidence_identity(value),
        },
        # An analysis point-estimate must never replace the native selection.
        "winner": {"screening_winner": {"trial": 99}},
        "scientific_status": "partially_resolved",
        "scientific_understanding": {
            "unresolved_questions": [{"question": "width", "confidence": 0.4}],
            "parameter_questions": [{"name": "width", "conclusion_kind": "UNRESOLVED"}],
            "interaction_questions": [],
            "practical_optimal_region": {"source": "predictive", "confidence": 0.6},
            "evidence": {"confidence_semantics": "realization agreement"},
        },
    }


def decide(value: dict[str, Any], **extra: Any) -> Any:
    return build_study_decision(value, name="chosen", contract="example/decision:v1", **extra)


def test_decision_preserves_native_selection_not_failed_auxiliary_or_posthoc_leader() -> None:
    value = source()
    original = copy.deepcopy(value)
    product = decide(value, analysis=analysis(value))
    assert product.payload["preferred_candidate"] == 1
    assert product.payload["preferred_parameters"] == {"width": 64}
    assert product.payload["scientific_status"] == "partially_resolved"
    assert product.payload["unresolved_questions"][0]["confidence"] == 0.4
    assert product.payload["uncertainty"]["screening"]["standard_error"] is None
    assert product.payload["evidence"][0]["attempt_number"] == 2
    assert product.producer["source_status"] == "failed"
    assert "/operational" not in json.dumps(product.scientific_meaning)
    assert pickle.loads(pickle.dumps(product)) == product
    assert value == original


def test_missing_analysis_is_honestly_unresolved_not_a_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Decision inspection must never fit scientific models")

    monkeypatch.setattr(StudyAnalysis, "compute", forbidden)
    product = decide(source())
    assert product.payload["status"] == "selection_available"
    assert product.payload["scientific_status"] == "unresolved"
    assert product.payload["point_estimate_is_scientific_resolution"] is False
    assert product.payload["unresolved_questions"]


def test_incomplete_confirmation_cannot_publish_survivor_selection() -> None:
    value = source()
    value["summary"]["confirmation"] = {"status": "incomplete", "confirmation_incomplete": True}
    product = decide(value)
    assert product.payload["status"] == "selection_unavailable"
    assert product.payload["preferred_candidate"] is None
    assert product.payload["selection_score"] is None


@pytest.mark.parametrize("mutation", ["stale", "different-execution", "provisional", "version"])
def test_stale_or_invalid_analysis_refuses_publication(mutation: str) -> None:
    value = source()
    cached = analysis(value)
    if mutation == "stale":
        value["runs"][0]["metrics"]["score"] = 0.8
    elif mutation == "different-execution":
        cached["source"]["execution_id"] = "other"
    elif mutation == "provisional":
        cached["source"]["status"] = "provisional"
    else:
        cached["analysis_version"] = True
    with pytest.raises(ValueError, match="does not match"):
        decide(value, analysis=cached)


@pytest.mark.parametrize("mutation", ["pruned", "later-failure", "missing", "infeasible", "live"])
def test_invalid_native_evidence_is_not_reinterpreted_as_a_winner(mutation: str) -> None:
    value = source()
    if mutation == "pruned":
        value["runs"][0]["pruned"] = True
    elif mutation == "later-failure":
        value["runs"].append({**value["runs"][0], "status": "failed", "attempt_number": 3})
    elif mutation == "missing":
        value["runs"] = []
    elif mutation == "infeasible":
        value["summary"]["best"]["feasibility"] = {"feasible": False}
    else:
        value["status"] = "running"
    with pytest.raises(ValueError):
        decide(value)


def test_analysis_identity_extraction_preserves_historical_formula() -> None:
    value = source()
    objective = StudyAnalysis.objective(value, None)
    candidates, _runs = normalize_evidence(value, objective)
    expected = evidence_fingerprint(
        {
            "execution_id": value["execution_id"],
            "objective": objective,
            "candidates": candidates,
            "authored_space": {},
            "scientific_policy": {},
        }
    )
    assert StudyAnalysis.evidence_identity(value) == expected


def test_decision_identity_ignores_operational_config_but_not_science() -> None:
    value = source()
    first = decide(value)
    value["scientific_fingerprint"] = "code-refactoring-not-new-producer"
    value["runs"][0]["run_dir"] = "/different/location"
    value["runs"][0]["inputs"][0]["path"] = "/different/dataset"
    moved = decide(value)
    assert moved.scientific_id == first.scientific_id
    assert moved.producer != first.producer
    value["runs"][0]["parameters"]["pooling"] = "mean"
    assert decide(value).scientific_id != first.scientific_id


def test_native_decide_preview_export_delete_import_keeps_product(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = source()
    results = tmp_path / "results"
    execution = results / "selection" / value["execution_id"]
    execution.mkdir(parents=True)
    (execution / "result.json").write_text(json.dumps(value))
    (execution / "configuration.json").write_text(
        json.dumps(
            {
                "name": "selection",
                "objective": {"metric": "score", "mode": "max"},
            }
        )
    )
    root = tmp_path / "products"
    argv = [
        "products",
        "decide",
        value["execution_id"],
        "--name",
        "chosen",
        "--contract",
        "example/decision:v1",
        "--root",
        str(root),
        "--results-root",
        str(results),
        "--json",
    ]
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert CommandLineInterface.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["decision"]["preferred_candidate"] == 1
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
    assert CommandLineInterface.main([*argv, "--apply"]) == 0
    registry = ProductRegistry(root)
    ProductBundle.export(registry, "chosen", tmp_path / "bundle", apply=True)
    import shutil

    shutil.rmtree(execution)
    imported = ProductRegistry(tmp_path / "imported")
    ProductBundle.import_bundle(imported, tmp_path / "bundle", apply=True)
    assert (
        imported.resolve("chosen", contract="example/decision:v1").payload["preferred_candidate"]
        == 1
    )


def test_native_cpu_study_decision_uses_persisted_analysis_without_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lambdaforge.work import ResultStore, WorkConfig, WorkRunner

    yaml_path = tmp_path / "work.yaml"
    yaml_path.write_text(
        "name: decided\nrun: tests.work_cases.ScoredModelSnapshotWork\n"
        "seeds: [1, 2]\nobjective: {metric: score, mode: max}\n"
        "resources: {cpu: 1, memory: 128MiB}\n"
    )
    monkeypatch.chdir(tmp_path)
    execution = WorkRunner().run(WorkConfig.from_yaml(yaml_path))
    assert execution.status == "succeeded"
    store = ResultStore(execution.execution_dir.parent.parent)
    before = {
        str(path): path.read_bytes()
        for path in execution.execution_dir.rglob("*")
        if path.is_file()
    }

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Opening a StudyDecision must not rerun scientific analysis")

    monkeypatch.setattr(StudyAnalysis, "compute", forbidden)
    product = store.decision(execution.execution_id, name="cpu-decision", contract="example/cpu:v1")
    assert product.payload["status"] == "selection_available"
    # The native decision is not a fresh ranking of checkpoint metadata. The separate ModelSet
    # selector owns those scored snapshots; this Work logged a different negative Run metric.
    assert product.payload["selection_score"] == execution.summary["best"]["selection_score"]
    assert len(product.payload["evidence"]) == 2
    assert {
        str(path): path.read_bytes()
        for path in execution.execution_dir.rglob("*")
        if path.is_file()
    } == before
