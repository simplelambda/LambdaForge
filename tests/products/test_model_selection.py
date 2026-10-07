"""Selected checkpoints need their own metrics, exact latest Attempts and independent bytes."""

from __future__ import annotations

import copy
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.products import ProductRegistry, SelectionPolicy, select_models
from lambdaforge.work.managed import fingerprint


def record(
    root: Path,
    run_id: str,
    score: float,
    width: int = 64,
    *,
    attempt: int = 1,
    status: str = "succeeded",
) -> dict[str, Any]:
    run_dir = root / "runs" / run_id / "attempts" / f"attempt-{attempt:04d}"
    run_dir.mkdir(parents=True)
    model = run_dir / "artifacts/model/weights.json"
    model.parent.mkdir(parents=True)
    model.write_text(
        json.dumps({"width": width, "seed": int(run_id.split("-")[-1]), "attempt": attempt})
    )
    digest, size = fingerprint(model)
    return {
        "name": "training",
        "run_id": run_id,
        "attempt_id": run_dir.name,
        "attempt_number": attempt,
        "run_dir": str(run_dir),
        "status": status,
        "seed": int(run_id.split("-")[-1]),
        "parameters": {"width": width},
        "metrics": {"score": 99.0},
        "artifacts": [
            {
                "name": "model",
                "path": "artifacts/model/weights.json",
                "role": "model",
                "sha256": digest,
                "size_bytes": size,
                "metadata": {"metrics": {"score": score, "accuracy": score}, "step": 7},
            }
        ],
        "inputs": [
            {
                "name": "dataset",
                "kind": "dataset",
                "content_id": "sha256:" + "b" * 64,
                "path": "/operational/path",
            }
        ],
    }


def source(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "execution_result_version": 1,
        "execution_id": root.name,
        "name": "models",
        "status": "failed",
        "scientific_fingerprint": "full-producer-config",
        "summary": {"objective": {"metric": "score", "mode": "max"}},
        "runs": rows,
    }


def test_grouped_selection_reuses_latest_success_and_ignores_failed_auxiliary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "execution-test"
    rows = [
        record(root, "run-1", 0.4),
        record(root, "run-2", 0.8),
        record(root, "run-3", 0.7, 128),
        record(root, "run-4", 0.9, 128, status="failed"),
        record(root, "run-5", 0.99, status="failed"),
        record(root, "run-5", 0.2, attempt=2),
    ]
    original = copy.deepcopy(rows)
    policy = SelectionPolicy("score", group_by=("width",))
    selection = select_models(
        source(root, rows), root, policy, name="per-width", contract="example/models:v1"
    )
    assert {model["run_id"] for model in selection.product.payload["models"]} == {"run-2", "run-3"}
    assert all(model["metrics"]["score"] != 99 for model in selection.product.payload["models"])
    assert rows == original
    assert pickle.loads(pickle.dumps(selection)) == selection
    assert "/operational/path" not in json.dumps(selection.product.scientific_meaning)
    registry = ProductRegistry(tmp_path / "products")
    registry.publish(selection.product, files=dict(selection.sources), apply=True)
    shutil.rmtree(root)
    assert registry.verify("per-width")["artifacts"] == 2
    assert len(registry.show("per-width").payload["models"]) == 2


def test_selection_excludes_pruned_and_low_fidelity_without_opening_unselected_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "execution-test"
    rows = [
        record(root, "run-1", 0.99),
        record(root, "run-2", 0.98),
        record(root, "run-3", 0.8),
        record(root, "run-4", 0.3),
    ]
    rows[0]["termination_type"] = "performance_pruned"
    rows[1]["fidelity"] = {"target": 5, "maximum": 50}
    for row in (rows[0], rows[1], rows[3]):
        (Path(row["run_dir"]) / row["artifacts"][0]["path"]).unlink()
    selection = select_models(
        source(root, rows), root, SelectionPolicy("score"), name="top", contract="example/models:v1"
    )
    assert selection.product.payload["models"][0]["run_id"] == "run-3"
    assert len(selection.sources) == 1


def test_explicit_ties_constraints_and_snapshot_metric_binding(tmp_path: Path) -> None:
    root = tmp_path / "execution-test"
    rows = [record(root, "run-1", 0.8), record(root, "run-2", 0.79), record(root, "run-3", 0.4)]
    policy = SelectionPolicy(
        "score",
        tie_policy="include_equivalent",
        practical_margin=0.02,
        constraints={"accuracy": {"min": 0.5}},
    )
    selection = select_models(
        source(root, rows), root, policy, name="equivalent", contract="example/models:v1"
    )
    assert len(selection.sources) == 2
    assert any(
        row["reason"] == "artifact-bound-constraint-not-satisfied"
        for row in selection.product.payload["excluded"]
    )
    for row in rows:
        row["artifacts"][0]["metadata"] = {}
    with pytest.raises(ValueError, match="latest/best Run metrics"):
        select_models(
            source(root, rows), root, policy, name="unknown", contract="example/models:v1"
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"top_k": True},
        {"mode": "unknown"},
        {"group_by": "width"},
        {"constraints": False},
        {"constraints": {"score": {"min": float("nan")}}},
        {"practical_margin": 0.1},
        {"unknown": 3},
    ],
)
def test_selection_rejects_ambiguous_policy(invalid: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SelectionPolicy.from_mapping({"rank_by": "score", **invalid})


def test_corrupt_and_unowned_selected_paths_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "execution-test"
    row = record(root, "run-1", 0.8)
    model = Path(row["run_dir"]) / row["artifacts"][0]["path"]
    original = model.read_bytes()
    model.write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="exact persisted"):
        select_models(
            source(root, [row]),
            root,
            SelectionPolicy("score"),
            name="bad",
            contract="example/models:v1",
        )
    model.write_bytes(original)
    row["run_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="owned Execution"):
        select_models(
            source(root, [row]),
            root,
            SelectionPolicy("score"),
            name="bad",
            contract="example/models:v1",
        )


def test_conditional_inactive_groups_are_not_null_groups(tmp_path: Path) -> None:
    root = tmp_path / "execution-test"
    inactive = record(root, "run-1", 0.8)
    active_null = record(root, "run-2", 0.7)
    inactive["parameters"] = {}
    active_null["parameters"] = {"width": None}
    selection = select_models(
        source(root, [inactive, active_null]),
        root,
        SelectionPolicy("score", group_by=("width",)),
        name="groups",
        contract="example/models:v1",
    )
    assert len(selection.sources) == 2
    assert {model["group"]["width"]["active"] for model in selection.product.payload["models"]} == {
        True,
        False,
    }


def test_native_select_has_read_only_preview_and_durable_apply(tmp_path: Path) -> None:
    results = tmp_path / "runs"
    root = results / "training" / "execution-test"
    rows = [record(root, "run-1", 0.6), record(root, "run-2", 0.8)]
    (root / "result.json").write_text(json.dumps(source(root, rows)))
    policy = tmp_path / "selection.yaml"
    policy.write_text("rank_by: score\nmode: max\ntop_k: 1\n")
    products = tmp_path / "products"
    args = [
        "products",
        "select",
        "execution-test",
        "--name",
        "best-model",
        "--contract",
        "example/models:v1",
        "--policy",
        str(policy),
        "--results-root",
        str(results),
        "--root",
        str(products),
        "--json",
    ]
    assert CommandLineInterface.main(args) == 0
    assert not products.exists()
    assert CommandLineInterface.main([*args, "--apply"]) == 0
    assert ProductRegistry(products).show("best-model").payload["models"][0]["run_id"] == "run-2"


def test_native_work_snapshot_is_selected_without_confusing_last_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.work import ResultStore, WorkConfig, WorkRunner

    source_yaml = tmp_path / "work.yaml"
    source_yaml.write_text(
        "name: scored\nrun: tests.work_cases.ScoredModelSnapshotWork\n"
        "seeds: [1, 2]\nresources: {cpu: 1, memory: 128MiB}\n"
    )
    monkeypatch.chdir(tmp_path)
    execution = WorkRunner().run(WorkConfig.from_yaml(source_yaml))
    assert execution.status == "succeeded"
    store = ResultStore(execution.execution_dir.parent.parent)
    selected = select_models(
        store.select(execution.execution_id),
        store.execution_directory(execution.execution_id),
        SelectionPolicy("score"),
        name="native-selected",
        contract="example/model-snapshot:v1",
    )
    model = selected.product.payload["models"][0]
    assert model["seed"] == 2 and model["metrics"]["score"] == 0.2 and model["step"] == 2
    registry = ProductRegistry(tmp_path / "durable-products")
    registry.publish(selected.product, files=dict(selected.sources), apply=True)
    assert json.loads(registry.artifact_path("native-selected", "model-1").read_text())["seed"] == 2
