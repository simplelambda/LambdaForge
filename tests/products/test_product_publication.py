"""Declared native post-Study publication and retry without repeating successful science."""

from __future__ import annotations

import copy
import errno
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.products import ProductRegistry, publish_declared_products
from lambdaforge.products.publication import pending_publications, publication_declarations
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner


def document() -> dict[str, Any]:
    return {
        "name": "publication",
        "run": "tests.work_cases.ScoredModelSnapshotWork",
        "seeds": [4, 7],
        "objective": {"metric": "score", "mode": "max"},
        "resources": {"cpu": 1, "memory": "128MiB"},
        "products": {
            "decision": {"kind": "StudyDecision", "contract": "example/decision:v1"},
            "models": {
                "kind": "ModelSet",
                "contract": "example/models:v1",
                "select": {"artifact": "model", "rank_by": "score", "top_k": 1},
            },
        },
    }


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkConfig:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="test-publication"\nversion="1"\n')
    monkeypatch.chdir(tmp_path)
    return WorkConfig.from_mapping(document(), source=tmp_path / "train.yaml")


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_publication_declaration_and_dry_run_do_not_create_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = prepare(tmp_path, monkeypatch)
    before = snapshot(tmp_path)
    plan = WorkRunner().run(config, dry_run=True)
    assert plan.execution_id
    assert config.explanation()["products"] == document()["products"]
    pending = pending_publications(config.raw, plan.execution_id)
    assert pending["status"] == "pending"
    assert len(pending["items"]) == 2
    assert snapshot(tmp_path) == before


def test_modelset_only_study_does_not_require_objective_or_final_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare(tmp_path, monkeypatch)
    value = document()
    value.pop("objective")
    value["products"].pop("decision")
    execution = WorkRunner().run(WorkConfig.from_mapping(value, source=tmp_path / "models.yaml"))
    assert execution.status == "succeeded"
    assert ResultStore().product_status(execution.execution_id)["status"] == "published"
    assert ProductRegistry().show("models").payload["models"][0]["seed"] == 7


def test_native_study_publishes_decision_and_independent_selected_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = WorkRunner().run(prepare(tmp_path, monkeypatch))
    assert execution.status == "succeeded"
    registry = ProductRegistry()
    decision, models = registry.show("decision"), registry.show("models")
    assert decision.payload["selection_score"] == execution.summary["best"]["selection_score"]
    assert models.payload["models"][0]["metrics"]["score"] == 0.7
    assert len(models.artifacts) == 1
    assert ResultStore().product_status(execution.execution_id)["status"] == "published"
    assert (
        json.loads((execution.execution_dir / "execution.json").read_text())["expected_products"]
        == document()["products"]
    )
    receipt = json.loads((execution.execution_dir / "products.json").read_text())
    assert receipt["status"] == "published"
    history = execution.execution_dir / "product-publication-history.jsonl"
    assert len(history.read_text().splitlines()) == 2
    # Promoted bytes/receipt remain usable when source model artifacts disappear.
    for run in execution.runs:
        for artifact in run.artifacts:
            (run.run_dir / artifact.path).unlink()
    before = snapshot(tmp_path)
    store = ResultStore()
    assert store.finalize_products(execution.execution_id)["status"] == "ready"
    assert snapshot(tmp_path) == before
    repeated = store.finalize_products(execution.execution_id, apply=True)
    assert all(item["reused"] for item in repeated["items"])
    assert history.read_text().count("\n") == 2
    assert registry.verify("models")["verified"]


def test_failed_promotion_preserves_results_and_retries_only_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = prepare(tmp_path, monkeypatch)
    original = ProductRegistry.publish

    def fail_model(self: ProductRegistry, product: Any, **kwargs: Any) -> dict[str, Any]:
        if product.name == "models":
            raise OSError(errno.ENOSPC, "simulated destination full")
        return original(self, product, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(ProductRegistry, "publish", fail_model)
        execution = WorkRunner().run(config)
    assert execution.status == "succeeded" and all(run.ok for run in execution.runs)
    assert "Training evidence retained" in capsys.readouterr().err
    receipt = execution.execution_dir / "products.json"
    failed = json.loads(receipt.read_text())
    assert [item["status"] for item in failed["items"]] == ["published", "failed"]
    assert failed["items"][1]["failure"]["phase"] == "product-publication"
    result_bytes = (execution.execution_dir / "result.json").read_bytes()
    run_paths = sorted((execution.execution_dir / "runs").rglob("result.json"))
    store = ResultStore()
    before = snapshot(tmp_path)
    plan = store.finalize_products(execution.execution_id)
    assert plan["status"] == "ready" and snapshot(tmp_path) == before
    assert store.finalize_products(execution.execution_id, apply=True)["status"] == "published"
    assert sorted((execution.execution_dir / "runs").rglob("result.json")) == run_paths
    assert (execution.execution_dir / "result.json").read_bytes() == result_bytes
    assert (
        len(
            (execution.execution_dir / "product-publication-history.jsonl").read_text().splitlines()
        )
        == 3
    )


def test_concurrent_publication_retry_is_idempotent_and_cli_preview_is_pure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    execution = WorkRunner().run(prepare(tmp_path, monkeypatch))
    before = snapshot(tmp_path)
    assert CommandLineInterface.main(["products", "status", execution.execution_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "published"
    assert (
        CommandLineInterface.main(["products", "finalize", execution.execution_id, "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert snapshot(tmp_path) == before
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: ResultStore().finalize_products(execution.execution_id, apply=True),
                range(4),
            )
        )
    assert all(result["status"] == "published" for result in results)
    assert len(ProductRegistry().list()) == 2
    assert (
        len(
            (execution.execution_dir / "product-publication-history.jsonl").read_text().splitlines()
        )
        == 2
    )


@pytest.mark.parametrize("change", ["result", "declaration", "symbolic", "corrupt"])
def test_publication_refuses_changed_authority_and_unsafe_or_corrupt_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    execution = WorkRunner().run(prepare(tmp_path, monkeypatch))
    source = execution.to_dict()
    config = document()
    if change == "result":
        source["runs"][0]["metrics"]["score"] = 999
    elif change == "declaration":
        config["products"]["models"]["select"]["top_k"] = 2
    elif change == "symbolic":
        receipt = execution.execution_dir / "products.json"
        other = tmp_path / "unowned-receipt.json"
        other.write_bytes(receipt.read_bytes())
        receipt.unlink()
        receipt.symlink_to(other)
    else:
        (execution.execution_dir / "products.json").write_text(
            '{"product_publication_version": true}'
        )
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        publish_declared_products(config, source, execution.execution_dir, ProductRegistry())
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "invalid", ["kind", "select", "contract", "unknown", "ordinary", "composition"]
)
def test_invalid_publication_is_rejected_before_any_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    prepare(tmp_path, monkeypatch)
    value = copy.deepcopy(document())
    if invalid == "kind":
        value["products"]["models"]["kind"] = "guess"
    elif invalid == "select":
        value["products"]["decision"]["select"] = {"rank_by": "score"}
    elif invalid == "contract":
        value["products"]["models"]["contract"] = "unversioned"
    elif invalid == "unknown":
        value["products"]["models"]["ignore_hash"] = True
    elif invalid == "ordinary":
        value.pop("seeds")
    else:
        value.pop("run")
        value.pop("seeds")
        value.pop("objective")
        value["steps"] = [{"run": "tests.work_cases.SeedWork"}]
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        WorkConfig.from_mapping(value, source=tmp_path / "invalid.yaml")
    assert snapshot(tmp_path) == before
    with pytest.raises(ValueError):
        publication_declarations({})
