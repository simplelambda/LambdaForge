"""Products are native typed Work inputs, not producer config/status dependencies."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.controlplane import ClusterProfile, ControlPlane, LocalTransport
from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder
from lambdaforge.products import ProductArtifact, ProductContract, ProductRegistry, StudyProduct
from lambdaforge.products.dependency import ProductRequirement, resolve_product_input
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.runner import _identity_values, _resolve_inputs


def prepared(tmp_path: Path) -> tuple[ProductRegistry, StudyProduct]:
    (tmp_path / "pyproject.toml").write_text('[project]\nname="consumer"\nversion="1"\n')
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"exact independently promoted weights")
    product = StudyProduct(
        "models",
        "ModelSet",
        ProductContract("example/models:v1", ("dataset",)),
        {"models": [{"artifact": "model", "width": 64}]},
        {"dataset": "data@1"},
        {
            "execution_id": "producer-original",
            "evidence_fingerprint": "evidence-original",
            "source_status": "failed",
            "config_fingerprint": "original-yaml-not-current-yaml",
        },
        (
            ProductArtifact(
                "model",
                "models/weights.bin",
                hashlib.sha256(weights.read_bytes()).hexdigest(),
                weights.stat().st_size,
            ),
        ),
    )
    registry = ProductRegistry(tmp_path / ".lambdaforge" / "products")
    registry.publish(product, files={"model": weights}, apply=True)
    weights.unlink()
    return registry, product


def marker() -> dict[str, Any]:
    return {
        "product": {
            "name": "models",
            "contract": "example/models:v1",
            "expect": {"dataset": "data@1"},
        }
    }


def consumer(root: Path) -> WorkConfig:
    return WorkConfig.from_mapping(
        {
            "name": "consumer",
            "run": "tests.work_cases.ProductConsumerWork",
            "with": {"selection": marker(), "artifact": "model"},
            "resources": {"cpu": 1, "memory": "128MiB"},
        },
        source=root / "consumer.yaml",
    )


def tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_product_validation_resolution_and_preview_are_read_only(tmp_path: Path) -> None:
    registry, product = prepared(tmp_path)
    before = tree(tmp_path)
    config = consumer(tmp_path)
    WorkRunner().plan(config)
    value, manifest = resolve_product_input(marker()["product"], tmp_path)
    assert value.product == product and manifest.is_file()
    assert pickle.loads(pickle.dumps(value)) == value
    parameters, inputs, identity = _resolve_inputs({"selection": marker()}, tmp_path)
    assert parameters["selection"].content_id == product.content_id
    assert inputs["selection"].kind == "product"
    assert identity == _identity_values({"selection": marker()}, tmp_path)
    assert "original-yaml" not in json.dumps(identity)
    assert tree(tmp_path) == before
    assert registry.consumers("models") == ()


def test_native_consumer_works_after_producer_deletion_and_tracks_exact_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, product = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    execution = WorkRunner().run(consumer(tmp_path))
    assert execution.status == "succeeded"
    run = execution.runs[0]
    assert run.primary_result["content_id"] == product.content_id
    assert run.inputs[0].content_id == product.content_id
    assert (
        run.run_dir / run.artifacts[0].path
    ).read_bytes() == b"exact independently promoted weights"
    audits = registry.consumers("models")
    assert len(audits) == 1
    assert audits[0]["consumer"] == {
        "execution_id": execution.execution_id,
        "run_id": run.run_id,
        "attempt_id": run.attempt_id,
        "input": "selection",
        "scientific_fingerprint": run.scientific_fingerprint,
    }
    before = tree(tmp_path)
    assert WorkRunner().run(consumer(tmp_path)).execution_id == execution.execution_id
    assert len(registry.consumers("models")) == 1
    assert tree(tmp_path) == before


@pytest.mark.parametrize("changed", ["name", "contract", "expect", "unknown", "shape"])
def test_missing_or_incompatible_requirement_has_actionable_diagnostic(
    tmp_path: Path,
    changed: str,
) -> None:
    prepared(tmp_path)
    requirement: Any = marker()["product"]
    if changed == "name":
        requirement["name"] = "missing"
    elif changed == "contract":
        requirement["contract"] = "example/models:v2"
    elif changed == "expect":
        requirement["expect"] = {"dataset": "other-data@1"}
    elif changed == "unknown":
        requirement["ignore_hash"] = True
    else:
        requirement = "models"
    with pytest.raises((ValueError, KeyError)):
        resolve_product_input(requirement, tmp_path)


def test_worker_without_operational_root_refuses_job_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared(tmp_path)
    monkeypatch.delenv("LAMBDAFORGE_PRODUCT_ROOT", raising=False)
    monkeypatch.setenv("LAMBDAFORGE_EXECUTION_MODE", "worker")
    before = tree(tmp_path)
    with pytest.raises(ValueError, match="control plane"):
        resolve_product_input(marker()["product"], tmp_path)
    assert tree(tmp_path) == before


def test_only_artifact_access_reads_and_checks_bytes(tmp_path: Path) -> None:
    registry, _product = prepared(tmp_path)
    path = registry.artifact_path("models", "model")
    original = path.read_bytes()
    path.write_bytes(b"X" * len(original))
    value, _manifest = resolve_product_input(marker()["product"], tmp_path)
    assert value.payload["models"][0]["width"] == 64
    with pytest.raises(ValueError, match="checksum"):
        value.artifact("model")


def test_bundle_pins_content_without_copying_any_models(tmp_path: Path) -> None:
    _registry, product = prepared(tmp_path)
    required: list[dict[str, Any]] = []
    pinned = ExecutionBundleBuilder._pin_products(
        {"with": {"selection": marker()}}, tmp_path, required
    )
    requirement = pinned["with"]["selection"]["product"]
    assert requirement["name"] == product.content_id
    assert requirement["contract"] == product.contract.identifier
    assert required == [requirement]
    assert ProductRequirement.from_mapping(requirement).expect == {"dataset": "data@1"}


def test_host_probe_checks_only_bounded_metadata_placement_and_contract(tmp_path: Path) -> None:
    registry, product = prepared(tmp_path)
    # Loopback tests the host script; this is not acceptance against a production SSH scheduler.
    profile = ClusterProfile(
        "fixture",
        python=sys.executable,
        storage={
            "state_root": str(registry.root.parent),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "jobs"),
        },
    )
    required = [{**marker()["product"], "name": product.content_id}]
    before = tree(tmp_path)
    ControlPlane._verify_product_inputs(LocalTransport(), profile, required)
    assert tree(tmp_path) == before
    registry.artifact_path("models", "model").unlink()
    with pytest.raises(ValueError, match="Export/import"):
        ControlPlane._verify_product_inputs(LocalTransport(), profile, required)


def test_consumer_audit_is_idempotent_concurrent_read_only_and_native_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry, _product = prepared(tmp_path)
    record = {
        "execution_id": "execution-consumer",
        "run_id": "run-consumer",
        "attempt_id": "attempt-0001",
        "input": "models",
        "scientific_fingerprint": "science",
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: registry.record_consumer("models", record), range(4)))
    before = tree(tmp_path)
    assert len(registry.consumers("models")) == 1
    assert (
        CommandLineInterface.main(
            [
                "products",
                "consumers",
                "models",
                "--root",
                str(registry.root),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["items"][0]["consumer"] == record
    assert tree(tmp_path) == before
    with pytest.raises(ValueError, match="Conflicting scientific"):
        registry.record_consumer("models", {**record, "scientific_fingerprint": "different"})
