"""Immutable publications outlive their producer and never bypass exact byte verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.products import ProductArtifact, ProductContract, ProductRegistry, StudyProduct


def model(tmp_path: Path) -> tuple[StudyProduct, Path]:
    source = tmp_path / "producer/checkpoints/model.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"selected durable model bytes")
    product = StudyProduct(
        "models",
        "ModelSet",
        ProductContract("example/models:v1", ("dataset", "rule")),
        {"selected_runs": ["run-1"]},
        {"dataset": "dataset@1", "rule": "best"},
        {"execution_id": "execution-1", "evidence_fingerprint": "evidence-1"},
        (
            ProductArtifact(
                "model",
                "artifacts/model.bin",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                source.stat().st_size,
                "checkpoint",
            ),
        ),
    )
    return product, source


def snapshot(path: Path) -> dict[str, bytes | None]:
    return {
        str(item.relative_to(path)): item.read_bytes() if item.is_file() else None
        for item in path.rglob("*")
    }


def test_empty_queries_and_preview_do_not_create_registry(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    before = snapshot(tmp_path)
    assert registry.list() == ()
    with pytest.raises(KeyError, match="Unknown"):
        registry.show("missing")
    plan = registry.publish(product, files={"model": source})
    assert not plan["applied"] and not plan["reuse"]
    assert snapshot(tmp_path) == before
    assert not registry.root.exists()


def test_promoted_bytes_survive_deleted_producer_and_reads_are_pure(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    registry.publish(product, files={"model": source}, apply=True)
    shutil.rmtree(source.parents[1])
    before = snapshot(tmp_path)
    assert registry.show(product.name).content_id == product.content_id
    assert registry.list() == (product,)
    assert (
        registry.resolve(
            product.name,
            contract=product.contract,
            scientific_expectations={"dataset": "dataset@1"},
        )
        == product
    )
    assert registry.verify(product.name)["size_bytes"] == product.artifacts[0].size_bytes
    assert registry.provenance(product.name) == (dict(product.producer),)
    assert registry.publish(product)["reuse"]
    assert snapshot(tmp_path) == before
    # The exact object can be reused even when no original producer remains.
    assert registry.publish(product, apply=True)["applied"]


def test_wrong_bytes_leave_no_partial_object_or_binding(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    source.write_bytes(b"X" * source.stat().st_size)
    with pytest.raises(ValueError, match="checksum"):
        registry.publish(product, files={"model": source}, apply=True)
    assert registry.list() == ()
    assert list((registry.root / "objects").iterdir()) == []
    assert source.read_bytes().startswith(b"X")


def test_publication_is_idempotent_and_origins_are_not_rewritten(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        plans = list(
            pool.map(
                lambda _: registry.publish(product, files={"model": source}, apply=True), range(4)
            )
        )
    assert all(plan["applied"] for plan in plans)
    assert len(list((registry.root / "objects").iterdir())) == 1
    assert len(registry.provenance(product.name)) == 1
    later = replace(
        product,
        producer={
            **product.producer,
            "execution_id": "execution-2",
            "full_config_hash": "operationally-changed",
        },
    )
    registry.publish(later, apply=True)
    assert registry.show(product.name).producer == product.producer
    assert len(registry.provenance(product.name)) == 2


def test_immutable_names_contracts_and_expectations_fail_closed(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    registry.publish(product, files={"model": source}, apply=True)
    with pytest.raises(ValueError, match="immutable"):
        registry.publish(replace(product, payload={"selected_runs": ["run-2"]}))
    other = replace(
        product,
        name="another",
        contract=ProductContract("example/models:v1", ("other",)),
        scientific_meaning={"other": True},
    )
    with pytest.raises(ValueError, match="different declaration"):
        registry.publish(other, files={"model": source})
    with pytest.raises(ValueError, match="contract mismatch"):
        registry.resolve(product.name, contract="example/models:v2")
    with pytest.raises(ValueError, match="expectation differs"):
        registry.resolve(
            product.name, contract=product.contract, scientific_expectations={"rule": False}
        )


def test_metadata_byte_bound_matches_written_encoding(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    large = replace(product, payload={"unicode": "é" * 200_000})
    registry.publish(large, files={"model": source}, apply=True)
    assert registry.show(large.name) == large
    with pytest.raises(ValueError, match="bounded"):
        registry.publish(
            replace(product, payload={"unicode": "é" * 300_000}), files={"model": source}
        )


def test_corruption_and_symbolic_provenance_are_rejected(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    registry.publish(product, files={"model": source}, apply=True)
    object_root = registry.root / "objects" / product.content_id.removeprefix("sha256:")
    stored = object_root / product.artifacts[0].path
    stored.write_bytes(b"X" * stored.stat().st_size)
    assert registry.show(product.name) == product  # metadata inspection is deliberately cheap
    with pytest.raises(ValueError, match="checksum"):
        registry.verify(product.name)
    stored.write_bytes(source.read_bytes())
    provenance = object_root / "provenance"
    entry = next(provenance.glob("*.json"))
    entry.write_text(json.dumps({**product.producer, "execution_id": "forged"}))
    with pytest.raises(ValueError, match="attestation checksum"):
        registry.provenance(product.name)
    shutil.rmtree(provenance)
    outside = tmp_path / "outside"
    outside.mkdir()
    provenance.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        registry.publish(product, apply=True)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    "path", ["manifest.json", "manifest.json/model", "provenance/model", "artifacts/\0bad"]
)
def test_artifact_metadata_collision_or_control_characters_rejected(
    tmp_path: Path, path: str
) -> None:
    product, _ = model(tmp_path)
    with pytest.raises(ValueError, match="metadata|relative path"):
        replace(product, artifacts=(replace(product.artifacts[0], path=path),))


def test_symlinked_sources_and_nonboolean_apply_rejected(tmp_path: Path) -> None:
    registry = ProductRegistry(tmp_path / "registry")
    product, source = model(tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlinked"):
        registry.publish(product, files={"model": link})
    unsafe: Any = "false"
    with pytest.raises(TypeError, match="boolean"):
        registry.publish(product, files={"model": source}, apply=unsafe)
    assert not registry.root.exists()
