"""Product portability closes the byte lifecycle without restarting its source experiment."""

from __future__ import annotations

import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.products import (
    ProductArtifact,
    ProductBundle,
    ProductContract,
    ProductRegistry,
    StudyProduct,
)


def prepared(tmp_path: Path) -> tuple[ProductRegistry, StudyProduct]:
    producer = tmp_path / "attempt"
    producer.mkdir()
    source = producer / "selected.ckpt"
    source.write_bytes(b"selected checkpoint")
    product = StudyProduct(
        "selected-models",
        "ModelSet",
        ProductContract("example/models:v1", ("dataset",)),
        {"selected_runs": ["run-1"]},
        {"dataset": "data@1"},
        {"execution_id": "execution-original", "evidence_fingerprint": "evidence-1"},
        (
            ProductArtifact(
                "checkpoint",
                "models/selected.ckpt",
                hashlib.sha256(source.read_bytes()).hexdigest(),
                source.stat().st_size,
            ),
        ),
    )
    registry = ProductRegistry(tmp_path / "original-registry")
    registry.publish(product, files={"checkpoint": source}, apply=True)
    registry.publish(
        replace(product, producer={**product.producer, "execution_id": "execution-second"}),
        apply=True,
    )
    return registry, product


def tree(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_export_refuses_missing_original_provenance_before_copying(tmp_path: Path) -> None:
    registry, product = prepared(tmp_path)
    original_key = hashlib.sha256(
        __import__("json")
        .dumps(
            dict(product.producer),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        .encode()
    ).hexdigest()
    missing = (
        registry.root
        / "objects"
        / product.content_id.removeprefix("sha256:")
        / "provenance"
        / f"{original_key}.json"
    )
    missing.unlink()
    before = tree(tmp_path)
    with pytest.raises(ValueError, match="original producer"):
        ProductBundle.export(registry, product.name, tmp_path / "export", apply=True)
    assert tree(tmp_path) == before


def test_preview_export_and_verified_import_create_nothing(tmp_path: Path) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"
    before = tree(tmp_path)
    assert not ProductBundle.export(original, product.name, destination)["applied"]
    assert tree(tmp_path) == before
    ProductBundle.export(original, product.name, destination, apply=True)
    receiving = ProductRegistry(tmp_path / "receiver")
    before = tree(tmp_path)
    preview = ProductBundle.import_bundle(receiving, destination)
    assert preview["provenance_records"] == 2 and not preview["applied"]
    assert tree(tmp_path) == before


def test_export_delete_import_is_idempotent_and_preserves_exact_origin(tmp_path: Path) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"
    ProductBundle.export(original, product.name, destination, apply=True)
    shutil.rmtree(original.root)
    shutil.rmtree(tmp_path / "attempt")
    receiving = ProductRegistry(tmp_path / "receiver")
    with ThreadPoolExecutor(max_workers=3) as pool:
        plans = list(
            pool.map(
                lambda _: ProductBundle.import_bundle(receiving, destination, apply=True), range(3)
            )
        )
    assert all(plan["applied"] for plan in plans)
    assert len(receiving.list()) == 1
    assert receiving.show(product.name) == product
    assert (
        receiving.artifact_path(product.name, "checkpoint").read_bytes() == b"selected checkpoint"
    )
    assert {origin["execution_id"] for origin in receiving.provenance(product.name)} == {
        "execution-original",
        "execution-second",
    }
    assert receiving.show(product.name).producer["execution_id"] == "execution-original"


@pytest.mark.parametrize("corruption", ["artifact", "manifest", "origin", "removed-origin"])
def test_corruption_rejects_import_before_catalog_mutation(tmp_path: Path, corruption: str) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"
    ProductBundle.export(original, product.name, destination, apply=True)
    if corruption == "artifact":
        (destination / "files" / product.artifacts[0].path).write_bytes(
            b"X" * product.artifacts[0].size_bytes
        )
    elif corruption == "manifest":
        path = destination / "product.json"
        path.write_bytes(
            path.read_bytes() + b" "
        )  # same JSON meaning is not the exported metadata bytes
    else:
        path = next((destination / "provenance").glob("*.json"))
        if corruption == "origin":
            path.write_text('{"execution_id":"altered"}')
        else:
            path.unlink()
    receiving = ProductRegistry(tmp_path / "receiver")
    with pytest.raises(ValueError, match="checksum|inventory"):
        ProductBundle.import_bundle(receiving, destination, apply=True)
    assert not receiving.root.exists()


def test_import_checks_bytes_even_when_same_product_already_registered(tmp_path: Path) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"
    ProductBundle.export(original, product.name, destination, apply=True)
    (destination / "files" / product.artifacts[0].path).unlink()
    with pytest.raises(ValueError, match="regular file"):
        ProductBundle.import_bundle(original, destination, apply=True)
    assert original.verify(product.name)["verified"]


def test_export_failure_cleans_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"

    def failed_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr("lambdaforge.products.bundle.shutil.copyfile", failed_copy)
    with pytest.raises(OSError, match="simulated"):
        ProductBundle.export(original, product.name, destination, apply=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".export.product-export-*"))
    assert original.verify(product.name)["verified"]


def test_paths_and_existing_destinations_are_protected(tmp_path: Path) -> None:
    original, product = prepared(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        ProductBundle.export(original, product.name, original.root / "nested")
    symbolic = tmp_path / "symbolic"
    symbolic.symlink_to(original.root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        ProductBundle.export(original, product.name, symbolic / "export")
    destination = tmp_path / "export"
    ProductBundle.export(original, product.name, destination, apply=True)
    with pytest.raises(FileExistsError):
        ProductBundle.export(original, product.name, destination, apply=True)


def test_native_product_export_import_are_preview_first(tmp_path: Path) -> None:
    original, product = prepared(tmp_path)
    destination = tmp_path / "export"
    root = tmp_path / "imported"
    assert (
        CommandLineInterface.main(
            [
                "products",
                "export",
                product.name,
                "--root",
                str(original.root),
                "--output",
                str(destination),
                "--json",
            ]
        )
        == 0
    )
    assert not destination.exists()
    assert (
        CommandLineInterface.main(
            [
                "products",
                "export",
                product.name,
                "--root",
                str(original.root),
                "--output",
                str(destination),
                "--apply",
                "--json",
            ]
        )
        == 0
    )
    assert (
        CommandLineInterface.main(
            ["products", "import", str(destination), "--root", str(root), "--json"]
        )
        == 0
    )
    assert not root.exists()
    assert (
        CommandLineInterface.main(
            ["products", "import", str(destination), "--root", str(root), "--apply", "--json"]
        )
        == 0
    )
    assert ProductRegistry(root).show(product.name).content_id == product.content_id
