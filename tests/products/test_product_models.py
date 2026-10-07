"""Scientific meaning, exact contents and operational provenance must not be conflated."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.products import ProductArtifact, ProductContract, StudyProduct


def product() -> StudyProduct:
    return StudyProduct(
        "pooling",
        "StudyDecision",
        ProductContract("example/pooling-selection:v1", ("dataset", "objective", "selection")),
        {"preferred_parameters": {"pooling": "max"}, "unresolved_questions": []},
        {
            "dataset": "sha256:" + "a" * 64,
            "objective": {"metric": "auprc", "mode": "max"},
            "selection": {"rule": "confirmed-best", "label_semantics": "binding"},
        },
        {
            "execution_id": "execution-0123456789abcdef",
            "evidence_fingerprint": "sha256:" + "b" * 64,
            "config": {"resources": {"cpu": 2}, "ui": "dark"},
        },
    )


def test_contract_content_meaning_and_provenance_have_separate_identities() -> None:
    original = product()
    moved = replace(
        original,
        name="another-alias",
        producer={
            **original.producer,
            "config": {"resources": {"cpu": 16}, "ui": "light"},
            "path": "/another/cluster",
        },
    )
    assert moved.scientific_id == original.scientific_id
    assert moved.content_id == original.content_id
    changed_objective = replace(
        original,
        scientific_meaning={
            **original.scientific_meaning,
            "objective": {"metric": "loss", "mode": "min"},
        },
    )
    assert changed_objective.scientific_id != original.scientific_id
    assert changed_objective.content_id != original.content_id
    other_bytes = replace(original, payload={**original.payload, "numeric_details": [0.8]})
    assert other_bytes.scientific_id == original.scientific_id
    assert other_bytes.content_id != original.content_id
    # Equal declared meaning is not a byte match or a numerical-equivalence certificate.
    assert original.content_descriptor() != other_bytes.content_descriptor()


def test_product_json_pickle_and_mutation_isolation() -> None:
    original = product()
    document = json.loads(json.dumps(original.to_dict()))
    restored = StudyProduct.from_dict(document)
    assert restored == original
    assert pickle.loads(pickle.dumps(original)) == original
    document["payload"]["preferred_parameters"]["pooling"] = "lse"
    assert original.payload["preferred_parameters"]["pooling"] == "max"
    with pytest.raises(TypeError, match="immutable"):
        original.payload["preferred_parameters"]["pooling"] = "lse"
    with pytest.raises(TypeError, match="immutable"):
        original.payload["unresolved_questions"].append("new")
    with pytest.raises(ValueError, match="identity"):
        StudyProduct.from_dict(document)


def test_contract_versions_and_scientific_fields_are_explicit() -> None:
    original = product()
    reordered = replace(
        original,
        contract=ProductContract(
            original.contract.identifier, tuple(reversed(original.contract.scientific_fields))
        ),
    )
    assert reordered.content_id == original.content_id
    version_two = replace(
        original,
        contract=ProductContract(
            "example/pooling-selection:v2", original.contract.scientific_fields
        ),
    )
    assert version_two.scientific_id != original.scientific_id
    with pytest.raises(ValueError, match="missing=.*objective"):
        replace(original, scientific_meaning={"dataset": "x", "selection": {}})
    with pytest.raises(ValueError, match="extra=.*machine"):
        replace(original, scientific_meaning={**original.scientific_meaning, "machine": "cluster"})
    for identifier in ("example/pooling-selection", "../x:v1", "example/x:v0"):
        with pytest.raises(ValueError, match="versioned"):
            ProductContract(identifier, ("objective",))


@pytest.mark.parametrize(
    "unsafe", [float("nan"), float("inf"), {1: "not a string key"}, Path("local")]
)
def test_product_evidence_never_coerces_arbitrary_formats(unsafe: object) -> None:
    with pytest.raises(TypeError, match="JSON"):
        replace(product(), payload={"unsafe": unsafe})


@pytest.mark.parametrize(
    "unsafe", ["../checkpoint", "/absolute", "a/../b", "a//b", "a\\b", "C:file"]
)
def test_artifact_descriptor_rejects_unsafe_paths(unsafe: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        ProductArtifact("model", unsafe, "c" * 64, 10)


def test_exact_artifact_contents_are_not_producer_paths() -> None:
    original = product()
    artifact = ProductArtifact("model", "artifacts/model.ckpt", "c" * 64, 10, "checkpoint")
    promoted = replace(original, artifacts=(artifact,))
    assert promoted.content_id != original.content_id
    assert StudyProduct.from_dict(json.loads(json.dumps(promoted.to_dict()))) == promoted
    with pytest.raises(ValueError, match="unique"):
        replace(original, artifacts=(artifact, artifact))
    with pytest.raises(ValueError, match="checksum"):
        replace(artifact, sha256="not verified")
    with pytest.raises(ValueError, match="integer"):
        replace(artifact, size_bytes=True)
