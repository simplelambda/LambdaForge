"""Persist explicit all-member approval, never weaken exact content or guess equivalence."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lambdaforge.data import (
    DatasetEquivalenceCertificate,
    DatasetOperations,
    DatasetPublisher,
    DatasetRegistry,
)
from lambdaforge.products import ProductBundle, ProductRegistry
from tests.datasets.test_dataset_reconstruction import POLICY, SCIENCE, make, verifier


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")
    }


def build(a: Path, b: Path, **kwargs) -> DatasetEquivalenceCertificate:
    return DatasetEquivalenceCertificate.build(
        a,
        b,
        name="protein-representation-v1",
        scientific_contract=SCIENCE,
        producer={
            "execution_id": "comparison-operation-1",
            "evidence_fingerprint": "report-1",
            "created_at_utc": "2026-10-07T00:00:00+00:00",
        },
        verifier=verifier,
        verifier_id="project.verifier:v1",
        policy=POLICY,
        **kwargs,
    )


@pytest.mark.parametrize("sign,delta", [(1, 0), (1, 4.66e-10), (-1, 0)])
def test_certifies_full_verified_comparison_without_rewriting_datasets(
    tmp_path: Path,
    sign: int,
    delta: float,
) -> None:
    a, left, _ = make(tmp_path, "a")
    b, right, _ = make(tmp_path, "b", sign=sign, delta=delta)
    before = snapshot(tmp_path)
    certificate = build(a, b)
    registry = ProductRegistry(tmp_path / "catalog")
    plan = certificate.publish(registry)
    assert not plan["applied"] and snapshot(tmp_path) == before
    assert not registry.root.exists()
    certificate.publish(registry, apply=True)
    restored = DatasetEquivalenceCertificate.load(registry, certificate.product.name)
    assert restored == certificate
    assert left.dataset_id != right.dataset_id
    assert restored.accepts(
        reference_content_id=left.dataset_id,
        candidate_content_id=right.dataset_id,
        scientific_contract=SCIENCE,
    )
    assert not restored.accepts(
        reference_content_id=right.dataset_id,
        candidate_content_id=left.dataset_id,
        scientific_contract=SCIENCE,
    )
    assert pickle.loads(pickle.dumps(restored)) == restored
    evidence = restored.product.payload["evidence"]
    assert not evidence["byte_equal"] and len(evidence["members"]) == 1
    assert "root" not in evidence["left"]
    assert restored.product.producer["dataset_comparison_roots"]["left"] == str(a)
    assert DatasetOperations.verify(a, left.dataset_id)["valid"]
    assert DatasetOperations.verify(b, right.dataset_id)["valid"]


def test_exact_bytes_are_certifiable_but_still_need_declared_contract(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    certificate = DatasetEquivalenceCertificate.build(
        a,
        a,
        name="exact-copy",
        scientific_contract=SCIENCE,
        producer={"execution_id": "compare-1", "evidence_fingerprint": "exact-1"},
    )
    assert certificate.product.payload["evidence"]["status"] == "exact"
    assert certificate.product.payload["evidence"]["byte_equal"]
    assert certificate.product.scientific_meaning["verifier"] is None
    wrong = {**SCIENCE, "algorithm": "different-method"}
    with pytest.raises(ValueError, match="scientific contract"):
        DatasetEquivalenceCertificate.build(
            a,
            a,
            name="wrong",
            scientific_contract=wrong,
            producer={"execution_id": "compare-1", "evidence_fingerprint": "exact-1"},
        )


def test_relocation_does_not_change_certificate_content_identity(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    certificate = build(a, b)
    copy_a, copy_b = tmp_path / "copy-a", tmp_path / "copy-b"
    shutil.copytree(a, copy_a)
    shutil.copytree(b, copy_b)
    relocated = build(copy_a, copy_b)
    assert relocated.product.content_id == certificate.product.content_id
    assert relocated.product.producer != certificate.product.producer


def test_certificate_transport_survives_producer_deletion_without_claiming_current_integrity(
    tmp_path: Path,
) -> None:
    a, _, _ = make(tmp_path, "a")
    b, right, _ = make(tmp_path, "b")
    certificate = build(a, b)
    registry = ProductRegistry(tmp_path / "catalog")
    certificate.publish(registry, apply=True)
    bundle = tmp_path / "bundle"
    ProductBundle.export(registry, certificate.product.name, bundle, apply=True)
    shutil.rmtree(a)
    shutil.rmtree(registry.root)
    ProductBundle.import_bundle(registry, bundle, apply=True)
    ProductBundle.import_bundle(registry, bundle, apply=True)
    restored = DatasetEquivalenceCertificate.load(registry, certificate.product.name)
    assert restored.product.content_id == certificate.product.content_id
    asset = b / next(iter(DatasetOperations._index(b))).assets["geometry"].path
    asset.write_bytes(b"corrupt")
    assert not DatasetOperations.verify(b, right.dataset_id)["valid"]
    # The certificate is historical evidence, not a waiver of the separate integrity check.
    assert restored.product.payload["result"] == "scientifically_equivalent"


@pytest.mark.parametrize("change", ["numeric", "label", "split", "contract", "corruption"])
def test_unapproved_or_corrupt_dataset_cannot_be_certified(tmp_path: Path, change: str) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(
        tmp_path,
        "b",
        delta=0.5 if change == "numeric" else 0,
        target=0 if change == "label" else 1,
        split="test" if change == "split" else "train",
        contract={**SCIENCE, "algorithm": "different"} if change == "contract" else None,
    )
    if change == "corruption":
        asset = b / next(iter(DatasetOperations._index(b))).assets["geometry"].path
        asset.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Cannot certify"):
        build(a, b)
    assert not (tmp_path / "catalog").exists()


def test_unresolved_comparison_is_not_an_approval(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    with pytest.raises(ValueError, match="trusted project verifier"):
        DatasetEquivalenceCertificate.build(
            a,
            b,
            name="unresolved",
            scientific_contract=SCIENCE,
            producer={"execution_id": "compare-1", "evidence_fingerprint": "exact-1"},
        )


@pytest.mark.parametrize("field", ["result", "approval", "content", "contract"])
def test_corrupt_certificate_fields_fail_closed(tmp_path: Path, field: str) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    certificate = build(a, b)
    value = json.loads(json.dumps(certificate.product.to_dict()))
    if field == "result":
        value["payload"]["result"] = "unresolved"
    elif field == "approval":
        value["payload"]["evidence"]["scientifically_equivalent"] = False
    elif field == "content":
        value["scientific_meaning"]["candidate_content_id"] = "sha256:" + "a" * 64
    else:
        value["scientific_meaning"]["scientific_contract"]["algorithm"] = "other"
    altered = replace(
        certificate.product,
        payload=value["payload"],
        scientific_meaning=value["scientific_meaning"],
    )
    with pytest.raises(ValueError):
        DatasetEquivalenceCertificate(altered)


def test_explicit_acceptance_never_guesses_contract_or_transitivity(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    certificate = build(a, b)
    meaning = certificate.product.scientific_meaning
    assert not certificate.accepts(
        reference_content_id=meaning["reference_content_id"],
        candidate_content_id="sha256:" + "f" * 64,
        scientific_contract=SCIENCE,
    )
    assert not certificate.accepts(
        reference_content_id=meaning["reference_content_id"],
        candidate_content_id=meaning["candidate_content_id"],
        scientific_contract={**SCIENCE, "algorithm": "new"},
    )
    # Both mapping layers are frozen and safe for spawned process transport.
    with pytest.raises(TypeError):
        certificate.product.payload["evidence"]["policy"]["eigenvectors"]["atol"] = 100
    json.dumps(certificate.product.to_dict(), allow_nan=False)


def test_every_changed_member_must_pass_not_just_one_example(tmp_path: Path) -> None:
    science = {**SCIENCE, "selection": ["a", "b"], "labels": {"a": 1, "b": 0}}
    roots = []
    for label in ("left", "right"):
        source = tmp_path / label
        source.mkdir()
        for name in ("a", "b"):
            np.savez(
                source / f"{name}.npz",
                eigenvectors=np.eye(2),
                metadata_json=json.dumps({"path": str(source)}),
            )
        record = DatasetPublisher(DatasetRegistry(tmp_path / "registry.json")).publish_members(
            "two-members",
            "1",
            [
                {
                    "id": name,
                    "split": "train",
                    "targets": {"label": name == "a"},
                    "assets": {"geometry": f"{name}.npz"},
                }
                for name in ("a", "b")
            ],
            source_root=source,
            publication_root=tmp_path / "sealed" / label,
            build_provenance={"comparison_fixture": label},
            intent="rebuild",
            scientific_identity=science,
        )
        roots.append(Path(record.placements[0].root))
    calls = []

    def check(context):
        calls.append(context.left.member_id)
        return {"equivalent": context.left.member_id == "a", "checked_assets": ["geometry"]}

    with pytest.raises(ValueError, match="Cannot certify.*different"):
        DatasetEquivalenceCertificate.build(
            *roots,
            name="all-members",
            scientific_contract=science,
            producer={"execution_id": "compare-1", "evidence_fingerprint": "all-1"},
            verifier=check,
            verifier_id="project.all-members:v1",
            policy=POLICY,
        )
    assert sorted(calls) == ["a", "b"]


def test_large_report_is_rejected_not_truncated_or_partially_published(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")

    def full_report(context):
        return {**verifier(context), "details": {"evidence": "x" * 600_000}}

    certificate = DatasetEquivalenceCertificate.build(
        a,
        b,
        name="large-report",
        scientific_contract=SCIENCE,
        producer={"execution_id": "compare-1", "evidence_fingerprint": "large-1"},
        verifier=full_report,
        verifier_id="project.verifier:v1",
        policy=POLICY,
    )
    registry = ProductRegistry(tmp_path / "catalog")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="bounded metadata"):
        certificate.publish(registry)
    with pytest.raises(ValueError, match="bounded metadata"):
        certificate.publish(registry, apply=True)
    assert snapshot(tmp_path) == before


def test_timestamp_is_required_operational_provenance(tmp_path: Path) -> None:
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    certificate = build(a, b)
    producer = dict(certificate.product.producer)
    assert producer.pop("dataset_comparison_completed_at_utc")
    with pytest.raises(ValueError, match="timezone-aware"):
        DatasetEquivalenceCertificate(replace(certificate.product, producer=producer))
