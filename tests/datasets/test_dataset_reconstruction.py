"""Exact transport identities, explicit scientific comparison and publication recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lambdaforge import Work
from lambdaforge.cli import CommandLineInterface
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.data import (
    DatasetArtifact,
    DatasetComparison,
    DatasetOperations,
    DatasetPublisher,
    DatasetRegistry,
    DatasetService,
)
from lambdaforge.data.errors import InvalidDatasetPublicationError
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.retention import retention_plan
from tests.work.test_checkpoint_retention import runtime

SCIENCE = {
    "sources": {"10FI_Y": "sha256:" + "a" * 64},
    "selection": ["10FI_Y"],
    "labels": {"10FI_Y": 1},
    "configuration": {"modes": 2},
    "algorithm": "spectral-contract-v1",
}
POLICY = {
    "metadata_json": {"method": "compare-scientific-fields-only"},
    "eigenvectors": {"method": "projector", "atol": 1e-8, "rtol": 0.0},
}


class LateDatasetPublisher(Work):
    """Legacy producer without early preflight: failure must still preserve its work."""

    def run(self, payload: str):
        asset = self.outputs.file("geometry", filename="geometry.txt")
        asset.write_text(payload)
        self.checkpoints.save_json("computed.json", {"completed": True})
        self.metrics.log("geometries", 1)
        return self.outputs.dataset(
            name="late", version="1", members=[{"id": "one", "path": str(asset)}]
        )


def make(tmp_path, name, *, sign=1, delta=0, contract=None, split="train", target=1):
    source = tmp_path / name
    source.mkdir()
    np.savez(
        source / "protein.npz",
        eigenvectors=sign * np.eye(2) + delta,
        metadata_json=json.dumps({"path": str(source), "framework": name}),
    )
    registry = DatasetRegistry(tmp_path / "registry.json")
    publisher = DatasetPublisher(registry)
    record = publisher.publish_members(
        "proteins",
        "6",
        [
            {
                "id": "10FI_Y",
                "split": split,
                "targets": {"label": target},
                "assets": {"geometry": "protein.npz"},
            }
        ],
        source_root=source,
        publication_root=tmp_path / "candidates" / name,
        build_provenance={"host": name},
        scientific_identity=contract or SCIENCE,
        intent="rebuild",
    )
    return Path(record.placements[0].root), record, publisher


def verifier(context):
    with (
        np.load(context.left_root / context.left.assets["geometry"].path) as a,
        np.load(context.right_root / context.right.assets["geometry"].path) as b,
    ):
        # Sign/base-independent spectral invariant, not indiscriminate rounding.
        x, y = a["eigenvectors"], b["eigenvectors"]
        result = np.allclose(x @ x.T, y @ y.T, atol=context.policy["eigenvectors"]["atol"], rtol=0)
    return {
        "equivalent": bool(result),
        "checked_assets": ["geometry"],
        "details": {"invariant": "spectral projector"},
    }


def compare(a, b):
    return DatasetComparison.compare(
        a, b, verifier=verifier, verifier_id="project.verifier:v1", policy=POLICY
    )


def test_exact_reconstruction_and_operational_provenance_are_separate(tmp_path):
    a, record, publisher = make(tmp_path, "original")
    # A manifest's execution path/version/date is not asset byte identity.
    artifact = DatasetArtifact.read_json(a / "dataset-artifact.json")
    value = artifact.to_dict()
    value["producer"] = {"execution": "/another/host", "framework": "future"}
    other = tmp_path / "identical"
    from lambdaforge.work.snapshot import copy_tree

    copy_tree(a, other)
    (other / "dataset-artifact.json").write_text(json.dumps(value))
    result = DatasetComparison.compare(a, other)
    assert result["byte_equal"] and result["status"] == "exact"
    assert result["left"]["scientific_id"] == result["right"]["scientific_id"]
    publisher.registry.register(record)
    before = publisher.registry.path.read_bytes()
    assert publisher.preflight("proteins", "6", intent="reuse")["existing"]
    with pytest.raises(InvalidDatasetPublicationError, match="Before computing"):
        publisher.preflight("proteins", "6")
    assert publisher.registry.path.read_bytes() == before


@pytest.mark.parametrize("sign,delta", [(1, 0), (1, 4.66e-10), (-1, 0)])
def test_explicit_verifier_covers_operational_numerical_and_spectral_changes(tmp_path, sign, delta):
    a, _, publisher = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b", sign=sign, delta=delta)
    result = compare(a, b)
    assert result["status"] == "equivalent"
    assert result["scientifically_equivalent"] is True
    assert result["byte_equal"] is False
    assert result["policy"] == POLICY
    assert len(result["members"]) == 1
    assert not publisher.registry.path.exists()
    assert DatasetComparison.compare(a, b)["status"] == "unresolved"


@pytest.mark.parametrize("change", ["source", "configuration", "label", "split", "numeric"])
def test_real_scientific_changes_cannot_be_tolerated(tmp_path, change):
    a, _, _ = make(tmp_path, "a")
    contract = json.loads(json.dumps(SCIENCE))
    if change == "source":
        contract["sources"]["10FI_Y"] = "sha256:" + "b" * 64
    if change == "configuration":
        contract["configuration"]["modes"] = 3
    b, _, _ = make(
        tmp_path,
        "b",
        contract=contract,
        delta=0.5 if change == "numeric" else 0,
        target=0 if change == "label" else 1,
        split="test" if change == "split" else "train",
    )
    assert compare(a, b)["scientifically_equivalent"] is False


def test_corruption_and_incomplete_verifier_are_not_equivalence(tmp_path):
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    with pytest.raises(ValueError, match="every changed"):
        DatasetComparison.compare(
            a,
            b,
            verifier=lambda ctx: {"equivalent": True, "checked_assets": []},
            verifier_id="bad:v1",
            policy=POLICY,
        )
    asset = next((b / "assets").rglob("geometry-*"))
    asset.write_bytes(b"corrupt")
    report = DatasetComparison.compare(
        a,
        b,
        verifier=lambda ctx: pytest.fail("verifier invoked"),
        verifier_id="project:v1",
        policy=POLICY,
    )
    assert report["status"] == "invalid"
    assert report["scientifically_equivalent"] is None


def test_publication_conflict_retains_candidate_and_retries_only_bytes(tmp_path, monkeypatch):
    context, _ = runtime(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "datasets.json"))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    source = context.run_dir / "artifacts/payload.bin"
    source.parent.mkdir()
    source.write_bytes(b"first")
    original = OutputCollection(context).dataset(
        name="corpus",
        version="1",
        members=[{"id": "a", "path": "artifacts/payload.bin"}],
    )
    registry = DatasetRegistry(tmp_path / "datasets.json")
    before = registry.path.read_bytes()
    source.write_bytes(b"rebuilt")
    with pytest.raises(InvalidDatasetPublicationError, match="preserved at"):
        OutputCollection(context).dataset(
            name="corpus",
            version="1",
            members=[{"id": "a", "path": "artifacts/payload.bin"}],
        )
    assert registry.path.read_bytes() == before
    candidate = next(context.checkpoints._root.rglob("dataset-artifact.json")).parent
    artifact = DatasetArtifact.read_json(candidate / "dataset-artifact.json")
    assert artifact.dataset_id != original["dataset_id"]
    assert DatasetOperations.verify(candidate, artifact.dataset_id)["valid"]
    result = SimpleNamespace(
        run_dir=context.run_dir, ok=False, failure={"type": "InvalidDatasetPublicationError"}
    )
    assert retention_plan(result)["paths"] == ()
    publisher = DatasetPublisher(registry)
    with pytest.raises(InvalidDatasetPublicationError):
        publisher.publish_candidate(candidate, publication_root=tmp_path / "published")
    preview = publisher.publish_candidate(
        candidate, publication_root=tmp_path / "published", version="2"
    )
    assert not preview["applied"] and registry.path.read_bytes() == before
    published = publisher.publish_candidate(
        candidate, publication_root=tmp_path / "published", version="2", apply=True
    )
    assert published["applied"] and registry.get("corpus@1").dataset_id == original["dataset_id"]
    assert registry.get("corpus@2").dataset_id == artifact.dataset_id
    assert candidate.is_dir() and source.read_bytes() == b"rebuilt"


def test_inventory_keeps_all_locations_and_conflicts_without_mutation(tmp_path, monkeypatch):
    _, record, publisher = make(tmp_path, "a")
    placement = record.placements[0]
    first = replace(record, placements=(replace(placement, cluster="gpu12"),))
    second = replace(record, placements=(replace(placement, cluster="gpu16"),))
    publisher.registry.register(first)
    before = publisher.registry.path.read_bytes()
    catalog = ClusterCatalog({name: ClusterProfile(name) for name in ("local", "gpu12", "gpu16")})
    service = DatasetService(publisher.registry, catalog)
    remote = {"gpu12": (first,), "gpu16": (second,)}
    monkeypatch.setattr(service, "_remote_records", lambda cluster: remote[cluster])
    records = service.list(all_clusters=True)
    assert len(records) == 1
    assert {p.cluster for p in records[0].placements} == {"gpu12", "gpu16"}
    assert publisher.registry.path.read_bytes() == before
    remote["gpu16"] = (replace(second, dataset_id="sha256:" + "f" * 64),)
    records = service.list(all_clusters=True)
    assert len(records) == 2 and service.discovery_warnings
    assert publisher.registry.path.read_bytes() == before
    assert not service.publication_preflight("proteins@6", intent="rebuild")["allowed"]
    monkeypatch.setattr(service, "_operation", lambda *args: {"exists": True})
    assert service.reconcile("proteins@6", cluster="gpu16")["action"] == "REFUSE"


def test_native_publication_preview_apply_and_preflight(tmp_path, monkeypatch, capsys):
    candidate, record, publisher = make(tmp_path, "a")
    storage = ClusterStoragePolicy(
        str(tmp_path / "state"),
        str(tmp_path / "cache"),
        str(tmp_path / "jobs"),
        str(tmp_path / "published"),
    )
    catalog = ClusterCatalog({"local": ClusterProfile("local", storage=storage)})
    monkeypatch.setattr(ClusterCatalog, "load", lambda *a, **kw: catalog)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(publisher.registry.path))
    assert CommandLineInterface.main(["datasets", "preflight", "proteins@6", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["allowed"]
    assert (
        CommandLineInterface.main(["datasets", "publish-candidate", str(candidate), "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["applied"] is False
    assert not publisher.registry.path.exists()
    assert (
        CommandLineInterface.main(
            ["datasets", "publish-candidate", str(candidate), "--apply", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["record"]["dataset_id"] == record.dataset_id
    assert CommandLineInterface.main(["datasets", "preflight", "proteins@6", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["allowed"] is False


def test_scientific_contract_and_path_validation(tmp_path):
    root, _, publisher = make(tmp_path, "a")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        publisher.publish_candidate(alias, publication_root=tmp_path / "published")
    value = json.loads((root / "dataset-artifact.json").read_text())
    value["scientific_identity"]["algorithm"] = "tampered"
    (root / "dataset-artifact.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="declaration differs"):
        DatasetArtifact.read_json(root / "dataset-artifact.json")


def test_native_comparison_reports_exact_and_unresolved_without_mutation(
    tmp_path, monkeypatch, capsys
):
    a, _, publisher = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    catalog = ClusterCatalog({"local": ClusterProfile("local")})
    monkeypatch.setattr(ClusterCatalog, "load", lambda *a, **kw: catalog)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(publisher.registry.path))
    assert CommandLineInterface.main(["datasets", "compare", str(a), str(a)]) == 0
    human = capsys.readouterr().out
    assert "Dataset comparison: EXACT" in human
    assert "Exact byte equality: True" in human
    report = tmp_path / "comparison.json"
    assert CommandLineInterface.main(
        ["datasets", "compare", str(a), str(b), "--output", str(report), "--json"]
    ) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unresolved" and not result["registration_changed"]
    assert json.loads(report.read_text()) == result
    assert not publisher.registry.path.exists()


def test_exact_bytes_do_not_relabel_an_existing_scientific_contract(tmp_path):
    candidate, record, publisher = make(tmp_path, "a")
    publisher.registry.register(record)
    before = publisher.registry.path.read_bytes()
    value = DatasetArtifact.read_json(candidate / "dataset-artifact.json").to_dict()
    value["metadata"]["lambdaforge_science"]["algorithm"] = "new-contract"
    # Construct a valid different declaration with identical asset bytes.
    from lambdaforge.data.DatasetArtifact import DatasetArtifact as Manifest

    value["scientific_identity"] = value["metadata"]["lambdaforge_science"]
    value["scientific_id"] = Manifest._digest(
        {"scientific_identity_version": 1, "contract": value["scientific_identity"]}
    )
    (candidate / "dataset-artifact.json").write_text(json.dumps(value))
    with pytest.raises(InvalidDatasetPublicationError, match="different scientific"):
        publisher.publish_candidate(candidate, publication_root=tmp_path / "published", apply=True)
    assert publisher.registry.path.read_bytes() == before


def test_conflict_candidate_survives_when_recovery_volume_is_full(tmp_path, monkeypatch):
    import errno
    import os

    candidate, record, publisher = make(tmp_path, "a")
    publisher.registry.register(record)
    source = tmp_path / "changed"
    source.mkdir()
    (source / "data.bin").write_bytes(b"a changed representation")
    recovery = tmp_path / "recovery"
    original_replace = os.replace

    def full_volume(src, dst):
        if Path(dst).is_relative_to(recovery):
            raise OSError(errno.ENOSPC, "simulated full checkpoint volume")
        return original_replace(src, dst)

    monkeypatch.setattr(os, "replace", full_volume)
    with pytest.raises(InvalidDatasetPublicationError, match="preserved at"):
        publisher.publish_members(
            "proteins",
            "6",
            [{"id": "x", "path": "data.bin"}],
            source_root=source,
            publication_root=tmp_path / "published",
            build_provenance={},
            recovery_root=recovery,
        )
    saved = next((tmp_path / "published").rglob(".reconstruction.*"))
    manifest = DatasetArtifact.read_json(saved / "dataset-artifact.json")
    assert DatasetOperations.verify(saved, manifest.dataset_id)["valid"]
    assert publisher.registry.get(record.key).dataset_id == record.dataset_id
    assert candidate.is_dir()


def test_complete_comparison_never_accepts_only_one_changed_member(tmp_path):
    publisher = DatasetPublisher(DatasetRegistry(tmp_path / "registry.json"))
    roots = []
    for label in ("a", "b"):
        source = tmp_path / label
        source.mkdir()
        (source / "x").write_text(label)
        (source / "y").write_text(label)
        record = publisher.publish_members(
            "corpus",
            "6",
            [{"id": key, "path": key} for key in ("x", "y")],
            source_root=source,
            publication_root=tmp_path / "candidate" / label,
            build_provenance={},
            scientific_identity=SCIENCE,
            intent="rebuild",
        )
        roots.append(record.placements[0].root)
    observed = []

    def project(ctx):
        observed.append(ctx.left.member_id)
        return {"equivalent": ctx.left.member_id == "x", "checked_assets": ["data"]}

    report = DatasetComparison.compare(
        *roots,
        verifier=project,
        verifier_id="project:v1",
        policy={"data": {"method": "project-specific"}},
    )
    assert observed == ["x", "y"] and report["status"] == "different"


def test_work_publication_failure_preserves_metrics_outputs_and_checkpoint_lifecycle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    registry = DatasetRegistry(tmp_path / "datasets.json")
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(registry.path))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")

    def config(payload):
        return WorkConfig.from_mapping(
            {
                "name": "late-publication",
                "run": "tests.datasets.test_dataset_reconstruction.LateDatasetPublisher",
                "with": {"payload": payload},
            },
            source=tmp_path / "work.yaml",
        )

    first = WorkRunner().run(config("first"))
    assert first.status == "succeeded"
    original_id = registry.get("late@1").dataset_id
    second = WorkRunner().run(config("different"))
    assert second.status == "failed"
    result = second.runs[0]
    assert result.failure["type"] == "InvalidDatasetPublicationError"
    assert result.metrics["geometries"] == 1
    assert list((result.run_dir / "artifacts").rglob("geometry.txt"))
    assert (result.run_dir.parent.parent / "checkpoints/computed.json").is_file()
    assert list((result.run_dir.parent.parent / "checkpoints").rglob("dataset-artifact.json"))
    assert registry.get("late@1").dataset_id == original_id


def test_legacy_reference_can_use_explicit_content_bound_project_contract_without_edits(tmp_path):
    a, _, _ = make(tmp_path, "a")
    b, _, _ = make(tmp_path, "b")
    path = a / "dataset-artifact.json"
    value = json.loads(path.read_text())
    value["metadata"].pop("lambdaforge_science")
    value.pop("scientific_identity")
    value.pop("scientific_id")
    path.write_text(json.dumps(value))
    before = path.read_bytes()
    legacy = DatasetArtifact.read_json(path)
    assert compare(a, b)["status"] == "unresolved"
    result = DatasetComparison.compare(
        a,
        b,
        verifier=verifier,
        verifier_id="project:v1",
        policy=POLICY,
        scientific_contracts={legacy.content_id: SCIENCE},
    )
    assert result["status"] == "equivalent"
    assert result["left"]["scientific_id"] is None
    assert result["left"]["comparison_scientific_id"] == result["right"]["scientific_id"]
    assert path.read_bytes() == before
    wrong = {**SCIENCE, "algorithm": "different"}
    modern = DatasetArtifact.read_json(b / "dataset-artifact.json")
    with pytest.raises(ValueError, match="cannot override"):
        DatasetComparison.compare(a, b, scientific_contracts={modern.content_id: wrong})
