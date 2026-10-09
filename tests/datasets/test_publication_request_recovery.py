"""Publication-only recovery covers failures before the first asset was copied."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lambdaforge.controlplane.StorageAdmission import StorageOwnershipError
from lambdaforge.data import DatasetIndex, DatasetOperations, DatasetPublisher, DatasetRegistry
from lambdaforge.data.errors import InvalidDatasetPublicationError


def preparation(tmp_path):
    source = tmp_path / "scientific-result"
    source.mkdir()
    (source / "geometry.bin").write_bytes(b"already calculated and scientifically validated")
    publisher = DatasetPublisher(DatasetRegistry(tmp_path / "registry.json"))
    members = [
        {
            "id": "protein",
            "split": "train",
            "targets": {"label": 1},
            "assets": {"geometry": "geometry.bin"},
        }
    ]
    options = dict(
        source_root=source,
        publication_root=tmp_path / "published",
        build_provenance={"execution_id": "original", "run_id": "same-run"},
        recovery_root=tmp_path / "checkpoints",
        metadata={"scientific_validation": "PASS"},
        target_schema={"type": "object", "required": ["label"]},
        scientific_identity={
            "sources": {"protein": "fixed"},
            "selection": ["protein"],
            "labels": {"protein": 1},
            "configuration": {},
            "algorithm": "unchanged",
        },
    )
    return publisher, members, options


def fail_before_copy(tmp_path, monkeypatch):
    from lambdaforge.work import snapshot

    publisher, members, options = preparation(tmp_path)
    original = snapshot.copy_file

    def refused(*args, **kwargs):
        raise StorageOwnershipError(
            "publication-reflink",
            Path(args[1]),
            {
                "reason": "unresolved-storage-owner",
                "free_bytes": 500_000_000_000,
                "blocking_lease": "synthetic-lease",
                "blocking_owner": {"pid": 123},
            },
        )

    monkeypatch.setattr(snapshot, "copy_file", refused)
    with pytest.raises(StorageOwnershipError):
        publisher.publish_members("corpus", "6", iter(members), **options)
    monkeypatch.setattr(snapshot, "copy_file", original)
    request = next((tmp_path / "checkpoints").glob("request-*"))
    return publisher, request, options


def test_ownership_failure_preserves_exact_request_and_retries_no_science(tmp_path, monkeypatch):
    publisher, request, options = fail_before_copy(tmp_path, monkeypatch)
    original = (options["source_root"] / "geometry.bin").read_bytes()
    assert publisher.registry.records() == ()
    assert not list((tmp_path / "published").rglob("dataset-artifact.json"))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    preview = publisher.publish_candidate(request, publication_root=options["publication_root"])
    assert preview["will_execute"] is False and preview["applied"] is False
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    restored = publisher.publish_candidate(
        request, publication_root=options["publication_root"], apply=True
    )
    assert restored["applied"] and restored["content_id"] == preview["content_id"]
    record = publisher.registry.get("corpus@6")
    assert record.producer["execution_id"] == "original"
    assert record.metadata["scientific_validation"] == "PASS"
    assert DatasetOperations.verify(record.placements[0].root, record.dataset_id)["valid"]
    assert (options["source_root"] / "geometry.bin").read_bytes() == original
    assert (
        publisher.publish_candidate(
            request, publication_root=options["publication_root"], apply=True
        )["content_id"]
        == preview["content_id"]
    )


@pytest.mark.parametrize("damage", ["source", "index", "source-symlink", "request-symlink"])
def test_saved_request_refuses_changed_evidence_and_unsafe_paths(tmp_path, monkeypatch, damage):
    publisher, request, options = fail_before_copy(tmp_path, monkeypatch)
    if damage == "source":
        (options["source_root"] / "geometry.bin").write_bytes(b"not the validated result")
    elif damage == "index":
        with (request / "members.jsonl").open("a") as stream:
            stream.write("\n")
    elif damage == "source-symlink":
        source = options["source_root"] / "geometry.bin"
        source.rename(source.with_suffix(".saved"))
        source.symlink_to(source.with_suffix(".saved"))
    else:
        path = request / "publication-request.json"
        path.rename(request / "saved.json")
        path.symlink_to(request / "saved.json")
    with pytest.raises((InvalidDatasetPublicationError, ValueError)):
        publisher.publish_candidate(
            request, publication_root=options["publication_root"], apply=True
        )
    assert publisher.registry.records() == ()


def test_duplicate_publication_recovery_serializes_and_keeps_original_request(
    tmp_path, monkeypatch
):
    publisher, request, options = fail_before_copy(tmp_path, monkeypatch)
    declaration = (request / "publication-request.json").read_bytes()
    with ThreadPoolExecutor(max_workers=2) as executor:
        replies = list(
            executor.map(
                lambda _: publisher.publish_candidate(
                    request, publication_root=options["publication_root"], apply=True
                ),
                range(2),
            )
        )
    assert all(reply["applied"] for reply in replies)
    assert replies[0]["content_id"] == replies[1]["content_id"]
    assert len(publisher.registry.records()) == 1
    assert (request / "publication-request.json").read_bytes() == declaration


def test_retained_index_can_prepare_recovery_without_recomputing_or_registering(tmp_path):
    publisher, members, options = preparation(tmp_path)
    request = publisher.prepare_publication(
        "corpus",
        "6",
        iter(members),
        source_root=options["source_root"],
        request_root=tmp_path / "requests",
        build_provenance=options["build_provenance"],
    )
    value = json.loads((request / "publication-request.json").read_text())
    assert value["index_sha256"] == DatasetIndex(request / "members.jsonl").file_sha256()
    assert publisher.registry.records() == ()
    assert not options["publication_root"].exists()


def test_request_works_through_native_cli_preview_and_apply(tmp_path, monkeypatch, capsys):
    from lambdaforge.cli import CommandLineInterface

    publisher, request, options = fail_before_copy(tmp_path, monkeypatch)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="publication-test"\nversion="1"\n')
    catalog = tmp_path / "lambdaforge.clusters.yaml"
    catalog.write_text(
        "clusters:\n  local:\n    storage:\n      dataset_root: "
        + str(options["publication_root"])
        + "\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(publisher.registry.path))
    arguments = ["datasets", "publish-candidate", str(request)]
    assert CommandLineInterface.main(arguments) == 0
    output = capsys.readouterr().out
    assert "Preview only" in output and str(options["publication_root"]) in output
    assert publisher.registry.records() == ()
    assert CommandLineInterface.main([*arguments, "--apply", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["applied"] and result["will_execute"] is False
