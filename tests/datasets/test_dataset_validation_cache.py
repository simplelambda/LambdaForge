"""Shared validation uses tiny published fixtures, never consumer data or real clusters."""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge import Work
from lambdaforge.data import (
    DatasetArtifact,
    DatasetIndex,
    DatasetMember,
    DatasetPublisher,
    DatasetRegistry,
    DatasetValidationCache,
)
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.work.models import WorkInput

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX validation receipt semantics")
CODE = {"provider": "git", "revision": "fixture-code-1", "dirty": False}


class ValidationWork(Work):
    """Toy consumer uses the public facade, with no private cache paths."""

    def run(self, source: Path) -> dict:
        def scientific(root):
            assert root == source
            assert payload(self.inputs["source"]).read_bytes() == b"immutable scientific content"

        receipt = self.validate_dataset(
            "source",
            validator=scientific,
            contract="fixture.assets",
            version="1",
            parameters={"families": ["data"]},
        )
        return {"reused": receipt["reused"]}


@pytest.fixture
def published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"immutable scientific content")
    record = DatasetPublisher(DatasetRegistry(tmp_path / "registry.json")).publish_members(
        "example",
        "1",
        [{"id": "sample", "assets": {"data": "payload.bin"}}],
        source_root=source,
        publication_root=tmp_path / "published",
        cluster="local",
        build_provenance={"work": "fixture"},
    )
    return WorkInput(
        "dataset",
        "dataset",
        record.key,
        Path(record.placements[0].root),
        content_id=record.dataset_id,
    )


def validate(cache, dataset, callback, **kwargs):
    options = {"contract": "project.structure", "version": "1", "implementation": CODE}
    return cache.validate(dataset, validator=callback, **(options | kwargs))


def payload(dataset):
    artifact = DatasetArtifact.read_json(dataset.path / "dataset-artifact.json")
    member = next(iter(DatasetIndex(dataset.path / artifact.index["path"])))
    return dataset.path / member.assets["data"].path


def test_reuses_only_metadata_and_does_not_rehash_or_rebuild(published, tmp_path, monkeypatch):
    cache = DatasetValidationCache(tmp_path / "cache")
    calls = []
    first = validate(cache, published, lambda path: calls.append(path))
    assert first["valid"] and not first["reused"]
    before = {
        p.relative_to(published.path): p.read_bytes()
        for p in published.path.rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("A receipt reuse read dataset bytes or invoked expensive verification")

    monkeypatch.setattr(DatasetOperations, "verify", forbidden)
    original = Path.open

    def open_metadata_only(path, *args, **kwargs):
        if path.is_relative_to(published.path):
            forbidden()
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_metadata_only)
        second = validate(cache, published, forbidden)
    assert second["reused"] and second["validated_at_utc"] == first["validated_at_utc"]
    assert calls == [published.path]
    assert before == {
        p.relative_to(published.path): p.read_bytes()
        for p in published.path.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("change", ["version", "contract", "implementation", "parameters", "force"])
def test_changed_validation_rules_require_validation_again(published, tmp_path, change):
    cache = DatasetValidationCache(tmp_path / "cache")
    calls = []

    def callback(root):
        calls.append(root)

    validate(cache, published, callback)
    options = {
        "version": "2",
        "contract": "project.other",
        "implementation": CODE | {"revision": "fixture-code-2"},
        "parameters": {"families": ["geometry"]},
        "force": True,
    }
    assert not validate(cache, published, callback, **{change: options[change]})["reused"]
    assert len(calls) == 2


def test_changed_bytes_even_restored_mtime_are_not_reused(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    validate(cache, published, lambda _: None)
    asset = payload(published)
    before = asset.stat()
    asset.write_bytes(b"X" * before.st_size)
    os.utime(asset, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="integrity validation failed"):
        validate(cache, published, lambda _: pytest.fail("Corrupt data reached validator"))


def test_same_content_new_placement_or_metadata_revalidates(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    validate(cache, published, lambda _: None)
    relocated = tmp_path / "other-placement"
    shutil.copytree(published.path, relocated)
    assert not validate(cache, replace(published, path=relocated), lambda _: None)["reused"]
    asset = payload(published)
    stamp = asset.stat().st_mtime_ns
    os.utime(asset, ns=(stamp, stamp + 1_000_000))
    assert not validate(cache, published, lambda _: None)["reused"]


def test_corrupt_receipt_is_not_success_and_force_checks_all(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    receipt = Path(validate(cache, published, lambda _: None)["receipt"])
    receipt.write_text("broken JSON")
    with pytest.raises(ValueError, match="Corrupt validation receipt"):
        validate(cache, published, lambda _: None)
    assert not validate(cache, published, lambda _: None, force=True)["reused"]
    with pytest.raises(ValueError, match="did not accept"):
        validate(cache, published, lambda _: False, force=True)
    assert not receipt.exists()
    assert not validate(cache, published, lambda _: True)["reused"]


def test_failure_and_dataset_mutation_never_publish_success(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")

    def fail(_):
        raise RuntimeError("scientific rejection")

    with pytest.raises(RuntimeError, match="scientific rejection"):
        validate(cache, published, fail)
    assert not list(cache.root.glob("*.json"))

    def modify(root):
        (root / "extra.bin").write_bytes(b"change during validation")

    with pytest.raises(ValueError, match="changed during validation"):
        validate(cache, published, modify)
    assert not list(cache.root.glob("*.json"))


def test_concurrent_callers_execute_one_validator(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    calls = []

    def expensive(_):
        calls.append(1)
        time.sleep(0.15)

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: validate(cache, published, expensive), range(4)))
    assert calls == [1]
    assert sum(not row["reused"] for row in receipts) == 1
    assert all(json.loads(Path(row["receipt"]).read_text())["valid"] for row in receipts)


def test_unsafe_paths_mismatched_identity_and_unknown_code_rejected(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    alias = tmp_path / "alias"
    alias.symlink_to(published.path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        validate(cache, replace(published, path=alias), lambda _: None)
    with pytest.raises(ValueError, match="integrity validation failed"):
        validate(cache, replace(published, content_id="sha256:" + "f" * 64), lambda _: None)
    with pytest.raises(ValueError, match="code identity"):
        validate(cache, published, lambda _: None, implementation={"provider": "unversioned"})
    with pytest.raises(ValueError, match="must not overlap"):
        validate(DatasetValidationCache(published.path / "cache"), published, lambda _: None)


def test_callback_holds_native_gc_lease_and_rejects_symlink_receipt(published, tmp_path):
    from lambdaforge.controlplane.StorageOperations import StorageOperations

    cache = DatasetValidationCache(tmp_path / "work" / "dataset-validation")

    def scientific(_):
        assert StorageOperations._work_cache_busy(cache.root)

    receipt = Path(validate(cache, published, scientific)["receipt"])
    assert not StorageOperations._work_cache_busy(cache.root)
    receipt.unlink()
    receipt.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="symlinks"):
        validate(cache, published, lambda _: pytest.fail("Unsafe callback"), force=True)


def test_missing_index_and_bounded_inventory_fail_closed(published, tmp_path, monkeypatch):
    cache = DatasetValidationCache(tmp_path / "cache")
    receipt = Path(validate(cache, published, lambda _: None)["receipt"])
    artifact = DatasetArtifact.read_json(published.path / "dataset-artifact.json")
    index = published.path / artifact.index["path"]
    original = index.read_bytes()
    index.unlink()
    with pytest.raises(ValueError, match="integrity validation failed"):
        validate(cache, published, lambda _: pytest.fail("Missing index reached callback"))
    assert not receipt.exists()
    index.write_bytes(original)
    monkeypatch.setattr(DatasetValidationCache, "MAX_ENTRIES", 1)
    with pytest.raises(ValueError, match="inventory bound"):
        validate(cache, published, lambda _: None)


def test_native_work_facade_reuses_validation_between_seed_runs(published, tmp_path, monkeypatch):
    import importlib

    from lambdaforge.work import WorkConfig, WorkRunner

    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setattr(runner, "_code_identity", lambda _: dict(CODE))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setenv("LAMBDAFORGE_CACHE_ROOT", str(tmp_path / "shared-cache"))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")

    def dispatch(specifications, **kwargs):
        return tuple(runner._execute_run(value) for value in specifications)

    monkeypatch.setattr(runner, "_execute_adaptive_dispatch", dispatch)
    config = WorkConfig.from_mapping(
        {
            "run": "tests.datasets.test_dataset_validation_cache.ValidationWork",
            "with": {"source": {"dataset": "dataset:" + published.logical_source}},
            "seeds": [1, 2],
            "resources": {"cpu": 1},
        },
        source=tmp_path / "work.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded", [run.failure for run in result.runs]
    assert [run.primary_result["reused"] for run in result.runs] == [False, True]
    assert len({run.run_id for run in result.runs}) == 2
    assert [run.seed for run in result.runs] == [1, 2]


def test_external_assets_and_legacy_manifests_cannot_get_local_receipts(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    index = DatasetIndex.write(
        published.path / "external-index.jsonl",
        [
            DatasetMember.from_mapping(
                {
                    "id": "external",
                    "assets": {"data": {"path": "https://example.invalid/data", "kind": "uri"}},
                }
            )
        ],
    )
    artifact = DatasetArtifact.create_v2(
        name="external",
        version="1",
        index=index,
        index_path="external-index.jsonl",
        build_provenance={"work": "fixture"},
    )
    manifest = published.path / "dataset-artifact.json"
    manifest.write_text(json.dumps(artifact.to_dict()))
    dataset = replace(published, content_id=artifact.content_id)
    with pytest.raises(ValueError, match="local checksummed assets"):
        validate(cache, dataset, lambda _: pytest.fail("External content reached callback"))
    raw = artifact.to_dict()
    raw["dataset_artifact_version"] = 1
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sealed DatasetArtifact v2"):
        validate(cache, dataset, lambda _: None)


def test_contract_bound_and_cache_permissions_are_checked(published, tmp_path):
    cache = DatasetValidationCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="metadata bound"):
        validate(cache, published, lambda _: None, parameters={"oversized": "x" * 40000})
    assert not cache.root.exists()
    cache.root.mkdir(mode=0o777)
    cache.root.chmod(0o777)
    with pytest.raises(ValueError, match="owned and not publicly writable"):
        validate(cache, published, lambda _: None)
