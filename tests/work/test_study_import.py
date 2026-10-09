"""Portable evidence registration must not execute or falsify original producer records."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StudyExportService import StudyExportService
from lambdaforge.products import ProductRegistry
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner
from lambdaforge.work.recovery import validate_execution
from lambdaforge.work.ResultStore import _inventory


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    project = tmp_path / "producer"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="import-test"\nversion="1"\n')
    monkeypatch.chdir(project)
    config = WorkConfig.from_mapping(
        {
            "name": "portable-study",
            "run": "tests.work_cases.ScoredModelSnapshotWork",
            "seeds": [4, 7],
            "objective": {"metric": "score", "mode": "max"},
            "resources": {"cpu": 1, "memory": "128MiB"},
            "products": {
                "models": {
                    "kind": "ModelSet",
                    "contract": "example/models:v1",
                    "select": {"artifact": "model", "rank_by": "score"},
                }
            },
        },
        source=project / "train.yaml",
    )
    execution = WorkRunner().run(config)
    export = ResultStore().export(execution.execution_id, tmp_path / "exports")
    return Path(export["path"]), execution


def roots(tmp_path: Path) -> tuple[ResultStore, ProductRegistry]:
    return ResultStore(tmp_path / "consumer/runs"), ProductRegistry(tmp_path / "consumer/products")


def snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None for p in root.rglob("*")
    }


def reseal(package: Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    inventory = [x for x in _inventory(package) if x["path"] != "manifest.json"]
    manifest.update(
        inventory=inventory,
        file_count=len(inventory),
        size_bytes=sum(x["size_bytes"] for x in inventory),
    )
    (package / "manifest.json").write_text(json.dumps(manifest))


def test_export_delete_import_satisfies_product_and_reads_logs(
    package: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, execution = package
    original = snapshot(source)
    ResultStore().delete(execution.execution_id, apply=True)
    shutil.rmtree(ProductRegistry().root)
    store, products = roots(tmp_path)
    before = snapshot(tmp_path)
    preview = store.import_export(source, product_root=products.root)
    assert preview["will_execute"] is False
    assert not preview["applied"] and len(preview["products"]) == 1
    assert snapshot(tmp_path) == before

    # Import never invokes application signatures or the native execution engine.
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Study import cannot execute or compute Analysis.")

    monkeypatch.setattr(WorkRunner, "run", forbidden)
    payload = store.import_export(source, product_root=products.root, apply=True)
    assert payload["status"] == "imported"
    record = store.select(execution.execution_id)
    assert record["imported"]["will_execute"] is False
    assert len(record["runs"]) == 2
    assert record["runs"][0]["run_dir"].startswith(str(store.root))
    assert store.log_report(execution.execution_id)["failure"] is None
    assert (
        store.analysis(execution.execution_id)["source"]["execution_id"] == execution.execution_id
    )
    assert products.verify("models")["verified"]
    assert products.show("models").producer["execution_id"] == execution.execution_id
    assert snapshot(source) == original
    native_origin = source / "execution/execution.json"
    assert (store.execution_directory(execution.execution_id) / "execution.json").read_bytes() == (
        native_origin.read_bytes()
    )
    with pytest.raises(ValueError, match="read-only"):
        store.configuration(execution.execution_id)
    with pytest.raises(ValueError, match="read-only"):
        validate_execution(Path(payload["path"]))
    with pytest.raises(ValueError, match="refit"):
        store.analysis(execution.execution_id, recompute=True)
    # Product ownership is independent of the imported archive too.
    store.delete(execution.execution_id, apply=True)
    assert products.verify("models")["verified"]


def test_imported_views_are_lazy_local_and_read_only(
    package: tuple[Path, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.tui.services import ConsoleServices
    from lambdaforge.work.ImportedStudy import ImportedStudy

    source, execution = package
    store, registry = roots(tmp_path)
    store.import_export(source, product_root=registry.root, apply=True)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Root Studies must not load result envelopes or contact providers.")

    monkeypatch.setattr(ResultStore, "list", forbidden)
    services = object.__new__(ConsoleServices)
    services.results = store
    services.jobs = SimpleNamespace(study=forbidden, study_run=forbidden, study_trial=forbidden)
    values: dict[str, Any] = {}
    services._include_imports(values)
    row = values["work"]["items"][0]
    selector = row["study_selector"]
    assert "primary_job_id" not in row  # Imports are evidence, never fabricated scheduler Jobs.
    assert row["imported"] is True and row["state"] == "succeeded"
    assert row["work_id"] == "import:" + execution.execution_id
    assert "runs" not in row and "candidates" not in row["study"]
    assert len(json.dumps(row)) < 2048
    study = services.study(selector)
    assert len(study["candidates"]) == 1
    assert "runs" not in study["candidates"][0]
    trial = services.study_trial(selector, 1)
    assert len(trial["runs"]) == 2
    run = services.study_run(selector, trial["runs"][0]["key"])
    assert run["state"] == "succeeded" and "log" in run
    assert run["paths"]["run_dir"].startswith(str(store.root))
    assert services.study_analysis(selector)["source"]["execution_id"] == execution.execution_id
    assert ImportedStudy.rows(store.root) == [row]
    services._include_imports(values)
    assert values["work"]["items"] == [row]


def test_import_index_does_not_present_ordinary_works_as_studies(
    package: tuple[Path, Any],
) -> None:
    from lambdaforge.work.ImportedStudy import import_index

    source, _ = package
    manifest = json.loads((source / "manifest.json").read_text())
    result = json.loads((source / "execution/result.json").read_text())
    configuration = source / "execution/configuration.json"
    configuration.write_text(json.dumps({"name": "ordinary", "run": "consumer.Ordinary"}))
    result["summary"] = {}
    assert import_index(manifest, result, package=source)["study_expected"] is False
    configuration.write_text(
        json.dumps({"steps": [{"parallel": [{"run": "consumer.Repeated", "seeds": [4, 7]}]}]})
    )
    assert import_index(manifest, result, package=source)["study_expected"] is True


def test_large_verified_aggregate_is_not_an_individual_metadata_document(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    from lambdaforge.work.StudyImport import _mapping

    source, execution = package
    path = source / "execution/result.json"
    # Whitespace padding reproduces the real 102 MiB native envelope without constructing
    # a giant artificial scientific object. Re-seal only this test-owned export.
    with path.open("ab") as stream:
        stream.write(b" " * (65 * 1024 * 1024))
    reseal(source)
    store, registry = roots(tmp_path)
    before = snapshot(tmp_path / "consumer")
    assert store.import_export(source, product_root=registry.root)["finalized"]
    assert snapshot(tmp_path / "consumer") == before
    with pytest.raises(ValueError, match="oversized"):
        _mapping(path)
    result = store.import_export(source, product_root=registry.root, apply=True)
    assert result["applied"] and result["execution_id"] == execution.execution_id
    assert store.import_export(source, product_root=registry.root, apply=True)["reused"]


def test_existing_import_can_explicitly_regenerate_missing_presentation_indexes(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    from lambdaforge.work.ImportedStudy import ImportedStudy

    source, _ = package
    store, registry = roots(tmp_path)
    imported = store.import_export(source, product_root=registry.root, apply=True)
    directory = Path(imported["path"])
    original = (directory / "import.json").read_bytes()
    (directory / "import-study.json").unlink()
    # A partially completed older presentation upgrade may already have a view directory.
    # Reapply replaces only that disposable view and leaves the scientific archive intact.
    assert ImportedStudy.rows(store.root) == []
    assert store.import_export(source, product_root=registry.root, apply=True)["reused"]
    assert (directory / "import.json").read_bytes() == original
    assert len(ImportedStudy.rows(store.root)) == 1


@pytest.mark.parametrize("damage", ["run-path", "log-path", "result-symlink", "view-symlink"])
def test_imported_observation_refuses_paths_outside_its_local_archive(
    package: tuple[Path, Any], tmp_path: Path, damage: str
) -> None:
    from lambdaforge.work.ImportedStudy import ImportedStudy

    source, execution = package
    store, registry = roots(tmp_path)
    imported = store.import_export(source, product_root=registry.root, apply=True)
    observer = ImportedStudy(store.root, "import:" + execution.execution_id)
    run = observer.view("trial", trial=1)["runs"][0]
    path = Path(imported["path"]) / "import-view/runs" / (run["key"] + ".json")
    value = json.loads(path.read_text())
    if damage == "view-symlink":
        path.unlink()
        path.symlink_to(source / "manifest.json")
    elif damage == "result-symlink":
        from lambdaforge.work.StudyImport import relative_run_path

        native = dict(value["run"])
        native["run_id"] = Path(native["run_dir"]).parts[-3]
        result_path = observer.portable / "execution" / relative_run_path(native) / "result.json"
        result_path.unlink()
        result_path.symlink_to(source / "manifest.json")
    else:
        value["run"]["run_dir" if damage == "run-path" else "log_path"] = "/etc/passwd"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="safe|symbolic|Symbolic|outside|escaped|Attempt"):
        observer.run(run["key"], tail=30, curve_points=80)


def test_imported_published_artifacts_use_local_verified_package_paths(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    from lambdaforge.work.ImportedStudy import ImportedStudy
    from lambdaforge.work.StudyImport import relative_run_path

    source, execution = package
    path = source / "execution/result.json"
    result = json.loads(path.read_text())
    native = result["runs"][0]
    artifact = native["artifacts"][0]
    body = (source / "execution" / relative_run_path(native) / artifact["path"]).read_bytes()
    artifact["metadata"].update(published_to="/original-host/model", retention="published-only")
    run_path = source / "execution" / relative_run_path(native) / "result.json"
    value = json.loads(run_path.read_text())
    value["artifacts"][0] = artifact
    run_path.write_text(json.dumps(value))
    path.write_text(json.dumps(result))
    published = source / "published-artifacts/run-00000/0000-model"
    published.parent.mkdir(parents=True)
    published.write_bytes(body)
    reseal(source)
    store, registry = roots(tmp_path)
    store.import_export(source, product_root=registry.root, apply=True)
    observer = ImportedStudy(store.root, "import:" + execution.execution_id)
    key = observer.view("trial", trial=1)["runs"][0]["key"]
    detail = observer.run(key, tail=30, curve_points=80)
    found = next(a for a in detail["artifacts"] if a["name"] == "model")
    assert Path(found["path"]).is_relative_to(store.root)
    assert Path(found["path"]).read_bytes() == body
    assert found["published_path"] == "/original-host/model"


def test_concurrent_idempotent_import_preserves_one_receipt(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    source, execution = package
    store, registry = roots(tmp_path)

    def register(_: int) -> dict[str, Any]:
        return store.import_export(source, product_root=registry.root, apply=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(register, range(3)))
    assert len({item["path"] for item in outcomes}) == 1
    receipt = Path(outcomes[0]["path"]) / "import.json"
    original = receipt.read_bytes()
    assert store.import_export(source, product_root=registry.root, apply=True)["reused"]
    assert receipt.read_bytes() == original
    assert len(store.list()) == 1
    assert len(registry.provenance("models")) == 1
    assert store.select(execution.execution_id)["status"] == "succeeded"


@pytest.mark.parametrize(
    "damage",
    [
        "bytes",
        "extra",
        "missing",
        "traversal",
        "duplicate",
        "symlink",
        "hardlink",
        "manifest-version",
        "origin",
        "run-path",
        "product-omitted",
        "product-corrupt",
    ],
)
def test_corrupt_or_unsafe_import_refuses_without_mutation(
    package: tuple[Path, Any], tmp_path: Path, damage: str
) -> None:
    source, _ = package
    manifest = json.loads((source / "manifest.json").read_text())
    if damage == "bytes":
        (source / "README.txt").write_text("Corrupt")
    elif damage == "extra":
        (source / "extra.txt").write_text("Unlisted")
    elif damage == "missing":
        (source / "README.txt").unlink()
    elif damage == "symlink":
        (source / "README.txt").unlink()
        (source / "README.txt").symlink_to(source / "manifest.json")
    elif damage == "hardlink":
        (tmp_path / "shared").hardlink_to(source / "README.txt")
    elif damage == "origin":
        origin = source / "execution/execution.json"
        value = json.loads(origin.read_text())
        value["scientific_fingerprint"] = "wrong"
        origin.write_text(json.dumps(value))
        reseal(source)
    elif damage == "run-path":
        path = source / "execution/result.json"
        value = json.loads(path.read_text())
        value["runs"][0]["run_dir"] = "/etc"
        path.write_text(json.dumps(value))
        reseal(source)
    elif damage == "product-corrupt":
        path = next((source / "products").glob("*/product.json"))
        value = json.loads(path.read_text())
        value["payload"] = {"fabricated": True}
        path.write_text(json.dumps(value))
        reseal(source)
    else:
        if damage == "traversal":
            manifest["inventory"][0]["path"] = "../outside"
        elif damage == "duplicate":
            manifest["inventory"].append(manifest["inventory"][0])
        elif damage == "manifest-version":
            manifest["lambdaforge_export_version"] = True
        elif damage == "product-omitted":
            manifest["products"] = []
        (source / "manifest.json").write_text(json.dumps(manifest))
    store, registry = roots(tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        store.import_export(source, product_root=registry.root, apply=True)
    assert snapshot(tmp_path) == before


def test_import_refuses_conflicting_native_execution_and_corrupt_idempotent_source(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    source, execution = package
    with pytest.raises((ValueError, RuntimeError), match="import|overwrite"):
        ResultStore().import_export(source, apply=True)
    store, registry = roots(tmp_path)
    store.import_export(source, product_root=registry.root, apply=True)
    (source / "README.txt").write_text("Corrupt incoming package after valid import")
    with pytest.raises(ValueError, match="size|checksum"):
        store.import_export(source, product_root=registry.root, apply=True)
    assert store.select(execution.execution_id)["status"] == "succeeded"


def test_cli_preview_apply_and_snapshot_without_final_conclusions(
    package: tuple[Path, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, execution = package
    manifest = json.loads((source / "manifest.json").read_text())
    (source / "execution/result.json").unlink()
    manifest.update(
        finalized=False, export_kind="snapshot", status="running", execution_status=None
    )
    (source / "manifest.json").write_text(json.dumps(manifest))
    reseal(source)
    store, products = roots(tmp_path)
    argv = [
        "import",
        str(source),
        "--results-root",
        str(store.root),
        "--products-root",
        str(products.root),
        "--json",
    ]
    assert CommandLineInterface.main(argv) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not preview["finalized"] and not preview["applied"]
    assert CommandLineInterface.main([*argv, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"]
    assert store.select(execution.execution_id)["status"] == "running"
    assert store.select(execution.execution_id)["runs"] == []
    assert products.verify("models")["verified"]


def test_import_detects_corrupt_stored_bytes_on_repeated_apply(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    source, _ = package
    store, registry = roots(tmp_path)
    imported = store.import_export(source, product_root=registry.root, apply=True)
    (Path(imported["path"]) / "portable/README.txt").write_text("tampered")
    with pytest.raises(ValueError, match="size|checksum"):
        store.import_export(source, product_root=registry.root, apply=True)


def test_inventory_hashes_every_imported_byte(package: tuple[Path, Any]) -> None:
    source, _ = package
    manifest = json.loads((source / "manifest.json").read_text())
    assert all(
        hashlib.sha256((source / x["path"]).read_bytes()).hexdigest() == x["sha256"]
        for x in manifest["inventory"]
    )


def test_provider_export_includes_promoted_products_before_producer_deletion(
    package: tuple[Path, Any], tmp_path: Path
) -> None:
    source, execution = package
    job_id = "job-20261007000000-products"
    job = tmp_path / "jobs" / job_id
    work = job / "work"
    copied = work / ".lambdaforge/runs" / execution.name / execution.execution_id
    shutil.copytree(source / "execution", copied)
    record = JobRecord(
        job_id=job_id,
        cluster="local",
        scheduler="local",
        scheduler_id="1",
        state=JobState.SUCCEEDED,
        command=("lf", "run", "config.yaml"),
        work_dir=str(work),
        resources={},
        created_at_utc="2026-10-07T00:00:00+00:00",
        updated_at_utc="2026-10-07T01:00:00+00:00",
    )

    class Jobs:
        def get(self, selector: str, *, refresh: bool = True) -> JobRecord:
            return record

        def study(self, selector: str) -> dict[str, str]:
            return {"execution_id": execution.execution_id}

    class Works:
        def show(self, selector: str) -> SimpleNamespace:
            return SimpleNamespace(work_id="work-products", name=execution.name, job_ids=(job_id,))

    class Catalog:
        def get(self, selector: str) -> ClusterProfile:
            return ClusterProfile("local", python=sys.executable)

    class Factory:
        def transport(self, profile: Any) -> LocalTransport:
            return LocalTransport()

    service = StudyExportService(
        Catalog(),
        jobs=Jobs(),
        works=Works(),
        factory=Factory(),  # type: ignore[arg-type]
    )
    exported = service.export(execution.name, tmp_path / "provider-export")
    exported_path = Path(exported["path"])
    assert len(json.loads((exported_path / "manifest.json").read_text())["products"]) == 1
    ResultStore().delete(execution.execution_id, apply=True)
    shutil.rmtree(ProductRegistry().root)
    store, registry = roots(tmp_path)
    store.import_export(exported_path, product_root=registry.root, apply=True)
    assert registry.verify("models")["verified"]
    assert store.product_status(execution.execution_id)["status"] == "published"
