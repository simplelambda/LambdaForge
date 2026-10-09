"""Ordinary Work evidence uses native persistence/portability, never fabricated Studies."""

from __future__ import annotations

import copy
import errno
import json
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from lambdaforge import Work
from lambdaforge.products import ProductRegistry
from lambdaforge.products.dependency import resolve_product_input
from lambdaforge.work import ResultStore, WorkConfig, WorkRunner
from lambdaforge.work.ResultInput import resolve_result_input


class EvidenceWork(Work):
    def run(self, multiplier: int = 1, fail: bool = False) -> dict[str, Any]:
        self.log("persisted scientific output")
        self.outputs.value("counts", {"processed": multiplier})
        for step in range(1, 5):
            self.metrics.log("quality", multiplier / step, step=step)
            self.metrics.log("seconds", step * 2, step=step)
        self.checkpoints.save_json("state.json", {"step": 4})
        if fail:
            raise ValueError("intentional consumer failure")
        self.outputs.file("report", filename="report.txt", role="report").write_text("exact bytes")
        self.outputs.html_section("viewer", title="Scientific viewer").write_text(
            '<html><body><script>document.body.dataset.loaded="yes"</script>'
            "<h1>Project view</h1></body></html>"
        )
        self.outputs.html_section(
            "viewer-two", section="viewer", title="Scientific viewer"
        ).write_text('<html><h1>Second document</h1><img src="data:image/png;base64,AA=="></html>')
        return {"scientific_result": multiplier}


class HistoricalConsumer(Work):
    def run(self, evidence: object) -> dict[str, Any]:
        from lambdaforge.work import ResultInput

        assert isinstance(evidence, ResultInput)
        self.outputs.value("parent", evidence.execution_id)
        return {"result": dict(evidence.result), "quality": evidence.metrics["quality"]}


class ProductConsumer(Work):
    def run(self, model: object) -> str:
        from lambdaforge.products.dependency import ProductInput

        assert isinstance(model, ProductInput)
        return model.artifact("report").read_text()


class SelectedProductConsumer(Work):
    def run(self, model: object) -> str:
        from lambdaforge.products.dependency import ProductInput

        assert isinstance(model, ProductInput)
        return model.artifact().read_text()


class PlainWork(Work):
    def run(self) -> dict[str, int]:
        self.outputs.value("counts", {"items": 3})
        return {"items": 3}


class PartialHtmlWork(Work):
    def run(self) -> None:
        self.outputs.html_section("partial", title="Partial visualization").write_text(
            "<html>partial evidence</html>"
        )
        self.metrics.log("observed", 1, step=1)
        raise ValueError("science did not finalize")


class PreviewWork(Work):
    def run(self) -> None:
        self.outputs.file("large", filename="large.txt").write_text("x" * (64 * 1024 + 1))


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="ordinary-results"\nversion="1"\n')
    return tmp_path


def execute(project: Path, **kwargs: Any) -> Any:
    document = {
        "name": "evidence",
        "run": "tests.work.test_work_results.EvidenceWork",
        "with": kwargs,
        "resources": {"cpu": 1, "memory": "128MiB"},
    }
    return WorkRunner().run(WorkConfig.from_mapping(document, source=project / "work.yaml"))


def test_work_result_report_export_import_and_reexport_without_consumer_code(project: Path) -> None:
    execution = execute(project)
    store = ResultStore()
    before = {str(p): p.read_bytes() for p in execution.execution_dir.rglob("*") if p.is_file()}
    view = store.view("evidence", view="attempt")
    assert view["study_expected"] is False
    assert len(view["curves"]) == 2
    assert view["latest"]["quality"]["value"] == 0.25
    assert view["result"] == {"scientific_result": 1}
    assert "persisted scientific output" in view["log"]
    assert len(view["artifacts"]) == 3
    assert {
        str(p): p.read_bytes() for p in execution.execution_dir.rglob("*") if p.is_file()
    } == before
    report = store.report("evidence", project / "work.html")
    text = report.read_text()
    assert "Scientific viewer" in text and "sandbox=" in text
    assert (
        "Scientific result" not in text
    )  # Authored keys are never interpreted as a Study objective.
    assert "Runs &amp; Attempts" in text
    package = store.export("evidence", project / "exports")
    assert (Path(package["path"]) / "reports/work-results.html").is_file()
    imported = ResultStore(project / "imported/runs")
    preview = imported.import_export(package["path"], product_root=project / "imported/products")
    assert not preview["applied"] and not imported.root.exists()
    imported.import_export(package["path"], product_root=project / "imported/products", apply=True)
    details = imported.view(execution.execution_id, view="attempt")
    assert details["imported"] and details["result"] == view["result"]
    assert imported.report(execution.execution_id, project / "offline.html").is_file()
    copied = imported.export(execution.execution_id, project / "copied")
    assert json.loads((Path(copied["path"]) / "manifest.json").read_text()) == json.loads(
        (Path(package["path"]) / "manifest.json").read_text()
    )


def test_failed_work_preserves_partial_metrics_logs_and_values(project: Path) -> None:
    execution = execute(project, fail=True)
    assert execution.status == "failed"
    store = ResultStore()
    view = store.view("evidence", view="attempt")
    assert view["outputs"]["counts"] == {"processed": 1}
    assert view["selected"]["failure"]["type"] == "ValueError"
    assert len(view["curves"]["quality"]) == 4
    assert store.report("evidence", project / "failed.html").is_file()
    package = store.export("evidence", project / "failed-export")
    assert package["export_kind"] == "snapshot" and package["finalized"]


def test_output_previews_are_explicit_small_verified_and_inert(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    from lambdaforge.cli.CommandLineInterface import CommandLineInterface

    execution = execute(project)
    store = ResultStore()
    preview = store.output_preview("evidence", "report")
    assert preview["verified"] and preview["content"] == "exact bytes"
    assert "<script>" in store.output_preview("evidence", "viewer")["content"]
    assert (
        CommandLineInterface.main(
            ["results", "preview-output", execution.execution_id, "report", "--json"]
        )
        == 0
    )
    assert '"verified": true' in capsys.readouterr().out
    path = execution.runs[0].run_dir / execution.runs[0].artifacts[0].path
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.output_preview("evidence", "report")
    large = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "large",
                "run": "tests.work.test_work_results.PreviewWork",
                "resources": {"cpu": 1},
            },
            source=project / "large.yaml",
        )
    )
    original = Path.open

    def no_large(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "large.txt":
            raise AssertionError("A large output must not be downloaded for a preview")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_large)
    assert not store.output_preview(large.execution_id, "large")["verified"]


def test_historical_evidence_pin_ambiguity_mutation_and_consumer_identity(project: Path) -> None:
    first = execute(project)
    historical, _ = resolve_result_input({"execution": "evidence"}, project)
    pin = historical.requirement.to_dict()
    assert historical.artifact("report").read_text() == "exact bytes"
    assert pickle.loads(pickle.dumps(historical)).metrics["quality"] == 0.25
    with pytest.raises(TypeError):
        historical.metadata["status"] = "changed"  # type: ignore[index]
    second = execute(project, multiplier=2)
    assert first.execution_id != second.execution_id
    with pytest.raises(ValueError, match="Execution ID"):
        resolve_result_input({"execution": "evidence"}, project)
    assert resolve_result_input(pin, project)[0].execution_id == first.execution_id
    consumer = WorkConfig.from_mapping(
        {
            "name": "historical-consumer",
            "run": "tests.work.test_work_results.HistoricalConsumer",
            "with": {"evidence": {"result": pin}},
            "resources": {"cpu": 1},
        },
        source=project / "consumer.yaml",
    )
    outcome = WorkRunner().run(consumer)
    assert outcome.runs[0].inputs[0].content_id == historical.evidence_id
    assert outcome.runs[0].primary_result["result"] == {"scientific_result": 1}
    result_path = first.execution_dir / "result.json"
    persisted = json.loads(result_path.read_text())
    persisted["runs"][0]["metrics"]["quality"] = 100
    result_path.write_text(json.dumps(persisted))
    with pytest.raises(ValueError, match="changed"):
        historical.verify()
    with pytest.raises(ValueError, match="identity differs"):
        resolve_result_input(pin, project)


def test_ordinary_product_publication_and_human_source_reference(project: Path) -> None:
    document = {
        "name": "published-work",
        "run": "tests.work.test_work_results.EvidenceWork",
        "resources": {"cpu": 1},
        "products": {
            "science-report": {
                "kind": "ScientificReport",
                "contract": "example/report:v1",
                "scientific_meaning": {"protocol": "explicit-v1"},
                "outputs": ["report", "counts"],
            }
        },
    }
    outcome = WorkRunner().run(WorkConfig.from_mapping(document, source=project / "published.yaml"))
    status = ResultStore().product_status(outcome.execution_id)
    assert outcome.status == "succeeded" and status["status"] == "published"
    product, _ = resolve_product_input(
        {"from": {"execution": "published-work", "output": "science-report"}}, project
    )
    assert product.product.contract.identifier == "example/report:v1"
    assert product.artifact("report").read_text() == "exact bytes"
    assert product.payload["outputs"]["counts"] == {"processed": 1}
    assert ResultStore().finalize_products(outcome.execution_id, apply=True)["items"][0]["reused"]
    consumed = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "product-consumer",
                "run": "tests.work.test_work_results.ProductConsumer",
                "with": {
                    "model": {
                        "product": {
                            "from": {"execution": "published-work", "output": "science-report"}
                        }
                    }
                },
                "resources": {"cpu": 1},
            },
            source=project / "consumer.yaml",
        )
    )
    assert consumed.status == "succeeded"
    assert consumed.runs[0].primary_result == "exact bytes"
    selected_marker = ResultStore().reference(
        outcome.execution_id, product="science-report", artifact="report"
    )
    selected, _ = resolve_product_input(selected_marker["product"], project)
    assert (
        selected.selected_artifact == "report" and selected.artifact().read_text() == "exact bytes"
    )
    with pytest.raises(ValueError, match="pinned product artifact"):
        selected.artifact("different")
    with pytest.raises(ValueError, match="not uniquely registered"):
        resolve_product_input({**selected_marker["product"], "artifact": "absent"}, project)
    selected_outcome = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "selected-consumer",
                "run": "tests.work.test_work_results.SelectedProductConsumer",
                "with": {"model": selected_marker},
                "resources": {"cpu": 1},
            },
            source=project / "selected.yaml",
        )
    )
    assert selected_outcome.status == "succeeded"
    assert selected_outcome.runs[0].parameters["model"]["artifact"] == "report"
    package = ResultStore().export(outcome.execution_id, project / "product-import-source")
    receiving = ResultStore(project / "receiving/runs")
    receiving.import_export(
        package["path"], product_root=project / ".lambdaforge/products", apply=True
    )
    assert (
        receiving.reference(outcome.execution_id, product="science-report", artifact="report")
        == selected_marker
    )
    ResultStore().delete(outcome.execution_id, apply=True)
    assert ProductRegistry().verify("science-report")["artifacts"] == 1
    assert product.artifact("report").read_text() == "exact bytes"


def test_projection_rejects_symbolic_artifacts_and_attempt_substitution(project: Path) -> None:
    execution = execute(project)
    store = ResultStore()
    run = execution.runs[0]
    original = store.select("evidence")
    changed = copy.deepcopy(original)
    changed["runs"][0]["attempt_number"] = 2
    from lambdaforge.work.result_projection import projection

    with pytest.raises(ValueError, match="identity"):
        projection(execution.execution_dir, changed, view="outputs")
    artifact = run.run_dir / run.artifacts[0].path
    artifact.unlink()
    artifact.symlink_to(project / "pyproject.toml")
    with pytest.raises(ValueError, match="symlink"):
        store.view("evidence", view="outputs")


def test_plain_and_compacted_partial_html_have_reports(project: Path) -> None:
    for name, work_class in (("plain", "PlainWork"), ("partial", "PartialHtmlWork")):
        outcome = WorkRunner().run(
            WorkConfig.from_mapping(
                {
                    "name": name,
                    "run": f"tests.work.test_work_results.{work_class}",
                    "resources": {"cpu": 1},
                },
                source=project / f"{name}.yaml",
            )
        )
        assert outcome.status == ("succeeded" if name == "plain" else "failed")
        store = ResultStore()
        assert store.report(name, project / f"{name}.html").is_file()
        if name == "plain":
            assert store.view(name, view="metrics")["curves"] == {}
        else:
            view = store.view(name, view="outputs")
            assert not view["artifacts"] or view["artifacts"][0]["availability"] == "not_retained"
            assert store.view(name, view="html")["sections"] == []


def test_compact_catalog_reads_neither_aggregates_nor_artifact_bytes(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = execute(project)
    original = Path.read_text

    def guarded(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name in {"result.json", "metrics.jsonl", "training-metrics.jsonl", "report.txt"}:
            raise AssertionError("root catalog read deep evidence")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    catalog = ResultStore().catalog()
    assert catalog[0]["execution_id"] == outcome.execution_id
    assert catalog[0]["run_count"] == 1 and catalog[0]["study_expected"] is False


def test_multiple_attempts_select_exact_evidence_and_latest(project: Path) -> None:
    outcome = execute(project)
    directory = outcome.execution_dir
    first = outcome.runs[0].run_dir
    second = first.with_name("attempt-0002")
    shutil.copytree(first, second)
    old = json.loads((first / "result.json").read_text())
    new = {
        **old,
        "attempt_id": "attempt-0002",
        "attempt_number": 2,
        "run_dir": str(second),
        "result": {"scientific_result": 9},
        "metrics": {**old["metrics"], "quality": 0.9},
    }
    (second / "result.json").write_text(json.dumps(new))
    native = json.loads((directory / "result.json").read_text())
    native["runs"].append(new)
    (directory / "result.json").write_text(json.dumps(native))
    store = ResultStore()
    assert store.view(outcome.execution_id, view="attempt", attempt=1)["result"] == {
        "scientific_result": 1
    }
    assert store.view(outcome.execution_id, view="attempt")["result"] == {"scientific_result": 9}
    assert len(store.view(outcome.execution_id)["attempts"]) == 2
    report = store.report(outcome.execution_id, project / "attempts.html")
    assert report.stat().st_size < 15 * 1024**2
    assert "Exact Run / Attempt series" in report.read_text()


@pytest.mark.parametrize("kind", ["result", "product"])
def test_fleet_preparation_attests_pinned_historical_inputs(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from dataclasses import replace

    import yaml

    from lambdaforge.controlplane.ShardPreparation import (
        prepared_equivalence,
        prepared_input_bindings,
        relocate_file_inputs,
    )
    from tests.controlplane.test_shard_preparation import preparation

    evidence = execute(project)
    parameter = "evidence"
    marker = {"result": {"execution": evidence.execution_id}}
    pinned = ResultStore().reference(evidence.execution_id)
    consumer = "HistoricalConsumer"
    if kind == "product":
        published = WorkRunner().run(
            WorkConfig.from_mapping(
                {
                    "name": "publisher",
                    "run": "tests.work.test_work_results.EvidenceWork",
                    "resources": {"cpu": 1},
                    "products": {
                        "report": {
                            "kind": "ScientificReport",
                            "contract": "example/report:v1",
                            "scientific_meaning": {"protocol": "v1"},
                            "outputs": ["report"],
                        }
                    },
                },
                source=project / "publish.yaml",
            )
        )
        parameter = "model"
        consumer = "SelectedProductConsumer"
        marker = {
            "product": {
                "from": {"execution": published.execution_id, "output": "report"},
                "artifact": "report",
            }
        }
        pinned = ResultStore().reference(
            published.execution_id, product="report", artifact="report"
        )
    source, prepared, _equivalence, _files = preparation(project)
    config = {
        "name": "historical-fleet",
        "run": f"tests.work.test_work_results.{consumer}",
        "with": {parameter: marker},
        "seeds": [4, 7],
        "resources": {"cpu": 1},
    }
    source.write_text(yaml.safe_dump(config))
    prepared.bundle.config_path.write_text(yaml.safe_dump({"with": {parameter: pinned}}))
    equivalence = prepared_equivalence(source, prepared)
    bindings = prepared_input_bindings(source, prepared, equivalence)
    assert relocate_file_inputs({parameter: marker}, bindings) == {parameter: pinned}
    assert not bindings[0].get("destination")  # Historical identities never become Job paths.
    with pytest.raises(ValueError, match="input content"):
        prepared_input_bindings(source, prepared, replace(equivalence, inputs="other"))
    variable = "LAMBDAFORGE_RESULT_INPUT_ROOT" if kind == "result" else "LAMBDAFORGE_PRODUCT_ROOT"
    monkeypatch.setenv(variable, str(project / "absent"))
    with pytest.raises(KeyError if kind == "result" else ValueError):
        relocate_file_inputs({parameter: marker}, bindings)


def test_publication_failure_is_recoverable_without_reexecution(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {
        "name": "publication-retry",
        "run": "tests.work.test_work_results.EvidenceWork",
        "resources": {"cpu": 1},
        "products": {
            "report": {
                "kind": "ScientificReport",
                "contract": "example/report:v1",
                "scientific_meaning": {"protocol": "v1"},
                "outputs": ["report"],
            }
        },
    }
    original = ProductRegistry.publish
    with monkeypatch.context() as patch:

        def full(*args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "fixture destination full")

        patch.setattr(ProductRegistry, "publish", full)
        outcome = WorkRunner().run(WorkConfig.from_mapping(document, source=project / "retry.yaml"))
    store = ResultStore()
    before = (outcome.execution_dir / "result.json").read_bytes()
    assert (
        outcome.status == "succeeded"
        and store.product_status(outcome.execution_id)["status"] == "failed"
    )
    assert original is ProductRegistry.publish
    artifact = next(row for row in outcome.runs[0].artifacts if row.name == "report")
    assert (outcome.runs[0].run_dir / artifact.path).is_file()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: store.finalize_products(outcome.execution_id, apply=True), range(2)
            )
        )
    assert all(value["items"][0]["status"] == "published" for value in results)
    assert before == (outcome.execution_dir / "result.json").read_bytes()


def test_local_launch_binding_pins_source_context_and_rejects_changed_yaml(project: Path) -> None:
    import yaml

    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.controlplane.ExecutionBundleBuilder import ExecutionBundleBuilder

    producer = execute(project)
    source = project / "consumer.yaml"
    document = {
        "name": "consumer",
        "run": "tests.work.test_work_results.HistoricalConsumer",
        "resources": {"cpu": 1},
    }
    source.write_text(yaml.safe_dump(document))
    bindings = {"evidence": {"result": {"execution": "evidence"}}}
    bundle = ExecutionBundleBuilder(project / ".lambdaforge/bundles").build(
        source, ClusterProfile("local", python=sys.executable), input_bindings=bindings
    )
    execute(project, multiplier=2)
    restored = WorkConfig.from_yaml(source, input_pins=bundle.directory / "input-pins.json")
    assert (
        restored.source == source
        and restored.raw["with"]["evidence"]["result"]["execution"] == producer.execution_id
    )
    outcome = WorkRunner().run(restored)
    assert outcome.runs[0].primary_result["result"] == {"scientific_result": 1}
    source.write_text(yaml.safe_dump({**document, "name": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        WorkConfig.from_yaml(source, input_pins=bundle.directory / "input-pins.json")


def test_live_projection_uses_exact_job_attestation_and_no_other_execution(project: Path) -> None:
    from lambdaforge.work.result_projection import live_result, projection

    outcome = execute(project)
    root = outcome.execution_dir
    origin = json.loads((root / "execution.json").read_text())
    origin["job_id"] = "job-fixture"
    (root / "execution.json").write_text(json.dumps(origin))
    (root / "result.json").unlink()
    (outcome.runs[0].run_dir / "result.json").unlink()
    selected_root, result = live_result([ResultStore().root], "job-fixture")
    assert selected_root == root and result["live"] and result["status"] == "running"
    assert projection(root, result, view="metrics")["latest"]["quality"]["value"] == 0.25
    with pytest.raises(ValueError, match="Job-attested"):
        live_result([ResultStore().root], "job-unrelated")


def test_provider_work_reader_keeps_exact_attempt_and_preview_on_host(project: Path) -> None:
    from datetime import datetime, timezone

    from lambdaforge.controlplane import (
        ClusterCatalog,
        ClusterProfile,
        JobService,
        JobState,
        JobStore,
    )
    from lambdaforge.controlplane.jobs import JobRecord

    outcome = execute(project)
    owner = project / "jobs/job-reader"
    work = owner / "work"
    work.mkdir(parents=True)
    (owner / "result.json").write_text((outcome.execution_dir / "result.json").read_text())
    (owner / "progress.json").write_text('{"completed":4,"total":4}')
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-reader",
        "local",
        "local",
        "provider-reader",
        JobState.SUCCEEDED,
        (sys.executable, "-m", "lambdaforge", "run", str(project / "work.yaml")),
        str(work),
        {},
        now,
        now,
        config_path=str(project / "work.yaml"),
        metadata={"source_config_path": str(project / "work.yaml")},
        job_type="work",
    )
    records = JobStore(project / "job-records")
    records.write(record)
    service = JobService(ClusterCatalog({"local": ClusterProfile("local")}), records)
    overview = service.work_result_view(record.job_id)
    assert "curves" not in overview and "artifacts" not in overview
    assert overview["operational"]["progress"]["completed"] == 4
    assert overview["operational"]["job_id"] == record.job_id
    options = {"run_id": outcome.runs[0].run_id, "attempt": 1}
    assert service.work_result_view(record.job_id, view="metrics", **options)["curves"]
    preview = service.work_result_view(
        record.job_id, view="outputs", artifact_preview="report", **options
    )
    assert preview["verified"] and preview["content"] == "exact bytes"
    with pytest.raises(RuntimeError, match="No persisted evidence"):
        service.work_result_view(record.job_id, view="outputs", run_id="run-unowned")

    from types import SimpleNamespace

    from lambdaforge.controlplane.StudyExportService import StudyExportService

    class ExactJobs:
        def get(self, job_id: str, **kwargs: Any) -> Any:
            assert job_id == record.job_id
            return record

        def study(self, job_id: str) -> Any:
            raise KeyError(job_id)  # Ordinary Works have no fabricated Study telemetry.

    class ExactWorks:
        def show(self, selector: str) -> Any:
            return SimpleNamespace(name="evidence", work_id="work-reader", job_ids=(record.job_id,))

    export = StudyExportService(
        service.catalog,
        jobs=ExactJobs(),  # type: ignore[arg-type]
        works=ExactWorks(),  # type: ignore[arg-type]
    ).export("work-reader", project / "provider-export")
    package = Path(export["path"])
    assert export["execution_id"] == outcome.execution_id
    assert (package / "reports/work-results.html").is_file()
    imported = ResultStore(project / "provider-import/runs")
    imported.import_export(package, product_root=project / "provider-import/products", apply=True)
    assert imported.view(outcome.execution_id, view="attempt")["result"] == {"scientific_result": 1}


def test_work_report_browser_has_lazy_safe_visualizations_and_metric_dashboard(
    project: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    execute(project)
    report = ResultStore().report("evidence", project / "browser-work.html")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium unavailable")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        external: list[str] = []
        page.on(
            "request",
            lambda request: (
                external.append(request.url)
                if request.url.startswith("https://example.com/")
                else None
            ),
        )
        page.route("https://example.com/**", lambda route: route.abort())
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        assert page.locator("iframe[srcdoc]").count() == 0
        page.get_by_role("button", name="Scientific viewer", exact=True).click()
        viewer = page.frame_locator("#project-section-0 iframe")
        assert viewer.locator("h1").inner_text() == "Project view"
        assert "allow-same-origin" not in page.locator("#project-section-0 iframe").get_attribute(
            "sandbox"
        )
        page.locator("#project-section-0-document").select_option("1")
        assert viewer.locator("h1").inner_text() == "Second document"
        viewer.locator("body").evaluate("() => { location.href = 'https://example.com/escape'; }")
        page.wait_for_timeout(200)
        assert not external  # Same parent navigation policy as native Study HTML.
        page.get_by_role("button", name="Metrics", exact=True).click()
        metrics = page.frame_locator("#project-section-1 iframe")
        metrics.locator(".js-plotly-plot").first.wait_for()
        assert "Step / observation" in metrics.locator("body").inner_text()
        page.screenshot(path=str(project / "work-results-browser.png"), full_page=True)
        assert not errors
        browser.close()


@pytest.mark.parametrize("kind", ["product", "result"])
def test_native_compressed_materialization_preview_apply_idempotence(
    project: Path, kind: str
) -> None:
    from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.controlplane.ControlPlane import ControlPlane
    from lambdaforge.controlplane.DependencyMaterialization import DependencyMaterialization
    from lambdaforge.controlplane.LocalTransport import LocalTransport

    if kind == "result":
        outcome = execute(project)
        selector = outcome.execution_id
    else:
        document = {
            "name": "materialized",
            "run": "tests.work.test_work_results.EvidenceWork",
            "resources": {"cpu": 1},
            "products": {
                "report": {
                    "kind": "ScientificReport",
                    "contract": "example/report:v1",
                    "scientific_meaning": {"protocol": "v1"},
                    "outputs": ["report"],
                }
            },
        }
        WorkRunner().run(WorkConfig.from_mapping(document, source=project / "publish.yaml"))
        selector = ProductRegistry().show("report").content_id
    target = project / "destination"
    profile = ClusterProfile.from_mapping(
        "fixture",
        {
            "transport": "local",
            "python": sys.executable,
            "workspace": str(target),
            "storage": {
                "state_root": str(target / "state"),
                "cache_root": str(target / "cache"),
                "run_root": str(target / "jobs"),
            },
        },
    )
    service = DependencyMaterialization(ClusterCatalog({"fixture": profile}))
    preview = service.materialize(selector, cluster="fixture", kind=kind)
    assert not preview["applied"] and not target.exists()
    verifier = (
        ControlPlane._verify_product_inputs
        if kind == "product"
        else ControlPlane._verify_result_inputs
    )
    expected = (
        {"name": selector, "contract": "example/report:v1", "artifact": "report"}
        if kind == "product"
        else ResultStore().reference(selector)["result"]
    )
    with pytest.raises(ValueError, match="not materialized"):
        verifier(LocalTransport(), profile, [expected])
    assert not target.exists()  # Availability inspection is read-only.
    applied = service.materialize(selector, cluster="fixture", kind=kind, apply=True)
    assert applied["applied"] and applied["compressed"] and not applied["will_execute"]
    assert service.materialize(selector, cluster="fixture", kind=kind, apply=True)["applied"]
    assert list((target / "cache").glob(".dependency-import-*")) == []
    verifier(LocalTransport(), profile, [expected])
    if kind == "product":
        with pytest.raises(ValueError, match="exact selected artifact"):
            verifier(LocalTransport(), profile, [{**expected, "artifact": "absent"}])
    if kind == "product":
        assert ProductRegistry(target / "state/products").verify(selector)["artifacts"] == 1
    else:
        restored = ResultStore(target / "state/results")
        assert restored.reference(selector)["result"]["evidence_id"] == preview["evidence_id"]
        assert restored.view(selector, view="attempt")["result"] == {"scientific_result": 1}
