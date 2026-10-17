"""Lazy native presentation preserves full-resolution bytes and exact consumer identities."""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.report_assets import remote_asset
from lambdaforge.analysis.ReportViewer import ReportViewer
from lambdaforge.study_projection import read_html_sections
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.report_bundle import ReportBundleError
from tests.analysis.test_observed_study_charts import sample_analysis


def make_bundle(root: Path, *, large: bool = False) -> tuple[dict, list[dict], Path]:
    run_dir = root / "runs/run-one/attempts/attempt-0001"
    run_dir.mkdir(parents=True)
    outputs = OutputCollection(SimpleNamespace(run_dir=run_dir))
    bundle = outputs.html_bundle("gallery", section="proteins", title="Proteins")
    bundle.entry.write_text("""<h1>Consumer inspector</h1><p id="status">Preparing</p>
    <button id="coordinates" onclick="show('coordinates')">Coordinates</button>
    <button id="features" onclick="show('features')">Features</button>
    <button id="cancel" onclick="controller?.abort()">Cancel</button>
    <script>
    let controller;
    async function show(id) {
      controller?.abort(); controller = new AbortController();
      document.querySelector('#status').textContent = 'Loading ' + id;
      try {
        const data = await lambdaforgeReport.load(id, {signal: controller.signal});
        document.querySelector('#status').textContent = id + ': ' + data.byteLength;
        document.body.dataset.loaded = id; document.body.dataset.bytes = data.byteLength;
      } catch (error) { document.querySelector('#status').textContent = error.code || error.name; }
    }
    try { parent.document.body.dataset.escaped = 'yes'; }
    catch (_) { document.body.dataset.isolated = 'yes'; }
    fetch('https://example.com/forbidden').catch(() => document.body.dataset.blocked = 'yes');
    lambdaforgeReport.ready.then(async () => {
      const buffer = await lambdaforgeReport.load('catalogue');
      const rows = JSON.parse(new TextDecoder().decode(buffer));
      document.body.dataset.proteins = rows.length;
      document.querySelector('#status').textContent = 'All ' + rows.length + ' records available';
    });
    </script>""")
    records = [
        {
            "configuration": {"width": n, "pooling": "max"},
            "saved_seed": [4, 54, 167][n % 3],
            "protein": f"P{n}",
            "selection_reason": ["success", "error", "worst-localization"][n % 3],
            "metric": None,
            "coordinates": "coordinates",
            "features": "features",
        }
        for n in range(206)
    ]
    bundle.asset("catalogue", filename="catalogue.json", media_type="application/json").write_json(
        records
    )
    coordinates = bundle.asset("coordinates", filename="proteins/surface.f32")

    def write_coordinates(path):
        with path.open("wb") as stream:
            stream.write(struct.pack("<4f", 1.25, -2.5, float("nan"), 0.125))
            if large:
                stream.seek(70 * 1024 * 1024 - 4)
                stream.write(struct.pack("<f", 4.75))

    coordinates.build(write_coordinates)
    bundle.asset("features", filename="channels/features.f32").write_bytes(
        struct.pack("<3f", 0.1, -3.0, float("inf"))
    )
    bundle.asset("validity", filename="channels/validity.u8").write_bytes(bytes([1, 0, 1]))
    outputs.finalize()
    result = {
        "execution_id": root.name,
        "run_id": "run-one",
        "attempt_id": "attempt-0001",
        "run_dir": str(run_dir),
        "seed": 54,
        "parameters": {"width": 32},
        "artifacts": [a.to_dict() for a in outputs.artifacts],
        "outputs": dict(outputs.values),
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return result, records, coordinates.path


def request(viewer, index, identifier, **headers):
    url = viewer.url.replace("report.html", f".assets/{index}/{identifier}")
    return urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers={
                "X-LambdaForge-Report": viewer.token,
                **headers,
            },
        ),
        timeout=10,
    )


def test_large_assets_are_not_embedded_eagerly_and_exact_arrays_round_trip(tmp_path, monkeypatch):
    result, records, coordinates = make_bundle(tmp_path, large=True)
    # Metadata/report generation must never open this >16/>64 MiB data file.
    original = Path.open
    opened = []

    def track(path, *args, **kwargs):
        if path == coordinates:
            opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    sections = read_html_sections([result], tmp_path)
    report = write_html(sample_analysis(), tmp_path / "report.html", sections=sections)
    assert not opened
    assert report.stat().st_size < 16 * 1024 * 1024
    assert coordinates.stat().st_size > 64 * 1024 * 1024
    viewer = ReportViewer(report)
    viewer.start()
    try:
        assert not opened
        with request(viewer, 0, "catalogue") as response:
            assert json.load(response) == records
        assert not opened
        with request(viewer, 0, "coordinates") as response:
            digest, count = hashlib.sha256(), 0
            first = response.read(16)
            assert first == struct.pack("<4f", 1.25, -2.5, float("nan"), 0.125)
            digest.update(first)
            count += len(first)
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                count += len(chunk)
        expected = sections[0]["bundle"]["manifest"]["assets"][1]
        assert count == 70 * 1024 * 1024 and digest.hexdigest() == expected["content_sha256"]
        with request(viewer, 0, "features") as response:
            assert response.read() == struct.pack("<3f", 0.1, -3.0, float("inf"))
        with request(viewer, 0, "validity") as response:
            assert response.read() == bytes([1, 0, 1])
    finally:
        viewer.close()


@pytest.mark.parametrize(
    "filename",
    ["../secret", "/secret", "a/../../secret", "a\\secret", "https://example.com/data", "a//b"],
)
def test_unsafe_bundle_names_are_rejected(tmp_path, filename):
    outputs = OutputCollection(SimpleNamespace(run_dir=tmp_path))
    bundle = outputs.html_bundle("report")
    with pytest.raises(ReportBundleError):
        bundle.asset("protein", filename=filename)
    bundle.asset("protein", filename="data.bin")
    with pytest.raises(ReportBundleError, match="Duplicate"):
        bundle.asset("PROTEIN", filename="other.bin")
    with pytest.raises(ReportBundleError, match="Duplicate"):
        bundle.asset("other", filename="DATA.BIN")


def test_presentation_finalization_problem_is_visible_without_failing_science(tmp_path):
    outputs = OutputCollection(SimpleNamespace(run_dir=tmp_path))
    bundle = outputs.html_bundle("gallery")
    bundle.entry.write_text("<h1>Retained view</h1>")
    bundle.asset("missing", filename="missing.bin")
    outputs.file("scientific", filename="result.json").write_json({"metric": 0.7})
    outputs.finalize()
    assert outputs.values["gallery"]["presentation_error"]["type"] == "FileNotFoundError"
    assert [a.name for a in outputs.artifacts] == ["scientific"]
    assert bundle.entry.path.exists()
    sections = read_html_sections(
        [{"run_dir": str(tmp_path), "outputs": dict(outputs.values), "artifacts": []}], tmp_path
    )
    assert "FileNotFoundError" in sections[0]["presentation_error"]


@pytest.mark.parametrize("damage", ["corrupt", "missing", "symlink", "cross-run", "manifest"])
def test_scoped_integrity_missing_and_corruption_errors_are_explicit(tmp_path, damage):
    result, _, coordinates = make_bundle(tmp_path)
    sections = read_html_sections([result], tmp_path)
    report = write_html(sample_analysis(), tmp_path / "report.html", sections=sections)
    if damage == "corrupt":
        coordinates.write_bytes(b"wrong")
    elif damage == "missing":
        coordinates.unlink()
    elif damage == "symlink":
        coordinates.unlink()
        coordinates.symlink_to(tmp_path / "unrelated")
    elif damage == "manifest":
        manifest = coordinates.parent.parent / "lambdaforge-report.json"
        manifest.write_text("{}")
    else:
        sections[0]["bundle"]["source"]["artifact"] = "unowned"
        write_html(sample_analysis(), report, sections=sections)
    viewer = ReportViewer(report)
    viewer.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as failed:
            request(viewer, 0, "coordinates")
        body = json.load(failed.value)
        assert body["code"] in {"ASSET_UNAVAILABLE", "MISSING_ASSET"}
        assert "message" in body and b"wrong" not in json.dumps(body).encode()
    finally:
        viewer.close()


def test_host_origin_undeclared_assets_and_active_view_limits(tmp_path):
    result, _, _ = make_bundle(tmp_path)
    report = write_html(
        sample_analysis(), tmp_path / "report.html", sections=read_html_sections([result], tmp_path)
    )
    viewer = ReportViewer(report, max_asset_bytes=8)
    viewer.start()
    try:
        for identifier, headers in (
            ("coordinates", {"Origin": "https://evil.invalid"}),
            ("coordinates", {"Host": "evil.invalid"}),
            ("undeclared", {}),
        ):
            with pytest.raises(urllib.error.HTTPError):
                request(viewer, 0, identifier, **headers)
        with pytest.raises(urllib.error.HTTPError) as large:
            request(viewer, 0, "coordinates")
        assert "--max-asset-mib" in json.load(large.value)["message"]
    finally:
        viewer.close()


def test_compressed_native_transport_reads_only_selected_asset_and_cleans_up(tmp_path):
    from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.controlplane.jobs import JobRecord, JobState
    from lambdaforge.controlplane.JobService import JobService
    from lambdaforge.controlplane.JobStore import JobStore

    owner = tmp_path / "job-loopback"
    result, _, coordinates = make_bundle(owner / "work/execution")
    source = read_html_sections([result], owner)[0]["bundle"]["source"]
    jobs = JobService(
        ClusterCatalog({"local": ClusterProfile("local")}), JobStore(tmp_path / "store")
    )
    jobs.store.write(
        JobRecord(
            "job-loopback",
            "local",
            "local",
            None,
            JobState.SUCCEEDED,
            ("not-executed",),
            str(owner / "work"),
            {},
            "now",
            "now",
        )
    )
    target = tmp_path / "transferred"
    remote_asset(
        jobs,
        "job-loopback",
        source,
        "coordinates",
        target,
        expected_size=coordinates.stat().st_size,
    )
    assert target.read_bytes() == coordinates.read_bytes()
    with pytest.raises((RuntimeError, EOFError, ValueError)):
        remote_asset(jobs, "job-loopback", source, "undeclared", tmp_path / "bad")
    source["owner"] = str(tmp_path / "different-job")
    with pytest.raises(ValueError, match="ownership"):
        remote_asset(jobs, "job-loopback", source, "coordinates", target)


def test_browser_scoped_lazy_loading_disposal_and_external_network_blocked(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    result, records, _ = make_bundle(tmp_path, large=True)
    report = write_html(
        sample_analysis(), tmp_path / "report.html", sections=read_html_sections([result], tmp_path)
    )
    viewer = ReportViewer(report)
    viewer.start()
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            viewer.close()
            pytest.skip("Optional Chromium unavailable")
        browser = runtime.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            requested, errors = [], []
            page.on("request", lambda req: requested.append(req.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(viewer.url)
            assert not any(".assets/" in url for url in requested)
            page.get_by_role("button", name="Proteins", exact=True).click()
            frame = page.frame_locator("iframe")
            frame.locator(f'body[data-proteins="{len(records)}"]').wait_for()
            assert not any(url.endswith("coordinates") for url in requested)
            frame.get_by_role("button", name="Coordinates", exact=True).click()
            frame.locator('body[data-loaded="coordinates"]').wait_for()
            assert frame.locator("body").get_attribute("data-bytes") == str(70 * 1024 * 1024)
            assert frame.locator("body").get_attribute("data-isolated") == "yes"
            frame.locator('body[data-blocked="yes"]').wait_for()
            assert not any("example.com" in url for url in requested)
            assert page.locator("body").get_attribute("data-escaped") is None
            page.screenshot(path=str(tmp_path / "managed-protein-tab.png"), full_page=True)
            page.get_by_role("button", name="Overview", exact=True).click()
            assert page.locator("iframe").get_attribute("srcdoc") is None
            assert not errors
        finally:
            browser.close()
            viewer.close()


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_native_work_retains_science_and_exports_portable_bundle_without_reexecution(
    tmp_path, monkeypatch, broken, remote
):
    import shutil

    from lambdaforge.work import ResultStore, WorkConfig, WorkRunner

    producer = tmp_path / "job-reader/work"
    producer.mkdir(parents=True)
    (producer / "pyproject.toml").write_text('[project]\nname="report-test"\nversion="1"\n')
    monkeypatch.chdir(producer)
    config = WorkConfig.from_mapping(
        {
            "name": "retained-report",
            "run": "tests.report_bundle_cases.ManagedReportWork",
            "with": {"broken": broken},
        },
        source=producer / "work.yaml",
    )
    execution = WorkRunner().run(config)
    assert execution.status == "succeeded"
    store = ResultStore()
    from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobService, JobStore
    from lambdaforge.controlplane.jobs import JobRecord, JobState

    job_owner = producer.parent
    (job_owner / "result.json").write_bytes((execution.execution_dir / "result.json").read_bytes())
    import sys

    from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
    from lambdaforge.controlplane.LocalTransport import LocalTransport

    class LoopbackFactory(ControlPlaneFactory):
        def transport(self, profile):
            return LocalTransport()

    profile = ClusterProfile(
        "fixture",
        transport="ssh" if remote else "local",
        host="fixture.invalid" if remote else None,
        python=sys.executable,
        workspace=str(tmp_path),
    )
    jobs = JobService(
        ClusterCatalog({"fixture": profile}), JobStore(tmp_path / "job-index"), LoopbackFactory()
    )
    jobs.store.write(
        JobRecord(
            "job-reader",
            "fixture",
            "local",
            None,
            JobState.SUCCEEDED,
            (),
            str(job_owner / "work"),
            {},
            "now",
            "now",
            config_path=str(producer / "work.yaml"),
            metadata={"source_config_path": str(producer / "work.yaml")},
        )
    )
    native = jobs.work_result_view("job-reader", view="html")
    assert native["sections"]
    if not broken:
        assert native["sections"][0]["bundle"]["source"]["job_id"] == "job-reader"
        active_report = write_html(
            sample_analysis(), tmp_path / "native.html", sections=native["sections"]
        )
        viewer = ReportViewer(active_report, jobs=jobs)
        viewer.start()
        try:
            with request(viewer, 0, "coordinates") as response:
                assert response.read() == b"exact retained coordinates"
        finally:
            viewer.close()
    exported = store.export(execution.execution_id, tmp_path / "exports")
    package = Path(exported["path"])
    report = package / "reports/work-results.html"
    assert report.is_file()
    if broken:
        assert "FileNotFoundError" in report.read_text()
        return
    # Moving the package must not depend on the original machine/temporary export paths.
    relocated = tmp_path / "moved"
    shutil.move(str(package), relocated)
    store.delete(execution.execution_id, apply=True)
    viewer = ReportViewer(relocated / "reports/work-results.html")
    viewer.start()
    try:
        with request(viewer, 0, "coordinates") as response:
            assert response.read() == b"exact retained coordinates"
    finally:
        viewer.close()

    consumer = ResultStore(tmp_path / "consumer/runs")

    def forbidden(*args, **kwargs):
        raise AssertionError("Import/regeneration must not execute science")

    monkeypatch.setattr(WorkRunner, "run", forbidden)
    imported = consumer.import_export(relocated, apply=True)
    assert imported["status"] == "imported"
    regenerated = consumer.report(execution.execution_id, tmp_path / "imported.html")
    viewer = ReportViewer(regenerated)
    viewer.start()
    try:
        with request(viewer, 0, "coordinates") as response:
            assert response.read() == b"exact retained coordinates"
    finally:
        viewer.close()


def test_manifest_schema_and_native_cli_contract(tmp_path):
    from importlib.resources import files

    from lambdaforge.cli.parser import build_parser

    result, _, _ = make_bundle(tmp_path)
    manifest = read_html_sections([result], tmp_path)[0]["bundle"]["manifest"]
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(files("lambdaforge").joinpath("schemas/report-bundle-v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)
    arguments = build_parser().parse_args(
        ["reports", "serve", "report.html", "--max-asset-mib", "512"]
    )
    assert arguments.command == "reports"


@pytest.mark.parametrize("timeout", [False, True])
def test_remote_reader_cancellation_or_deadline_terminates_only_reader(tmp_path, timeout):
    import threading
    import time
    from contextlib import contextmanager

    from lambdaforge.controlplane.ClusterProfile import ClusterProfile

    ended = threading.Event()

    class Owner:
        def terminate(self):
            ended.set()

    class Stalled:
        def read(self, count=-1):
            assert ended.wait(2), "Native read must have a bounded cancellation/deadline"
            raise InterruptedError("Reader stopped")

    @contextmanager
    def stream(command):
        assert "-c" in command  # Read-only asset script, not a Work/claim launcher.
        yield SimpleNamespace(owner=Owner(), stdout=Stalled(), wait=lambda: None)

    owner = tmp_path / "job-fixture"
    source = {"owner": str(owner / "work/execution")}
    jobs = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(
            cluster="local", work_dir=str(owner / "work"), command=(), metadata={}
        ),
        catalog=SimpleNamespace(get=lambda key: ClusterProfile("local")),
        factory=SimpleNamespace(transport=lambda profile: SimpleNamespace(stream=stream)),
    )
    started = time.monotonic()
    with pytest.raises(InterruptedError):
        remote_asset(
            jobs,
            "job-fixture",
            source,
            "selected",
            tmp_path / "target",
            cancel=lambda: not timeout,
            timeout_seconds=0.15,
        )
    assert ended.is_set() and time.monotonic() - started < 2
    assert not [t for t in threading.enumerate() if t.name == "lf-report-read-deadline"]
