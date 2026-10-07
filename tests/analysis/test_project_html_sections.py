"""Project HTML is explicit, finalized, portable presentation, never HPO input."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.analysis.Report import write_html
from lambdaforge.study_projection import projection_page, read_html_sections
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.ResultStore import ResultStore
from tests.analysis.test_observed_study_charts import sample_analysis


def declared_run(root: Path, content: str = "<h1>Protein α</h1>") -> dict:
    run_dir = root / "runs" / "run-1" / "attempts" / "attempt-0001"
    run_dir.mkdir(parents=True)
    outputs = OutputCollection(SimpleNamespace(run_dir=run_dir))
    report = outputs.html_section("prediction", section="proteins", title="Proteins")
    report.write_text(content)
    outputs.finalize()
    result = {
        "execution_id": root.name,
        "run_id": "run-1",
        "attempt_id": "attempt-0001",
        "run_dir": str(run_dir),
        "trial": {"index": 1},
        "seed": 7,
        "artifacts": [artifact.to_dict() for artifact in outputs.artifacts],
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return result


def test_managed_section_and_checksum_guards(tmp_path: Path) -> None:
    run = declared_run(tmp_path)
    sections = read_html_sections([run], tmp_path)
    assert sections == [
        {
            "name": "proteins",
            "title": "Proteins",
            "label": "Trial 1 · seed 7 · prediction",
            "html": "<h1>Protein α</h1>",
        }
    ]
    path = Path(run["run_dir"]) / run["artifacts"][0]["path"]
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        read_html_sections([run], tmp_path)
    with pytest.raises(ValueError, match="outside"):
        read_html_sections([run], tmp_path / "different-owner")
    path.unlink()
    path.symlink_to(tmp_path / "secret.html")
    with pytest.raises(ValueError, match="Symlinked"):
        read_html_sections([run], tmp_path)


def test_invalid_declarations_and_html_size(tmp_path: Path) -> None:
    outputs = OutputCollection(SimpleNamespace(run_dir=tmp_path))
    with pytest.raises(ValueError, match="title"):
        outputs.html_section("report", title=" " * 121)
    with pytest.raises(ValueError, match="name"):
        write_html(sample_analysis(), tmp_path / "bad.html", sections=[{"name": ""}])
    with pytest.raises(ValueError, match="16 MiB"):
        write_html(
            sample_analysis(),
            tmp_path / "large.html",
            sections=[{"name": "big", "title": "Big", "html": "x" * (16 * 1024 * 1024 + 1)}],
        )
    run = declared_run(tmp_path)
    run["artifacts"][0]["path"] = "../../secret.html"
    with pytest.raises(ValueError, match="Unsafe"):
        read_html_sections([run], tmp_path)


def test_native_results_report_collects_registered_sections(tmp_path: Path) -> None:
    run = declared_run(tmp_path)
    selected = {"_manifest_path": str(tmp_path / "result.json"), "runs": [run]}
    store = SimpleNamespace(
        select=lambda _selector: selected,
        _evidence_dir=lambda _manifest: tmp_path,
        analysis=lambda _selector, **_kwargs: sample_analysis(),
    )
    report = ResultStore.report(store, "study", tmp_path / "native.html")
    assert 'data-target="project-section-0"' in report.read_text()
    assert "Trial 1 · seed 7 · prediction" in report.read_text()


def test_export_reconnects_exact_owned_run_without_using_remote_paths(tmp_path: Path) -> None:
    run = declared_run(tmp_path / "execution-export")
    run["run_dir"] = "/remote/execution-export/runs/run-1/attempts/attempt-0001"
    sections = read_html_sections([run], tmp_path / "execution-export", relocate=True)
    assert sections[0]["html"] == "<h1>Protein α</h1>"
    run["attempt_id"] = "attempt-0002"
    with pytest.raises(ValueError, match="exact exported"):
        read_html_sections([run], tmp_path / "execution-export", relocate=True)
    # Ordinary archived Runs must remain exportable and never cause remote file reads.
    assert read_html_sections([dict(run, artifacts=[])], tmp_path, relocate=True) == []


def test_remote_report_only_sections_are_paged_and_cache_is_reused(tmp_path: Path) -> None:
    run = declared_run(tmp_path, "<pre>" + "α" * 400_000 + "</pre>")
    study = tmp_path / "study"
    study.mkdir()
    (study / "summary.json").write_text(
        json.dumps({"candidates": [{"trial": 1, "runs": [{"run_dir": run["run_dir"], "seed": 7}]}]})
    )
    first = projection_page(str(study), {"view": "report-sections"})
    assert not first["eof"]
    content = base64.b64decode(first["data"])
    cache = study / "report-sections-read-cache.bin"
    before = cache.stat().st_mtime_ns
    offset = first["next_offset"]
    while True:
        page = projection_page(
            str(study),
            {"view": "report-sections", "offset": offset, "fingerprint": first["fingerprint"]},
        )
        content += base64.b64decode(page["data"])
        if page["eof"]:
            break
        offset = page["next_offset"]
    assert cache.stat().st_mtime_ns == before
    sections = json.loads(content)["sections"]
    assert sections[0]["label"] == "Trial 1 · seed 7 · prediction"
    assert sections[0]["html"].count("α") == 400_000
    # Only the explicit report view has HTML bytes; ordinary interactive reads stay compact.
    interactive = projection_page(str(study), {"view": "interactive"})
    assert "sections" not in json.loads(base64.b64decode(interactive["data"]))


def test_project_tabs_render_scripts_in_isolated_offline_frames(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    content = """<h1>Protein α</h1>
    <button onclick="location.href='https://example.com/navigation'">Navigate out</button><script>
    try { parent.document.body.dataset.escaped = 'yes'; }
    catch(e) { document.body.dataset.isolated = 'yes'; }
    fetch('https://example.com/secret').catch(() => document.body.dataset.networkBlocked = 'yes');
    </script>"""
    report = write_html(
        sample_analysis(),
        tmp_path / "sections.html",
        sections=[
            {"name": "proteins", "title": "Proteins", "html": content, "label": "Trial 1"},
            {
                "name": "proteins",
                "title": "Proteins",
                "html": "<h1>Trial 2</h1>",
                "label": "Trial 2",
            },
        ],
    )
    assert content not in report.read_text()  # Consumer markup is encoded, not parent DOM.
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
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(report.as_uri())
        assert page.locator("iframe").get_attribute("srcdoc") is None
        page.get_by_role("button", name="Proteins", exact=True).click()
        frame = page.frame_locator("iframe")
        assert frame.locator("h1").inner_text() == "Protein α"
        assert frame.locator("body").get_attribute("data-isolated") == "yes"
        frame.locator('body[data-network-blocked="yes"]').wait_for()
        assert page.locator("body").get_attribute("data-escaped") is None
        page.click("#project-section-0-document-picker")
        page.locator('#study-dropdown-options input[data-choice="1"]').check()
        assert frame.locator("h1").inner_text() == "Trial 2"
        page.reload()
        assert frame.locator("h1").inner_text() == "Trial 2"
        page.screenshot(path=str(tmp_path / "project-proteins-tab.png"), full_page=True)
        page.click("#project-section-0-document-picker")
        page.locator('#study-dropdown-options input[data-choice="0"]').check()
        frame.get_by_role("button", name="Navigate out").click()
        page.wait_for_timeout(200)
        assert not external  # Parent frame-src also blocks iframe navigation/exfiltration.
        assert not errors
        browser.close()
