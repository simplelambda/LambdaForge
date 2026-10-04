"""Browser regressions for the Study report, independent of execution and HPO policy."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.analysis.Report import write_html
from tests.analysis.test_observed_study_charts import sample_analysis


def presentation_evidence() -> dict[str, Any]:
    document = sample_analysis()
    document["search_space"].update(
        {
            "pooling": {"values": ["max", "lse"]},
            "temperature": {"values": [1, 2], "when": {"pooling": "lse"}},
        }
    )
    for index, candidate in enumerate(document["candidates"]):
        candidate["parameters"]["pooling"] = "lse" if index == 0 else "max"
        if index == 0:
            candidate["parameters"]["temperature"] = 1
    document["parameter_importance"] = {
        "width": {"importance": 0.6, "support": 3, "reliability": "low"},
        "temperature": {"importance": 0.2, "support": 1, "reliability": "low"},
    }
    document["scientific_understanding"] = {
        "seed_noise_model": {"status": "unresolved", "repeated_candidates": 0},
        "parameter_questions": [
            {
                "parameter": "width",
                "conclusion_kind": "UNRESOLVED",
                "summary": "The question remains unresolved.",
                "descriptive_stability": 0.3,
                "response_support": {"direct": [32, 64], "censored_only": [128]},
                "stability_diagnostics": {"realizations": 256, "monte_carlo_resolution": 1 / 256},
                "missing_evidence": [],
            }
            for _ in range(20)
        ],
    }
    return document


def test_overview_dropdowns_overlay_localization_and_identity(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    document = presentation_evidence()
    original = copy.deepcopy(document)
    report = write_html(document, tmp_path / "study.html")
    assert document == original  # Presentation must not reclassify the persisted evidence.
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.set_default_timeout(8000)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        assert page.locator(".tab").first.inner_text() == "Overview"
        assert page.locator("#study-overview").is_visible()
        assert not page.locator("#study-ranking").is_visible()
        page.wait_for_function("document.querySelector('#overview-states').data?.length > 0")
        states = page.eval_on_selector("#overview-states", "c => c.data[0]")
        assert states["y"][:2] == [3, 1]
        importance = page.eval_on_selector("#overview-importance .js-plotly-plot", "c => c.data[0]")
        assert "temperature" not in importance["x"]
        assert 'pooling = "lse"' in page.locator("#overview-conditional-table").inner_text()
        assert "1 / 4" in page.locator("#overview-conditional-table").inner_text()
        assert "0.2" in page.locator("#overview-conditional-table").inner_text()
        assert "responsibility" in page.locator("#study-overview").inner_text()
        strip = page.locator("#overview-interpretation .card-carousel")
        assert strip.evaluate("n => n.scrollWidth > n.clientWidth")
        assert strip.evaluate("n => n.offsetHeight") < 500
        assert "Monte Carlo" not in page.locator("#overview-interpretation").inner_text()
        page.click("#interpretation-help")
        assert "not the probability" in page.locator("#research-detail-body").inner_text()
        page.click("#research-detail-close")
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(tmp_path / "overview-en.png"), full_page=True)

        page.get_by_role("button", name="Parameters", exact=True).click()
        page.click("#parameter-select-picker")
        page.fill("#study-dropdown-query", "width")
        page.locator('#study-dropdown-options input[data-choice="param:width"]').check()
        page.click("#parameter-metric-picker")
        assert not page.locator("#research-search").is_visible()
        page.fill("#study-dropdown-query", "accuracy")
        page.locator('#study-dropdown-options input[data-choice="accuracy"]').check()
        assert page.locator("#study-dropdown").is_visible()
        page.fill("#study-dropdown-query", "train_loss")
        page.locator('#study-dropdown-options input[data-choice="train_loss"]').check()
        page.wait_for_function(
            "document.querySelector('#parameter-observed .js-plotly-plot').data.length===3"
        )
        graph = page.eval_on_selector("#parameter-observed .js-plotly-plot", "c => c.layout")
        assert not graph.get("grid", {}).get("rows")  # One chart, not three stacked charts.
        assert graph["yaxis2"]["overlaying"] == "y"
        page.fill("#study-dropdown-query", "accuracy")
        assert page.locator('#study-dropdown-options input[data-choice="accuracy"]').is_checked()
        page.locator('#study-dropdown-options input[data-choice="accuracy"]').uncheck()
        page.keyboard.press("Escape")
        page.wait_for_function(
            "document.querySelector('#parameter-observed .js-plotly-plot').data.length===2"
        )
        page.screenshot(path=str(tmp_path / "parameter-overlay-en.png"), full_page=True)

        page.select_option("#study-language", "es")
        assert page.locator("html").get_attribute("lang") == "es"
        assert page.locator(".tab").first.inner_text() == "Resumen"
        page.click("#parameter-metric-picker")
        assert (
            page.locator("#study-dropdown-query").get_attribute("placeholder").startswith("Buscar")
        )
        page.keyboard.press("Escape")
        assert page.locator("#parameter-metric").input_value() == "__selection__"
        page.get_by_role("button", name="Resumen", exact=True).click()
        assert "activo cuando" in page.locator("#overview-conditional-table").inner_text().lower()
        assert 'pooling = "lse"' in page.locator("#overview-conditional-table").inner_text()
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=str(tmp_path / "overview-es.png"), full_page=True)
        page.reload()
        assert page.locator("html").get_attribute("lang") == "es"
        assert page.evaluate("window.lfResearchServices.getPrefs().parameterMetrics") == [
            "metric:train_loss"
        ]
        page.get_by_role("button", name="Cobertura", exact=True).click()
        page.wait_for_function(
            "document.querySelector('#study-coverage .js-plotly-plot').layout.title.text"
            "==='Cobertura marginal observada'"
        )
        page.select_option("#study-language", "en")
        assert page.locator(".tab").first.inner_text() == "Overview"
        assert page.evaluate("window.lfResearchServices.data.candidates") == original["candidates"]
        assert errors == []
        browser.close()


def test_dropdown_zero_selection_keyboard_and_mobile(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    report = write_html(sample_analysis(), tmp_path / "small.html")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 430, "height": 900})
        page.set_default_timeout(8000)
        page.goto(report.as_uri())
        page.get_by_role("button", name="Parameters", exact=True).click()
        page.click("#parameter-metric-picker")
        page.locator('#study-dropdown-options input[data-choice="selection_objective"]').uncheck()
        assert "No metrics selected" in page.locator("#parameter-observed-status").inner_text()
        page.fill("#study-dropdown-query", "accuracy")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Space")
        page.keyboard.press("Escape")
        page.wait_for_function(
            "document.querySelector('#parameter-observed .js-plotly-plot').data[0]?.y?.length>0"
        )
        assert page.locator("#parameter-metric").input_value() == "accuracy"
        page.click("#parameter-metric-picker")
        bounds = page.locator("#study-dropdown").bounding_box()
        assert bounds and bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= 430
        page.screenshot(path=str(tmp_path / "mobile-dropdown.png"), full_page=True)
        page.keyboard.press("Escape")
        page.select_option("#study-language", "es")
        page.get_by_role("button", name="Explorar", exact=True).click()
        page.get_by_text("Opciones avanzadas de visualización", exact=True).click()
        page.click("#study-chart-palette-picker")
        page.fill("#study-dropdown-query", "rojo")
        page.locator('#study-dropdown-options input[data-choice="Blue ↔ red"]').check()
        assert page.locator("#study-chart-palette").input_value() == "Blue ↔ red"
        # Persisted catalog and numeric observation values are not translated.
        assert (
            json.loads(page.locator("#lf-study-data").text_content() or "{}")["candidates"]
            == sample_analysis()["candidates"]
        )
        browser.close()
