"""Offline candidate plots use persisted evidence, not epochs or new HPO calculations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.analysis.Report import write_html


def select_evidence(page: Any, selector: str, value: str) -> None:
    """Exercise the semantic picker, rather than restoring deleted legacy selects."""
    page.click(selector + "-picker")
    name = value
    if selector == "#parameter-select":
        name = "param:" + value
    if name.startswith("metric:") and not name.startswith("metric:__"):
        name = name.removeprefix("metric:")
    if name in {"__selection__", "metric:__selection__"}:
        name = "selection_objective"
    query = (
        name.removeprefix("param:")
        .removeprefix("metric:")
        .replace("__best_observed__", "Best observed")
        .replace("__current_observed__", "Current observed")
    )
    page.fill("#research-global-query", "" if name == "selection_objective" else query)
    page.locator(f'#research-search-results button[data-choice="{name}"]').click()


def sample_analysis() -> dict[str, Any]:
    candidates = []
    for trial, width, depth, flag, score in (
        (1, 32, 1, True, 0.6),
        (2, 32, 1, False, 0.8),
        (3, 64, 2, True, 0.75),
        (4, 128, 1, False, None),
    ):
        candidates.append(
            {
                "trial": trial,
                "parameters": {"width": width, "depth": depth, "flag": flag},
                "mean": score,
                "n": 1 if score is not None else 0,
                "state": "completed" if score is not None else "pruned",
                "best_observed_objective": score if score is not None else 0.91,
                "diagnostic_metrics": {"accuracy": {"mean": score}},
                "runs": [
                    {
                        "state": "succeeded" if score is not None else "pruned",
                        "phase": "search",
                        "censored": score is None,
                        "current_observed_objective": score if score is not None else 0.88,
                        "metrics": {"train_loss": 0.0 if trial == 1 else trial / 10},
                    }
                ],
            }
        )
    return {
        "source": {"status": "final"},
        "objective": {"metric": "score", "mode": "max"},
        "search_space": {
            "width": {"values": [32, 64, 128]},
            "depth": {"values": [1, 2]},
            "flag": {"values": [True, False]},
        },
        "candidates": candidates,
    }


def test_observed_dashboard_embeds_all_numeric_metrics_and_local_renderer(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    source = sample_analysis()
    before = json.dumps(source, sort_keys=True)
    output = write_html(source, tmp_path / "observed.html").read_text()
    encoded = re.search(
        r'<script id="lf-study-data" type="application/json">(.*?)</script>', output
    )
    assert encoded is not None
    payload = json.loads(encoded.group(1))
    assert payload["metrics"] == ["accuracy", "train_loss"]
    assert payload["parameter_space"]["width"]["values"] == [32, 64, 128]
    assert "Observed heatmap" in output
    assert "Observed 3D surface" in output
    assert "3D scatter" in output
    assert "Compare with…" in output
    assert "window.LambdaForgeStudyCharts" in output
    assert "65536" in output  # Bounded observed grid, not an unbounded Cartesian allocation.
    assert "never model predictions" in output
    assert "predictions; untested combinations stay blank" in output
    assert json.dumps(source, sort_keys=True) == before


def test_observed_charts_in_browser_and_saved_preferences(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    report = write_html(sample_analysis(), tmp_path / "study.html")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Install the optional Playwright Chromium browser to run this UI test.")
        browser = runtime.chromium.launch(
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        page = browser.new_page(viewport={"width": 1400, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(report.as_uri())
        page.get_by_role("button", name="Explore", exact=True).click()
        page.get_by_text("Advanced visualization options", exact=True).click()

        # Editing axes renders immediately; Save is persistence, never a drawing prerequisite.
        select_evidence(page, "#study-chart-x", "param:width")
        select_evidence(page, "#study-chart-y", "metric:train_loss")
        page.wait_for_function(
            """() => {const c = document.querySelector('#study-custom-chart .js-plotly-plot');
            return c.data[0]?.x?.length === 3 && c.data[0].y[0] === 0;}"""
        )
        assert page.locator(".saved-study-chart").count() == 0

        def create(kind: str, x: str, y: str, z: str = "metric:__selection__") -> None:
            page.select_option("#study-chart-kind", kind)
            select_evidence(page, "#study-chart-x", x)
            select_evidence(page, "#study-chart-y", y)
            if kind in {"scatter3d", "heatmap", "surface"}:
                select_evidence(page, "#study-chart-z", z)
            page.click("#save-study-chart")

        def trace() -> dict[str, Any]:
            return page.eval_on_selector(
                "#study-custom-chart .js-plotly-plot", "chart => chart.data[0]"
            )

        create("scatter", "param:width", "metric:train_loss")
        assert trace()["x"] == [32, 32, 64]
        assert trace()["y"] == [0.0, 0.2, 0.3]
        page.select_option("#study-chart-aggregate", "mean")
        page.get_by_role("button", name="Compare with…", exact=True).click()
        page.fill("#research-global-query", "train_loss")
        page.locator('#research-search-results button[data-choice="train_loss"]').click()
        assert "train loss" in page.locator("#research-compare-chips").inner_text()
        create("line", "param:width", "metric:__selection__")
        assert trace()["y"] == [0.7, 0.75]
        assert trace()["customdata"][0] == ["1, 2", 2]
        assert (
            page.eval_on_selector(
                "#study-custom-chart .js-plotly-plot", "chart => chart.data.length"
            )
            == 2
        )

        create("heatmap", "param:width", "param:depth")
        assert trace()["type"] == "heatmap"
        assert trace()["x"] == [32, 64, 128]
        assert trace()["y"] == [1, 2]
        assert trace()["z"] == [[0.7, None, None], [None, 0.75, None]]
        page.select_option("#study-chart-palette", "Purple ↔ green")
        create("heatmap", "param:flag", "param:depth", "metric:train_loss")
        assert trace()["x"] == ["false", "true"]
        assert trace()["z"] == [[0.2, 0.0], [None, 0.3]]

        create("scatter3d", "param:flag", "param:width", "metric:accuracy")
        assert trace()["type"] == "scatter3d"
        assert trace()["z"] == [0.6, 0.8, 0.75]
        assert page.eval_on_selector(
            "#study-custom-chart .js-plotly-plot", "chart => chart.layout.scene.xaxis.ticktext"
        ) == ["false", "true"]
        create("surface", "param:width", "param:depth")
        assert trace()["type"] == "surface"
        assert trace()["z"][0] == [0.7, None, None]

        page.check("#study-chart-partial")
        create("scatter", "param:width", "metric:__best_observed__")
        assert 0.91 in trace()["y"]
        assert "x" in trace()["marker"]["symbol"]
        create("scatter", "param:width", "metric:__selection__")
        assert len(trace()["x"]) == 2  # Grouped completed values only; censored final is absent.
        assert "Partial/censored" in page.locator("#study-chart-status").inner_text()
        page.reload()
        page.wait_for_function("document.querySelector('#study-custom:not([hidden])') !== null")
        assert page.locator(".saved-study-chart").count() == 8
        assert trace()["y"] == [0.7, 0.75]
        page.wait_for_function(
            """() => {const chart = document.querySelector('#study-custom-chart .js-plotly-plot');
            return chart._fullLayout.width === chart.clientWidth;}"""
        )
        page.screenshot(path=str(tmp_path / "observed-study-charts.png"), full_page=True)

        page.get_by_role("button", name="Parameters", exact=True).click()
        select_evidence(page, "#parameter-select", "width")
        select_evidence(page, "#parameter-metric", "train_loss")
        parameter = page.eval_on_selector(
            "#parameter-observed .js-plotly-plot", "chart => chart.data[0]"
        )
        assert parameter["x"] == [32, 64]
        assert parameter["y"] == [0.1, 0.3]
        assert "train loss" in page.locator("#parameter-value-metric").inner_text().lower()
        select_evidence(page, "#parameter-select", "flag")
        assert "false" in page.locator("#parameter-values tbody").inner_text()
        page.reload()
        assert page.locator("#parameter-metric").input_value() == "train_loss"
        assert errors == []
        browser.close()


def test_missing_selection_is_explained_without_substituting_partial_evidence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    source = sample_analysis()
    for candidate in source["candidates"]:
        candidate.update(mean=None, n=0, state="pruned")
    report = write_html(source, tmp_path / "partial.html")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Playwright Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page()
        page.goto(report.as_uri())
        page.get_by_role("button", name="Explore", exact=True).click()
        page.get_by_text("Advanced visualization options", exact=True).click()
        assert "No final selection objective" in page.locator("#study-chart-status").inner_text()
        select_evidence(page, "#study-chart-y", "metric:__best_observed__")
        page.check("#study-chart-partial")
        page.wait_for_function(
            """() => {
            const chart = document.querySelector('#study-custom-chart .js-plotly-plot');
            return chart.data[0].x.length === 4;
            }"""
        )
        select_evidence(page, "#study-chart-y", "metric:__selection__")
        page.wait_for_function(
            """() => {
            const chart = document.querySelector('#study-custom-chart .js-plotly-plot');
            return chart.data[0].x.length === 0;
            }"""
        )
        assert "No final selection objective" in page.locator("#study-chart-status").inner_text()
        browser.close()
