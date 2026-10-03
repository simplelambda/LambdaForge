"""Generic research semantics and deterministic discovery, without consumer projects."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lambdaforge.analysis import AnalysisProfile
from lambdaforge.analysis.MetricCatalog import MetricCatalog, resolve_semantics
from lambdaforge.analysis.Report import write_html
from lambdaforge.analysis.ResearchAnalysis import ResearchAnalysis, candidate_values


def candidates(count: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "trial": i + 1,
            "parameters": {"width": i, "category": "a" if i % 2 else "b"},
            "mean": i / max(count, 1),
            "n": 2,
            "diagnostic_metrics": {
                "quality": {"mean": i / max(count, 1)},
                "duplicate": {"mean": 2 * i / max(count, 1)},
                "fixed": {"mean": 1.0},
                "small": {"mean": i * 1e-12},
            },
            "runs": [
                {
                    "seed": j,
                    "state": "succeeded",
                    "final_objective": i / max(count, 1),
                    "metrics": {"quality": i / max(count, 1)},
                }
                for j in (1, 2)
            ],
        }
        for i in range(count)
    ]


def semantics() -> dict[str, Any]:
    return resolve_semantics(
        {
            "metrics": {
                "quality": {
                    "label": "Quality",
                    "aliases": ["q"],
                    "category": "validation/quality",
                    "unit": "ratio",
                    "range": [0, 1],
                    "aggregation": "latest",
                    "visibility": "primary",
                },
                "duplicate": {"derived_from": ["quality"], "transformation": "double"},
                "fixed": {"expected_to_vary": True},
                "small": {"label": "Small signal", "unit": "ratio"},
            },
            "questions": [
                {
                    "id": "relationship",
                    "kind": "relationship",
                    "x": "quality",
                    "y": "duplicate",
                    "expected": "positive",
                }
            ],
            "discovery": {"resamples": 32, "max_pairs": 16},
        }
    )


def test_catalog_precedence_identity_patterns_and_lineage() -> None:
    declared = {
        "defaults": [{"pattern": "val_*", "metadata": {"category": "validation", "unit": "ratio"}}],
        "metrics": {"val_accuracy": {"label": "Default accuracy", "aggregation": "best"}},
    }
    profile = AnalysisProfile.resolve(
        declared, {"metrics": {"val_accuracy": {"label": "Accuracy"}}}
    )
    resolved = MetricCatalog.resolve(profile.document, {"val_accuracy"})
    assert resolved["metrics"]["val_accuracy"]["label"] == "Accuracy"
    assert resolved["metrics"]["val_accuracy"]["aggregation"] == "best"
    assert resolved["metrics"]["val_accuracy"]["unit"] == "ratio"
    assert (
        profile.identity
        == AnalysisProfile.resolve(
            declared, {"metrics": {"val_accuracy": {"label": "Accuracy"}}}
        ).identity
    )
    with pytest.raises(TypeError):
        profile.document["metrics"]["val_accuracy"]["label"] = "Changed"


@pytest.mark.parametrize(
    "declaration",
    [
        {"metrics": {"a": {"range": [1, 0]}}},
        {"metrics": {"a": {"aliases": ["same"]}, "b": {"aliases": ["same"]}}},
        {"metrics": {"a": {"derived_from": ["b"]}, "b": {"derived_from": ["a"]}}},
        {"metrics": {"a": {"derived_from": ["missing"]}}},
        {"questions": [{"id": "q", "kind": "relationship", "x": "unknown", "y": "missing"}]},
        {"discovery": {"resamples": 100000}},
        {"defaults": [{"pattern": "*", "metadata": {"aliases": ["ambiguous"]}}]},
        {
            "families": {
                "unsafe": {
                    "dimensions": {"x": {"values": [1, 2]}},
                    "template": "{x:1000000000}",
                }
            }
        },
    ],
)
def test_invalid_semantics_are_rejected(declaration: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        resolve_semantics(declaration)


def test_test_leakage_is_rejected_before_hpo() -> None:
    from lambdaforge.hpo.ObjectiveUtility import ObjectiveUtility

    composite = ObjectiveUtility.normalize(
        {"metrics": {"val_accuracy": {"weight": 1}, "test_accuracy": {"weight": 1}}}
    )
    with pytest.raises(ValueError, match="Test-split"):
        resolve_semantics(objective=composite)
    for name, declaration in (
        ("test_accuracy", {}),
        (
            "derived",
            {
                "metrics": {
                    "test_quality": {"split": "test"},
                    "derived": {"derived_from": ["test_quality"]},
                }
            },
        ),
    ):
        with pytest.raises(ValueError, match="Test-split"):
            resolve_semantics(declaration, objective={"metric": name, "mode": "max"})
    resolve_semantics({"metrics": {"test_accuracy": {"split": "test"}}})


def test_profiling_constants_small_scales_redundancy_and_determinism() -> None:
    source = candidates()
    original = copy.deepcopy(source)
    first = ResearchAnalysis.compute(
        source, fingerprint="fixed", semantics=semantics(), parameter_names=["width", "category"]
    )
    second = ResearchAnalysis.compute(
        source, fingerprint="fixed", semantics=semantics(), parameter_names=["width", "category"]
    )
    assert first == second
    assert source == original
    assert first["metric_profiles"]["fixed"]["constant"]
    assert not first["metric_profiles"]["small"]["near_constant"]
    assert first["metric_profiles"]["quality"]["seed_support"] == 2
    assert first["metric_profiles"]["selection_objective"]["run_support"] == 48
    assert any("duplicate" in group["members"] for group in first["redundancy_groups"])
    assert any(f["kind"] == "invariant" for f in first["findings"])
    structural = next(f for f in first["findings"] if f["id"] == "association:quality:duplicate")
    assert structural["structural_relationship"]
    assert structural["reliability"]["adjusted_p"] >= structural["evidence"]["p_value"]
    json.dumps(first, allow_nan=False)


def test_missing_partial_and_live_test_metrics_do_not_become_exact_evidence() -> None:
    source = candidates(5)
    source[0].update(mean=None, censored_observations=1)
    assert candidate_values(source[0]) == {}
    source[1]["diagnostic_metrics"]["quality"] = {"mean": float("nan")}
    for c in source:
        c["diagnostic_metrics"]["test_accuracy"] = {"mean": c["trial"] / 5}
    declaration = resolve_semantics()
    live = ResearchAnalysis.compute(
        source,
        fingerprint="partial",
        semantics=declaration,
        status="provisional",
        parameter_names=["width"],
    )
    assert "test_accuracy" in live["metric_profiles"]
    assert not any("test_accuracy" in (r["x"], r["y"]) for r in live["relationships"])
    assert live["metric_profiles"]["selection_objective"]["finite_candidates"] == 4


def test_family_template_preserves_order_categories_and_members() -> None:
    profile = resolve_semantics(
        {
            "families": {
                "efficiency": {
                    "dimensions": {
                        "strategy": {"kind": "categorical", "values": ["fast", "safe"]},
                        "fraction": {"kind": "ordered", "values": [10, 25, 100]},
                    },
                    "template": "score_{strategy}_{fraction}",
                }
            }
        }
    )
    assert len(profile["families"]["efficiency"]["members"]) == 6
    rows = candidates(4)
    for c in rows:
        c["diagnostic_metrics"].update({m: {"mean": c["trial"] / 4} for m in profile["metrics"]})
    analysis = ResearchAnalysis.compute(rows, fingerprint="family", semantics=profile)
    family = analysis["families"]["efficiency"]
    assert family["dimensions"]["fraction"]["values"] == [10, 25, 100]
    assert all(p["support"] == 4 for p in family["points"])


def test_nonlinear_response_detected_without_scientific_replanning() -> None:
    rows = candidates(30)
    for c in rows:
        c["diagnostic_metrics"]["quality"] = {"mean": (c["parameters"]["width"] - 15) ** 2}
    result = ResearchAnalysis.compute(
        rows, fingerprint="u-shaped", semantics=semantics(), parameter_names=["width"]
    )
    response = next(r for r in result["relationships"] if r["x"] == "width" and r["y"] == "quality")
    assert response["shape"] == "nonlinear"
    assert response["effect"] > 0.5


def test_large_catalog_has_bounded_discovery_and_portable_payload() -> None:
    rng = np.random.default_rng(42)
    rows = candidates(500)
    for c in rows:
        c["parameters"] = {f"p{i}": float(rng.random()) for i in range(15)}
        c["diagnostic_metrics"] = {f"m{i}": {"mean": float(rng.random())} for i in range(300)}
        c["runs"] = [
            {"seed": seed, "metrics": {}, "final_objective": c["mean"]} for seed in range(15)
        ]
    start = time.monotonic()
    result = ResearchAnalysis.compute(
        rows,
        fingerprint="large",
        semantics=resolve_semantics({"discovery": {"resamples": 32, "max_pairs": 32}}),
        parameter_names=[f"p{i}" for i in range(15)],
    )
    assert len(result["metric_profiles"]) == 301
    assert len(result["relationships"]) <= 32
    assert result["methodology"]["screened_metric_count"] <= 64
    assert time.monotonic() - start < 30
    assert len(json.dumps(result)) < 12_000_000


def test_frozen_profile_is_used_on_reload_after_class_and_yaml_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from lambdaforge.work import ResultStore, WorkConfig, WorkRunner
    from tests.work_cases import SeedWork

    monkeypatch.setattr(SeedWork, "analysis_profile", {"metrics": {"score": {"label": "Original"}}})
    source = tmp_path / "work.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "profiled",
                "run": "tests.work_cases.SeedWork",
                "seeds": [1, 2],
                "objective": {"metric": "score", "mode": "max"},
            }
        )
    )
    result = WorkRunner().run(WorkConfig.from_yaml(source))
    assert result.status == "succeeded"
    frozen_path = result.execution_dir / "analysis-semantics.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["profiled"]["metrics"]["score"]["label"] == "Original"
    monkeypatch.setattr(SeedWork, "analysis_profile", {"metrics": {"score": {"label": "Changed"}}})
    source.write_text("not a valid Work YAML\n")
    store = ResultStore(tmp_path / ".lambdaforge" / "runs")
    analysis = store.analysis(result.execution_id, recompute=True)
    assert analysis["research"]["metric_catalog"]["metrics"]["score"]["label"] == "Original"
    assert analysis["source"]["analysis_semantics_identity"] == frozen["profiled"]["identity"]


def test_discovery_interleaves_parameters_with_metric_pairs() -> None:
    rows = candidates(8)
    for c in rows:
        c["parameters"] = {f"p{i}": c["trial"] + i for i in range(4)}
    result = ResearchAnalysis.compute(
        rows,
        fingerprint="balanced-budget",
        semantics=resolve_semantics({"discovery": {"max_pairs": 8, "resamples": 32}}),
        parameter_names=[f"p{i}" for i in range(4)],
    )
    assert {r["x"] for r in result["relationships"]} >= {f"p{i}" for i in range(4)}
    assert any(r["x"] not in rows[0]["parameters"] for r in result["relationships"])


def test_parameter_metric_name_collision_and_sparse_coverage() -> None:
    rows = candidates(12)
    for candidate in rows:
        candidate["parameters"]["quality"] = -candidate["trial"]
        if candidate["trial"] < 5:
            candidate["diagnostic_metrics"]["sparse"] = {"mean": candidate["trial"]}
    result = ResearchAnalysis.compute(
        rows, fingerprint="namespaces", semantics=semantics(), parameter_names=["quality"]
    )
    response = next(
        r
        for r in result["relationships"]
        if r["x"] == r["y"] == "quality" and r["x_kind"] == "parameter"
    )
    assert response["effect"] < 0
    assert result["metric_profiles"]["sparse"]["low_coverage"]
    assert result["metric_profiles"]["sparse"]["coverage"] == 4 / 12
    assert result["metric_profiles"]["quality"]["seed_coverage"] == 1
    assert len(result["inbox_findings"]) <= 8
    assert all("ranking_components" in f for f in result["findings"])


def test_research_html_browser_catalog_search_and_finding_drilldown(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    source = candidates()
    research = ResearchAnalysis.compute(
        source, fingerprint="ui", semantics=semantics(), parameter_names=["width", "category"]
    )
    analysis = {
        "source": {"status": "final"},
        "objective": {"metric": "quality", "mode": "max"},
        "search_space": {"width": {"range": [0, 23]}, "category": {"values": ["a", "b"]}},
        "candidates": source,
        "research": research,
    }
    path = write_html(analysis, tmp_path / "research.html")
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 1450, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.as_uri())
        assert page.locator("#study-research").is_visible()
        assert page.locator("#research-inbox .finding").count() > 0
        page.locator("#research-inbox button").first.click()
        assert page.locator("#research-detail").is_visible()
        page.click("#research-detail-close")
        page.keyboard.press("Control+k")
        page.fill("#research-global-query", "validation quality")
        assert page.locator("#research-search-results").inner_text().count("Quality") >= 1
        page.locator("#research-search-results button").first.click()
        assert page.locator("#study-custom").is_visible()
        page.get_by_role("button", name="Metrics & health", exact=True).click()
        page.fill("#research-metric-search", "fixed")
        assert "No matching" in page.locator("#research-metrics-body").inner_text()
        page.check("#research-show-all")
        assert "constant" in page.locator("#research-metrics-body").inner_text()
        page.screenshot(path=str(tmp_path / "research.png"), full_page=True)
        assert errors == []
        browser.close()


def test_browser_units_families_comparison_and_portable_views(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    source = candidates(12)
    declaration = semantics()
    declaration = resolve_semantics(
        {
            **{k: declaration[k] for k in ("metrics", "questions", "discovery")},
            "metrics": {**declaration["metrics"], "seconds": {"unit": "seconds"}},
            "families": {
                "efficiency": {
                    "label": "Quality by fraction",
                    "dimensions": {"fraction": {"kind": "ordered", "values": [10, 50]}},
                    "template": "quality_{fraction}",
                }
            },
        }
    )
    for c in source:
        c["diagnostic_metrics"].update(
            {
                "seconds": {"mean": c["trial"] * 60},
                "quality_10": {"mean": c["mean"]},
                "quality_50": {"mean": c["mean"] + 0.1},
            }
        )
    research = ResearchAnalysis.compute(
        source,
        fingerprint="ui-detailed",
        semantics=declaration,
        parameter_names=["width", "category"],
    )
    path = write_html(
        {
            "source": {"status": "final"},
            "objective": {"metric": "quality", "mode": "max"},
            "search_space": {"width": {"range": [0, 11]}, "category": {"values": ["a", "b"]}},
            "candidates": source,
            "research": research,
        },
        tmp_path / "detailed.html",
    )
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 1450, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.as_uri())
        page.screenshot(path=str(tmp_path / "inbox.png"), full_page=True)
        page.get_by_role("button", name="Metrics & health", exact=True).click()
        page.get_by_role("button", name="Quality by fraction", exact=True).click()
        family = page.eval_on_selector("#research-family-plot", "c => c.data[0]")
        assert family["x"] == ["10", "50"]
        assert family["y"][1] > family["y"][0]
        page.get_by_role("button", name="Explore", exact=True).click()
        page.evaluate(
            "window.lfResearchServices.customCharts.open({kind:'scatter',x:'param:width',y:'metric:quality',metrics:['metric:seconds']})"
        )
        page.wait_for_function(
            "document.querySelector('#study-custom-chart .js-plotly-plot').data?.length === 2"
        )
        assert (
            page.eval_on_selector("#study-custom-chart .js-plotly-plot", "c => c.layout.grid.rows")
            == 2
        )
        assert "independent" in page.locator("#study-chart-status").inner_text()
        page.get_by_text("Advanced visualization options", exact=True).click()
        page.check("#study-chart-normalize")
        page.wait_for_function(
            "Math.max(...document.querySelector('#study-custom-chart "
            ".js-plotly-plot').data[1].y)===1"
        )
        assert research["rows"][0]["values"]["seconds"] == 60
        page.evaluate(
            "window.lfResearchServices.customCharts.open({kind:'parallel',x:'param:width',y:'metric:quality'})"
        )
        page.wait_for_function(
            "document.querySelector('#study-custom-chart "
            ".js-plotly-plot').data[0].type==='parcoords'"
        )
        page.evaluate("""window.lfResearchServices.customCharts.importViews({views_version:1,views:[{
            name:'Portable',kind:'scatter',x:'param:width',y:'metric:quality',
            metrics:[],notes:'My note',palette:'Accessible'}]})""")
        assert page.locator(".saved-study-chart").count() == 1
        with pytest.raises(playwright.Error, match="Invalid saved chart"):
            page.evaluate(
                "window.lfResearchServices.customCharts.importViews({views_version:1,views:[{name:'Invalid',kind:'scatter',x:'param:width',y:'metric:unknown'}]})"
            )
        page.reload()
        assert page.locator(".saved-study-chart").inner_text() == "Portable"
        assert page.locator("#research-view-notes").input_value() == "My note"
        page.get_by_role("button", name="Trials", exact=True).click()
        page.locator(".trial-compare").nth(0).check()
        page.locator(".trial-compare").nth(1).check()
        assert "difference (right" in page.locator("#trial-comparison").inner_text().lower()
        assert "Quality" in page.locator("#trial-comparison").inner_text()
        assert errors == []
        browser.close()


def test_large_shared_picker_questions_hierarchy_and_smart_explore(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    playwright = pytest.importorskip("playwright.sync_api")
    source = candidates(12)
    for c in source:
        c["parameters"].update({f"p{i}": c["trial"] / 12 for i in range(13)})
        c["diagnostic_metrics"].update(
            {f"m{i}": {"mean": (c["trial"] + i) / 400} for i in range(300)}
        )
    declaration = resolve_semantics(
        {
            "defaults": [{"pattern": "m*", "metadata": {"category": "hardware/gpu"}}],
            "metrics": {
                "quality": {
                    "label": "Quality",
                    "aliases": ["predictive quality"],
                    "category": "validation/global",
                    "unit": "ratio",
                    "visibility": "primary",
                },
                "duplicate": {"category": "validation/surface/quality", "unit": "ratio"},
                "small": {"category": "validation/surface/calibration", "unit": "ratio"},
                "missing": {"category": "validation/global"},
            },
            "questions": [
                {
                    "id": "available",
                    "label": "Agreement question",
                    "priority": 10,
                    "kind": "relationship",
                    "x": "quality",
                    "y": "duplicate",
                },
                {
                    "id": "missing",
                    "label": "Unobserved question",
                    "optional": True,
                    "kind": "relationship",
                    "x": "quality",
                    "y": "missing",
                },
            ],
            "discovery": {"max_metrics": 16, "max_pairs": 8, "resamples": 32},
        }
    )
    research = ResearchAnalysis.compute(
        source,
        fingerprint="shared-controls",
        semantics=declaration,
        parameter_names=list(source[0]["parameters"]),
    )
    path = write_html(
        {
            "source": {"status": "final"},
            "objective": {"metric": "quality", "mode": "max"},
            "search_space": {
                "width": {"range": [0, 11]},
                "category": {"values": ["a", "b"]},
                **{f"p{i}": {"range": [0, 1]} for i in range(13)},
            },
            "candidates": source,
            "research": research,
        },
        tmp_path / "shared-controls.html",
    )
    markup = path.read_text(encoding="utf-8")
    assert markup.count('"metric_catalog"') == 1
    assert '<option value="m299"' not in markup
    with playwright.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).exists():
            pytest.skip("Optional Chromium is unavailable.")
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 1450, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.as_uri())
        assert page.locator("select option").count() < 120
        assert page.locator("#research-questions .configured-question").count() == 2
        assert page.locator("#research-questions h3").first.inner_text() == "Agreement question"
        page.locator("#research-questions button").first.click()
        assert "Support and status" in page.locator("#research-detail-body").inner_text()
        page.click("#research-detail-close")
        page.screenshot(path=str(tmp_path / "shared-summary.png"), full_page=True)

        page.get_by_role("button", name="Metrics & health", exact=True).click()
        page.locator("#research-category-tree > summary").click()
        page.get_by_role("button", name="validation", exact=True).click()
        categories = page.locator("#research-metrics-body tr td:nth-child(2)").all_text_contents()
        assert len(categories) >= 3
        assert all(value.startswith("validation/") for value in categories)
        assert any("surface/calibration" in value for value in categories)

        page.get_by_role("button", name="Explore", exact=True).click()
        page.click("#research-primary-metric")
        page.fill("#research-global-query", "predictive quality")
        page.locator('#research-search-results button[data-choice="quality"]').locator(
            ".."
        ).get_by_title("Toggle favourite").click()
        assert page.get_by_title("Toggle favourite").inner_text() == "★"
        page.locator("#research-global-query").focus()
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Enter")
        assert "quality" in page.evaluate("window.lfResearchServices.getPrefs().researchFavorites")
        assert "quality" in page.evaluate("window.lfResearchServices.getPrefs().researchRecent")
        page.get_by_role("button", name="By · add parameter…", exact=True).click()
        page.fill("#research-global-query", "width")
        page.locator('#research-search-results button[data-choice="param:width"]').click()
        page.click("#research-explore")
        page.wait_for_function("document.querySelector('#study-chart-kind').value==='heatmap'")
        page.get_by_role("button", name="By · add parameter…", exact=True).click()
        page.fill("#research-global-query", "p12")
        page.locator('#research-search-results button[data-choice="param:p12"]').click()
        page.click("#research-explore")
        page.wait_for_function(
            "document.querySelector('#study-custom-chart "
            ".js-plotly-plot').data[0].type==='parcoords'"
        )
        dimensions = page.eval_on_selector(
            "#study-custom-chart .js-plotly-plot", "c => c.data[0].dimensions.map(d=>d.label)"
        )
        assert all(name in dimensions for name in ("width", "category", "p12"))
        page.screenshot(path=str(tmp_path / "shared-explore.png"), full_page=True)

        page.get_by_role("button", name="Parameters", exact=True).click()
        page.click("#parameter-metric-picker")
        page.fill("#research-global-query", "quality")
        page.locator('#research-search-results button[data-choice="quality"]').click()
        page.get_by_role("button", name="Add comparison metric…", exact=True).click()
        page.fill("#research-global-query", "small")
        page.locator('#research-search-results button[data-choice="small"]').click()
        assert "small" in page.locator("#parameter-metric-chips").inner_text()
        page.locator("#parameter-metric-chips button").click()
        assert page.locator("#parameter-metric-chips button").count() == 0
        page.get_by_role("button", name="Interactions", exact=True).click()
        page.locator("#study-interactions").get_by_role(
            "button", name="Analyze", exact=False
        ).click()
        page.fill("#research-global-query", "quality")
        page.locator('#research-search-results button[data-choice="quality"]').click()
        page.wait_for_function(
            "document.querySelector('#research-observed-interactions "
            ".js-plotly-plot').data.length>0"
        )
        page.get_by_role("button", name="Evidence", exact=True).click()
        assert page.locator("#study-findings .finding").count() == 0
        assert "Surrogate diagnostics" in page.locator("#study-findings").inner_text()
        page.get_by_text("Complete reproducible analysis JSON", exact=True).click()
        page.wait_for_function("document.querySelector('#study-raw-analysis').dataset.loaded==='true'")
        snapshot = json.loads(page.locator("#study-raw-analysis").inner_text())
        assert snapshot["research"]["metric_catalog"] == research["metric_catalog"]
        assert snapshot["candidates"] == source
        assert errors == []
        browser.close()
