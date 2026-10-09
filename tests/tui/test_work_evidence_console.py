"""First-class Work views and exact historical input selection are lazy and service-backed."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("textual")
from textual.widgets import DataTable, Input, TabbedContent, TextArea  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp, WorkLaunchDialog  # noqa: E402
from lambdaforge.tui.screens.Workspace import ResultWorkspace  # noqa: E402
from lambdaforge.tui.widgets.DependencyPicker import DependencyPicker  # noqa: E402
from lambdaforge.work import ResultStore  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402
from tests.work.test_work_results import execute  # noqa: E402


class EvidenceServices(FakeServices):
    def __init__(self, store: ResultStore) -> None:
        super().__init__()
        self.store = store

    def work_result_view(self, work: dict[str, Any], **options: Any) -> dict[str, Any]:
        self.calls.append(("work_view", options))
        return self.store.view(work["execution_id"], **options)

    def dependency_choices(self, *, offset: int = 0) -> list[dict[str, Any]]:
        return [
            {
                "kind": "result",
                "label": row["display_name"],
                "state": row["status"],
                "identity": row["execution_id"],
            }
            for row in self.store.catalog()
        ][offset : offset + 100]

    def dependency_details(self, choice: dict[str, Any]) -> dict[str, Any]:
        return self.store.view(choice["identity"], view="overview")

    def historical_result(self, execution_id: str) -> dict[str, Any]:
        self.calls.append(("historical_result", execution_id))
        return next(row for row in self.store.catalog() if row["execution_id"] == execution_id)

    def dependency_reference(self, choice: dict[str, Any], **options: Any) -> dict[str, Any]:
        self.calls.append(("dependency_reference", choice["identity"]))
        return self.store.reference(choice["identity"], **options)

    def work_output_preview(
        self, work: dict[str, Any], name: str, **options: Any
    ) -> dict[str, Any]:
        self.calls.append(("output_preview", name))
        return self.store.output_preview(work["execution_id"], name, **options)

    def validate_work(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("validate_bound", kwargs))
        return {"valid": True}

    def explain_work(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("explain_bound", kwargs))
        return {"name": "consumer"}

    def submit_work(self, path: Path, cluster: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("submit_bound", kwargs))
        return {"job_id": "job-fixture", "state": "preparing"}


def test_work_view_defers_curves_and_outputs_until_visible(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="console-fixture"\nversion="1"\n')
    execution = execute(tmp_path)
    services = EvidenceServices(ResultStore())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(150, 55)) as pilot:
            screen = ResultWorkspace(services.store.catalog()[0], services)
            app.push_screen(screen)
            for _ in range(8):
                await pilot.pause(0.1)
                if any(key == "work_view" for key, _ in services.calls):
                    break
            views = [value for key, value in services.calls if key == "work_view"]
            assert views and all(value["view"] == "overview" for value in views)
            tabs = screen.query_one("#evidence-tabs", TabbedContent)
            tabs.active = "evidence-metrics"
            await pilot.pause(0.4)
            assert any(
                key == "work_view" and value["view"] == "metrics" for key, value in services.calls
            )
            tabs.active = "evidence-outputs"
            await pilot.pause(0.4)
            table = screen.query_one("#evidence-output-table", DataTable)
            assert table.row_count == 4  # Three artifacts plus the registered checkpoint.
            table.focus()
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert any(key == "output_preview" for key, _ in services.calls)
            assert not any(key == "result_analysis" for key, _ in services.calls)
            assert execution.execution_id == screen.result["execution_id"]

    asyncio.run(exercise())


def test_historical_picker_returns_exact_pin_and_launch_passes_it(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="console-fixture"\nversion="1"\n')
    first = execute(tmp_path)
    execute(tmp_path, multiplier=2)
    services = EvidenceServices(ResultStore())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        selected: list[Any] = []
        async with app.run_test(size=(150, 60)) as pilot:
            picker = DependencyPicker(services)
            app.push_screen(picker, selected.append)
            await pilot.pause(0.4)
            table = picker.query_one("#dependency-table", DataTable)
            table.focus()
            assert table.row_count == 2
            row = next(
                index
                for index, value in enumerate(picker._rows)
                if value["identity"] == first.execution_id
            )
            table.move_cursor(row=row)
            await pilot.press("enter")
            await pilot.pause(0.4)
            assert first.execution_id in picker.query_one("#dependency-declaration", TextArea).text
            picker.query_one("#dependency-attempt").value = f"{first.runs[0].run_id}/1"
            await pilot.pause(0.3)
            assert picker._reference["result"]["attempt"] == 1
            await pilot.click("#dependency-use")
            await pilot.pause()
            assert selected[0]["result"]["execution"] == first.execution_id
            launch = WorkLaunchDialog(services)
            app.push_screen(launch)
            await pilot.pause()
            launch.query_one("#launch-config", Input).value = str(tmp_path / "consumer.yaml")
            launch._dependency_selected("evidence", selected[0])
            await pilot.click("#launch-submit")
            await pilot.pause(0.5)
            submissions = [value for key, value in services.calls if key == "submit_bound"]
            assert submissions == [{"input_bindings": {"evidence": selected[0]}}]

    asyncio.run(exercise())


def test_provenance_opens_exact_parent_without_reexecuting_or_using_newer_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from lambdaforge.work import WorkConfig, WorkRunner

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="console-fixture"\nversion="1"\n')
    parent = execute(tmp_path)
    marker = ResultStore().reference(parent.execution_id)
    consumer = WorkRunner().run(
        WorkConfig.from_mapping(
            {
                "name": "consumer",
                "run": "tests.work.test_work_results.HistoricalConsumer",
                "with": {"evidence": marker},
                "resources": {"cpu": 1},
            },
            source=tmp_path / "consumer.yaml",
        )
    )
    execute(tmp_path, multiplier=2)
    services = EvidenceServices(ResultStore())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(150, 60)) as pilot:
            screen = ResultWorkspace(services.historical_result(consumer.execution_id), services)
            app.push_screen(screen)
            await pilot.pause(0.4)
            screen.query_one("#evidence-tabs", TabbedContent).active = "evidence-provenance"
            await pilot.pause(0.4)
            table = screen.query_one("#evidence-input-table", DataTable)
            assert table.row_count == 1
            table.focus()
            await pilot.press("enter")
            await pilot.pause(0.5)
            assert app.screen is not screen
            assert app.screen.result["execution_id"] == parent.execution_id
            assert ("historical_result", parent.execution_id) in services.calls
            assert not any(key == "submit_bound" for key, _ in services.calls)

    asyncio.run(exercise())


def test_picker_pins_a_registered_product_artifact(tmp_path: Path, monkeypatch: Any) -> None:
    from lambdaforge.products import ProductRegistry
    from lambdaforge.tui.services import ConsoleServices
    from lambdaforge.work import WorkConfig, WorkRunner

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="console-product"\nversion="1"\n')
    WorkRunner().run(
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
                        "outputs": ["report", "viewer"],
                    }
                },
            },
            source=tmp_path / "publish.yaml",
        )
    )
    product = ProductRegistry().show("report")

    class ProductEvidenceServices(EvidenceServices):
        def dependency_choices(self, **kwargs: Any) -> Any:
            return [
                {
                    "kind": "product",
                    "label": "report",
                    "state": "published",
                    "identity": product.content_id,
                    "contract": product.contract.identifier,
                }
            ]

        def dependency_details(self, choice: dict[str, Any]) -> Any:
            return {"product": product.to_dict()}

        def dependency_reference(self, choice: Any, **options: Any) -> Any:
            return ConsoleServices.dependency_reference(self, choice, **options)

    services = ProductEvidenceServices(ResultStore())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        selected: list[Any] = []
        async with app.run_test(size=(150, 60)) as pilot:
            picker = DependencyPicker(services)
            app.push_screen(picker, selected.append)
            await pilot.pause(0.4)
            picker.query_one("#dependency-table", DataTable).focus()
            await pilot.press("enter")
            await pilot.pause(0.4)
            picker.query_one("#dependency-artifact").value = "0"
            await pilot.pause(0.4)
            assert picker._reference["product"]["artifact"] == product.artifacts[0].name
            await pilot.click("#dependency-use")
            await pilot.pause()
            assert selected[0]["product"]["name"] == product.content_id
            assert selected[0]["product"]["artifact"] == product.artifacts[0].name

    asyncio.run(exercise())
