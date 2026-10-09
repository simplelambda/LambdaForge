"""Lazy ordinary-Work evidence explorer shared by native and imported results."""

# Human-facing diagnostic copy stays as complete strings, as in Workspace.py.
# ruff: noqa: E501

from __future__ import annotations

import time
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, RichLog, Select, Static, TabbedContent, TabPane

from lambdaforge.tui.viewmodels import structured_text
from lambdaforge.tui.widgets.MetricDashboard import MetricDashboard


class WorkEvidence(Vertical):
    """Never combine Attempts; fetch only the currently visible evidence panel."""

    DEFAULT_CSS = """
    WorkEvidence { height: 1fr; min-height: 18; }
    WorkEvidence > TabbedContent { height: 1fr; }
    #evidence-output-table { height: 1fr; min-height: 6; }
    #evidence-attempt { width: 1fr; min-width: 24; }
    """

    def __init__(self, work: Mapping[str, Any], services: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.work, self.services = dict(work), services
        self._overview: dict[str, Any] | None = None
        self._detail: dict[str, Any] = {}
        self._loading = False
        self._selection: tuple[str, int] | None = None
        self._page = 0
        self._updated: float | None = None
        self._preview_text: str | None = None
        self._active_output_name: str | None = None
        self._product_loading = False
        self._dependency_loading = False
        self._dependencies: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        with Horizontal(classes="workspace-actions"):
            yield Select([], prompt="Select Run / Attempt", id="evidence-attempt")
            yield Button("Open HTML report", id="evidence-report", variant="primary")
            yield Button("Choose reusable input…", id="evidence-reference")
            yield Button("Materialize…", id="evidence-materialize")
        yield Static(
            "Loading persisted evidence…",
            id="evidence-status",
            classes="freshness-line",
            markup=False,
        )
        with TabbedContent(id="evidence-tabs"):
            with TabPane("Overview", id="evidence-overview"):
                yield VerticalScroll(Static(id="evidence-summary", markup=False))
            with TabPane("Metrics", id="evidence-metrics"):
                with Horizontal(classes="workspace-actions"):
                    yield Button("Previous curves", id="evidence-previous")
                    yield Button("Next curves", id="evidence-next")
                    yield Static(id="evidence-metric-page")
                yield MetricDashboard(id="evidence-curves")
                yield Static(id="evidence-metric-summary", markup=False)
            with TabPane("Outputs", id="evidence-outputs"):
                yield DataTable(id="evidence-output-table", cursor_type="row", zebra_stripes=True)
                yield VerticalScroll(Static(id="evidence-output-detail", markup=False))
            with TabPane("Visualizations", id="evidence-visualizations"):
                yield VerticalScroll(Static(id="evidence-html", markup=False))
            with TabPane("Products", id="evidence-products"):
                yield DataTable(id="evidence-product-table", cursor_type="row", zebra_stripes=True)
                yield Static(id="evidence-product-detail", markup=False)
            with TabPane("Logs", id="evidence-logs"):
                yield RichLog(id="evidence-log", wrap=True, auto_scroll=False)
            with TabPane("Resources", id="evidence-resources"):
                yield VerticalScroll(Static(id="evidence-resource", markup=False))
            with TabPane("Provenance", id="evidence-provenance"):
                yield DataTable(id="evidence-input-table", cursor_type="row", zebra_stripes=True)
                yield Static(
                    "Enter opens an exact historical result or independent product when available in this catalog.",
                    id="evidence-input-status",
                    markup=False,
                )
                yield VerticalScroll(Static(id="evidence-provenance-content", markup=False))

    def on_mount(self) -> None:
        self.query_one("#evidence-output-table", DataTable).add_columns(
            "Name", "Kind / role", "Bytes", "Availability", "Location"
        )
        self.query_one("#evidence-product-table", DataTable).add_columns(
            "Product", "Kind", "Contract", "Publication", "Exact content"
        )
        self.query_one("#evidence-input-table", DataTable).add_columns(
            "Input", "Kind", "Exact source", "Content / evidence identity"
        )
        self.refresh_evidence()
        self.call_after_refresh(self.refresh_evidence)
        self.set_interval(3, self._poll)

    def _poll(self) -> None:
        if (
            self.screen is not self.app.screen
            or not self.is_on_screen
            or not self.region.width
            or not self.region.height
        ):
            return
        self.refresh_evidence()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.tabbed_content.id == "evidence-tabs":
            event.stop()
            self.refresh_evidence()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "evidence-attempt" or event.value is Select.BLANK:
            return
        event.stop()
        run, attempt = str(event.value).rsplit("/", 1)
        selection = (run, int(attempt))
        if selection != self._selection:
            self._selection = selection
            self._preview_text = None
            self._page = 0
            self.refresh_evidence()

    def refresh_evidence(self) -> None:
        if self._loading or not self.is_mounted or not self.region.height:
            return
        tab = self.query_one("#evidence-tabs", TabbedContent).active
        view = {
            "evidence-metrics": "metrics",
            "evidence-outputs": "outputs",
            "evidence-visualizations": "outputs",
            "evidence-logs": "logs",
            "evidence-resources": "provenance",
            "evidence-provenance": "provenance",
        }.get(tab, "overview")
        if self._overview is None:
            view = "overview"
        options: dict[str, Any] = {"view": view}
        if self._selection and view != "overview":
            options.update(run_id=self._selection[0], attempt=self._selection[1])
        self._loading = True
        captured = self._selection

        def load() -> None:
            try:
                value = self.services.work_result_view(self.work, **options)
                self.app.call_from_thread(self._loaded, view, captured, value, None)
            except Exception as error:
                self.app.call_from_thread(self._loaded, view, captured, None, error)

        Thread(target=load, daemon=True, name="lambdaforge-work-evidence").start()

    def _loaded(self, view: str, selection: Any, value: Any, error: Exception | None) -> None:
        self._loading = False
        if not self.is_mounted:
            return
        status = self.query_one("#evidence-status", Static)
        if error:
            status.update(
                f"Evidence unavailable · {type(error).__name__}: {error} · Job logs and controls remain available."
            )
            return
        self._updated = time.monotonic()
        status.update("Persisted evidence · updated now · artifact bytes are not downloaded")
        if view == "overview":
            previous = self._overview
            self._overview = value
            rows = value.get("attempts", ())
            if previous is None or previous.get("attempts") != rows:
                options = [
                    (
                        f"{row['name']} · {row['run_id']} · {row['attempt_id']} · {row['status']}",
                        f"{row['run_id']}/{row['attempt_number']}",
                    )
                    for row in rows
                ]
                select = self.query_one("#evidence-attempt", Select)
                select.set_options(options)
                if self._selection and any(
                    option[1] == f"{self._selection[0]}/{self._selection[1]}" for option in options
                ):
                    select.value = f"{self._selection[0]}/{self._selection[1]}"
                elif options:
                    select.value = options[-1][1]
                    run, attempt = str(select.value).rsplit("/", 1)
                    self._selection = run, int(attempt)
            summary = structured_text(value, heading="WORK RESULTS", limit=120)
            self.query_one("#evidence-summary", Static).update(summary)
            products = (value.get("products") or {}).get("items", ())
            if getattr(self, "_products", None) != products:
                self._products = products
                table = self.query_one("#evidence-product-table", DataTable)
                table.clear()
                for index, row in enumerate(products):
                    table.add_row(
                        str(row["name"]),
                        str(row.get("kind", "—")),
                        str(row.get("contract", "—")),
                        str(row.get("status", "unknown")),
                        str(row.get("content_id", "not published")),
                        key=str(index),
                    )
                self.query_one("#evidence-product-detail", Static).update(
                    "Enter opens the exact independent product when it is locally materialized. "
                    "Publication failures can be retried with lf products finalize EXECUTION --apply."
                    if products
                    else "No independent product promotion declared. Registered outputs remain Execution-owned evidence."
                )
            if self.query_one("#evidence-tabs", TabbedContent).active not in {
                "evidence-overview",
                "evidence-products",
            }:
                self.refresh_evidence()
            return
        if selection != self._selection:
            self.refresh_evidence()
            return
        self._detail = value
        if view == "metrics":
            self._show_curves()
        elif view == "outputs":
            table = self.query_one("#evidence-output-table", DataTable)
            cursor, scroll = table.cursor_row, table.scroll_offset
            artifacts = list(value.get("artifacts", ())) + list(value.get("checkpoints", ()))
            if getattr(self, "_artifacts", None) != artifacts:
                self._preview_text = None
                table.clear()
                self._artifacts = artifacts
                for index, row in enumerate(artifacts):
                    table.add_row(
                        str(row["name"]),
                        str(row.get("role", row.get("kind", "artifact"))),
                        str(row.get("size_bytes", "—")),
                        str(row["availability"]),
                        str(row["location"]),
                        key=str(index),
                    )
                if artifacts:
                    table.move_cursor(row=min(cursor, len(artifacts) - 1))
                table.scroll_to(x=scroll.x, y=scroll.y, animate=False)
            self.query_one("#evidence-output-detail", Static).update(
                self._preview_text
                or structured_text(
                    {
                        "primary_result": value.get("result"),
                        "values": value.get("outputs"),
                        "datasets": value.get("datasets"),
                        "checkpoints": value.get("checkpoints"),
                    },
                    heading="REGISTERED STRUCTURED OUTPUTS",
                )
            )
            sections = [row for row in artifacts if row.get("role") == "html-section"]
            self.query_one("#evidence-html", Static).update(
                structured_text(sections, heading="PROJECT VISUALIZATIONS")
                if sections
                else "No HTML sections registered by this Attempt. A generic Work report is still available."
            )
        elif view == "logs":
            content = value.get("log", "")
            failure = value.get("selected", {}).get("failure")
            if failure:
                content += "\n\nSCIENTIFIC FAILURE\n" + structured_text(failure)
            log = self.query_one("#evidence-log", RichLog)
            if getattr(self, "_log", None) != content:
                scroll = log.scroll_offset
                log.clear()
                log.write(content or "No persisted output yet.")
                log.scroll_to(x=scroll.x, y=scroll.y, animate=False)
                self._log = content
        elif view == "provenance":
            provenance = value.get("provenance", {})
            dependencies = [
                dict(row)
                for row in provenance.get("inputs", ())
                if row.get("kind") in {"result", "product"}
            ]
            if self._dependencies != dependencies:
                self._dependencies = dependencies
                table = self.query_one("#evidence-input-table", DataTable)
                table.clear()
                for index, row in enumerate(dependencies):
                    table.add_row(
                        str(row["name"]),
                        str(row["kind"]),
                        str(row["logical_source"]),
                        str(row.get("content_id", "—")),
                        key=str(index),
                    )
                self.query_one("#evidence-input-status", Static).update(
                    "Enter opens the exact recorded dependency; physical availability is checked on demand."
                    if dependencies
                    else "No historical result or independent product input recorded."
                )
            self.query_one("#evidence-provenance-content", Static).update(
                structured_text(provenance, heading="EXACT INPUTS & PROVENANCE")
            )
            self.query_one("#evidence-resource", Static).update(
                structured_text(
                    {
                        "requested": provenance.get("resources", {}),
                        "observed": value.get("resource_observation"),
                    },
                    heading="RECORDED RESOURCES",
                )
            )

    def _show_curves(self) -> None:
        curves = self._detail.get("curves", {})
        names = list(curves)
        self._page = min(self._page, max(0, (len(names) - 1) // 4))
        start = self._page * 4
        self.query_one("#evidence-curves", MetricDashboard).show_curves(
            curves,
            names[start : start + 4],
            selected_step=None,
            best_step=None,
            axis_label="step / observation",
            display_names=self._detail.get("chart_filter", {}).get("display_names"),
        )

        self.query_one("#evidence-metric-page", Static).update(
            f"{start + 1 if names else 0}–{min(start + 4, len(names))} / {len(names)} metrics"
        )
        self.query_one("#evidence-metric-summary", Static).update(
            structured_text(self._detail.get("latest", {}), heading="LATEST OBSERVATIONS")
            if names
            else "No scalar metrics registered by this Attempt."
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "evidence-product-table":
            event.stop()
            rows = getattr(self, "_products", ())
            if 0 <= event.cursor_row < len(rows):
                self.query_one("#evidence-product-detail", Static).update(
                    structured_text(rows[event.cursor_row], heading="EXACT PRODUCT PUBLICATION")
                )
            return
        if event.data_table.id != "evidence-output-table":
            return
        event.stop()
        rows = getattr(self, "_artifacts", ())
        if 0 <= event.cursor_row < len(rows):
            self._preview_text = None
            self._active_output_name = str(rows[event.cursor_row]["name"])
            self.query_one("#evidence-output-detail", Static).update(
                structured_text(rows[event.cursor_row], heading="SELECTED OUTPUT · EXACT METADATA")
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "evidence-input-table":
            event.stop()
            if self._dependency_loading or not 0 <= event.cursor_row < len(self._dependencies):
                return
            dependency = self._dependencies[event.cursor_row]
            self._dependency_loading = True
            self.query_one("#evidence-input-status", Static).update(
                "Opening exact recorded dependency…"
            )

            def resolve_dependency() -> None:
                try:
                    kind = dependency["kind"]
                    selector = str(dependency["logical_source"])
                    value = (
                        self.services.historical_result(selector)
                        if kind == "result"
                        else self.services.product_detail(selector)["product"]
                    )
                    self.app.call_from_thread(self._dependency_ready, kind, value, None)
                except Exception as error:
                    self.app.call_from_thread(self._dependency_ready, "", None, error)

            Thread(
                target=resolve_dependency, daemon=True, name="lambdaforge-evidence-relation"
            ).start()
            return
        if event.data_table.id == "evidence-output-table":
            event.stop()
            rows = getattr(self, "_artifacts", ())
            if not self._selection or not 0 <= event.cursor_row < len(rows):
                return
            artifact = rows[event.cursor_row]
            if artifact.get("kind") == "checkpoint":
                self.query_one("#evidence-output-detail", Static).update(
                    structured_text(artifact, heading="SHARED RUN CHECKPOINT · METADATA ONLY")
                )
                return
            selection = self._selection
            self.query_one("#evidence-output-detail", Static).update(
                "Verifying selected small output…"
            )

            def preview() -> None:
                try:
                    value = self.services.work_output_preview(
                        self.work, artifact["name"], run_id=selection[0], attempt=selection[1]
                    )
                    self.app.call_from_thread(
                        self._preview_ready, selection, artifact["name"], value, None
                    )
                except Exception as error:
                    self.app.call_from_thread(
                        self._preview_ready, selection, artifact["name"], None, error
                    )

            Thread(target=preview, daemon=True, name="lambdaforge-output-preview").start()
            return
        if event.data_table.id != "evidence-product-table":
            return
        event.stop()
        rows = getattr(self, "_products", ())
        if (
            self._product_loading
            or not 0 <= event.cursor_row < len(rows)
            or not rows[event.cursor_row].get("content_id")
        ):
            return
        self._product_loading = True
        selector = str(rows[event.cursor_row]["content_id"])
        self.query_one("#evidence-product-detail", Static).update("Opening exact product metadata…")

        def load() -> None:
            try:
                value = self.services.product_detail(selector)
                self.app.call_from_thread(self._product_ready, value, None)
            except Exception as error:
                self.app.call_from_thread(self._product_ready, None, error)

        Thread(target=load, daemon=True, name="lambdaforge-work-product").start()

    def _product_ready(self, value: Any, error: Exception | None) -> None:
        self._product_loading = False
        if not self.is_mounted:
            return
        if error:
            self.query_one("#evidence-product-detail", Static).update(
                f"Product not locally available: {error}. Export/import or materialize its exact content first."
            )
            return
        from lambdaforge.tui.screens.ProductScreen import ProductWorkspace

        self.app.push_screen(ProductWorkspace(value["product"], self.services))

    def _dependency_ready(self, kind: str, value: Any, error: Exception | None) -> None:
        self._dependency_loading = False
        if not self.is_mounted:
            return
        if error:
            self.query_one("#evidence-input-status", Static).update(
                f"Dependency unavailable here: {error}. No producer was launched and no identity substituted."
            )
            return
        from lambdaforge.tui.screens.Base import EntitySelected

        self.post_message(EntitySelected(kind, value))

    def _preview_ready(
        self, selection: Any, name: str, value: Any, error: Exception | None
    ) -> None:
        if not self.is_mounted or selection != self._selection or name != self._active_output_name:
            return
        text = (
            f"Output preview failed: {error}"
            if error
            else (str(value.get("notice", "")) + "\n\n" + str(value.get("content") or ""))
        )
        self._preview_text = text
        self.query_one("#evidence-output-detail", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "evidence-reference":
            from lambdaforge.tui.widgets.DependencyPicker import DependencyPicker

            event.stop()
            self.app.push_screen(DependencyPicker(self.services))
            return
        if event.button.id == "evidence-materialize":
            from lambdaforge.tui.widgets.DependencyPicker import MaterializationDialog

            event.stop()
            selector = (self._overview or {}).get("execution_id")
            if not selector:
                self.query_one("#evidence-status", Static).update(
                    "Wait for owned Execution metadata."
                )
                return
            self.app.push_screen(MaterializationDialog(self.services, str(selector), kind="result"))
            return
        if event.button.id not in {"evidence-previous", "evidence-next", "evidence-report"}:
            return
        event.stop()
        if event.button.id == "evidence-previous":
            self._page = max(0, self._page - 1)
            self._show_curves()
        elif event.button.id == "evidence-next":
            self._page += 1
            self._show_curves()
        else:
            event.button.disabled = True
            self.query_one("#evidence-status", Static).update("Generating offline Work report…")

            def report() -> None:
                try:
                    path = self.services.work_report(
                        self.work,
                        Path.cwd()
                        / f"{self.work.get('execution_id', self.work.get('primary_job_id', 'work'))}-results.html",
                    )
                    self.app.call_from_thread(self._report_ready, path, None)
                except Exception as error:
                    self.app.call_from_thread(self._report_ready, None, error)

            Thread(target=report, daemon=True, name="lambdaforge-work-report").start()

    def _report_ready(self, path: Path | None, error: Exception | None) -> None:
        if not self.is_mounted:
            return
        self.query_one("#evidence-report", Button).disabled = False
        self.query_one("#evidence-status", Static).update(
            f"Report failed: {error}" if error else f"Offline report: {path}"
        )
        if path is not None:
            webbrowser.open(path.resolve().as_uri())
