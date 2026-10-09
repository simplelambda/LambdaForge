"""Read-only native product browser and explicit portable-evidence import controls."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.screens.Workspace import (
    ExactConfirmation,
    ExportDirectoryPicker,
    ResearchWorkspace,
)
from lambdaforge.tui.viewmodels import structured_text


class ProductScreen(DataScreen):
    """Catalog metadata only; importing or inspecting does not submit a Work."""

    TITLE = "Products · durable contracts, independent artifacts and provenance"

    def __init__(self, loader: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.page_offset = 0
        super().__init__(lambda: loader(offset=self.page_offset), *args, **kwargs)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="workspace-actions"):
            yield Button("Import Study…", id="products-import", flat=True, variant="primary")
            yield Button("Previous 100", id="products-page-previous", flat=True)
            yield Button("Next 100", id="products-page-next", flat=True)
        yield from super().compose()

    def populate(self, table: DataTable[Any], value: Any) -> None:
        self.query_one("#screen-status", Static).update(
            f"Catalog items {self.page_offset + 1}–{self.page_offset + len(value)}"
            if value
            else "No products on this page."
        )
        table.clear(columns=True)
        table.add_columns("Product", "Kind", "Contract", "Artifacts", "Bytes")
        for item in value:
            table.add_row(
                str(item["name"]),
                str(item["kind"]),
                str(item["contract"]["identifier"]),
                str(item["artifact_count"]),
                str(item["size_bytes"]),
                key=str(item["name"]),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in {"products-page-previous", "products-page-next"}:
            event.stop()
            if not self._loading:
                self.page_offset = max(
                    0, self.page_offset + (100 if event.button.id == "products-page-next" else -100)
                )
                self.reload()

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return (
                "No products registered. Publish a native Study product "
                "or import verified evidence."
            )
        item = self.last_success[row]
        return (
            f"{item['name']} · {item['kind']}\nContract: {item['contract']['identifier']}\n"
            f"Content: {item['content_id']}\nScientific meaning: {item['scientific_id']}\n"
            "Enter opens metadata and audit history; artifact verification is an explicit action."
        )

    def open_row(self, row: int) -> None:
        if isinstance(self.last_success, list) and 0 <= row < len(self.last_success):
            self.post_message(EntitySelected("product", self.last_success[row]))


class ProductWorkspace(ResearchWorkspace):
    """Bounded metadata/audit views and service-backed byte operations."""

    def __init__(self, product: Mapping[str, Any], services: Any) -> None:
        self.product = dict(product)
        self.services = services
        self.busy = False
        self.audit_offset = 0
        super().__init__(f"Products / {self.product['name']}")

    def compose_workspace(self) -> ComposeResult:
        yield Static(
            f"{self.product['name']} · {self.product['kind']}\n"
            f"Contract: {self.product['contract']['identifier']}",
            classes="workspace-header",
            markup=False,
        )
        with Horizontal(classes="workspace-actions"):
            yield Button("Verify bytes", id="product-verify", flat=True)
            yield Button("Export product…", id="product-export", flat=True)
            yield Button("Materialize…", id="product-materialize", flat=True)
            yield Button("Choose reusable input…", id="product-reference", flat=True)
            yield Button("Previous audit page", id="product-previous", flat=True)
            yield Button("Next audit page", id="product-next", flat=True)
        yield Static(
            "Loading product metadata…", id="product-status", classes="status-line", markup=False
        )
        with TabbedContent():
            for label, key in (
                ("Meaning", "meaning"),
                ("Artifacts", "artifacts"),
                ("Producer history", "provenance"),
                ("Consumers", "consumers"),
            ):
                with TabPane(label):
                    yield VerticalScroll(Static("Loading…", id=f"product-{key}", markup=False))

    def on_mount(self) -> None:
        self._operation(
            "Loading metadata",
            lambda: self.services.product_detail(self.product["name"], offset=self.audit_offset),
            self._loaded,
        )

    def _loaded(self, value: dict[str, Any]) -> None:
        product = value["product"]
        self.query_one("#product-meaning", Static).update(
            structured_text(
                {
                    "scientific_meaning": product["scientific_meaning"],
                    "payload": product["payload"],
                    "content_id": product["content_id"],
                    "scientific_id": product["scientific_id"],
                },
                heading="SEALED MEANING · publisher declaration, not independent verification",
            )
        )
        for key in ("artifacts", "provenance", "consumers"):
            entries = product[key] if key == "artifacts" else value[key]
            self.query_one(f"#product-{key}", Static).update(
                structured_text(
                    entries, heading=f"{key.upper()} · page starts at {value['offset']}"
                )
                if entries
                else "No records on this page."
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        selected = event.button.id
        if self.busy:
            return
        if selected == "product-verify":
            self._operation(
                "Verifying artifact checksums",
                lambda: self.services.verify_product(self.product["name"]),
                lambda value: None,
            )
        elif selected == "product-export":
            self.app.push_screen(ExportDirectoryPicker(), self._export)
        elif selected == "product-materialize":
            from lambdaforge.tui.widgets.DependencyPicker import MaterializationDialog

            self.app.push_screen(
                MaterializationDialog(
                    self.services, str(self.product["content_id"]), kind="product"
                )
            )
        elif selected == "product-reference":
            from lambdaforge.tui.widgets.DependencyPicker import DependencyPicker

            self.app.push_screen(DependencyPicker(self.services))
        elif selected in {"product-next", "product-previous"}:
            self.audit_offset = max(
                0, self.audit_offset + (20 if selected == "product-next" else -20)
            )
            self.on_mount()

    def _export(self, parent: Path | None) -> None:
        if parent is not None:
            name = hashlib.sha256(self.product["name"].encode("utf-8")).hexdigest()[:16]
            self._operation(
                "Exporting independent product",
                lambda: self.services.export_product(
                    self.product["name"], parent / f"product-{name}"
                ),
                lambda value: None,
            )

    def _operation(
        self, label: str, action: Callable[[], Any], ready: Callable[[Any], None]
    ) -> None:
        if self.busy:
            return
        self.busy = True
        self.query_one("#product-status", Static).update(label + "…")
        for button in self.query(".workspace-actions Button"):
            button.disabled = True
        app = self.app

        def run() -> None:
            try:
                value = action()
            except Exception as error:
                app.call_from_thread(
                    self._finished, label, ready, None, f"{type(error).__name__}: {error}"
                )
            else:
                app.call_from_thread(self._finished, label, ready, value, None)

        Thread(target=run, daemon=True, name="lambdaforge-product-view").start()

    def _finished(
        self, label: str, ready: Callable[[Any], None], value: Any, error: str | None
    ) -> None:
        self.busy = False
        if not self.is_mounted:
            return
        for button in self.query(".workspace-actions Button"):
            button.disabled = False
        if error is None:
            ready(value)
        self.query_one("#product-status", Static).update(
            f"{label} failed · {error}"
            if error
            else f"{label} complete · audit offset {self.audit_offset}"
        )


class StudyImportWorkspace(ResearchWorkspace):
    """Preview, human confirmation and revalidated import through ResultStore only."""

    def __init__(self, services: Any) -> None:
        self.services = services
        self.busy = False
        self.source: Path | None = None
        self.preview: dict[str, Any] | None = None
        super().__init__("Studies / Import Study")

    def compose_workspace(self) -> ComposeResult:
        yield Static(
            "Import portable Study evidence · no computation will be launched",
            classes="workspace-header",
        )
        yield Input(placeholder="Directory containing manifest.json", id="study-import-path")
        with Horizontal(classes="workspace-actions"):
            yield Button("Browse…", id="study-import-browse", flat=True)
            yield Button("Verify package", id="study-import-verify", flat=True, variant="primary")
            yield Button(
                "Import verified evidence…",
                id="study-import-apply",
                disabled=True,
                flat=True,
                variant="warning",
            )
        yield Static(
            "Select a package. Verification reads every included byte; it never executes code.",
            id="study-import-status",
            classes="status-line",
            markup=False,
        )
        yield VerticalScroll(
            Static("No package inspected yet.", id="study-import-plan", markup=False)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.busy:
            return
        if event.button.id == "study-import-browse":
            self.app.push_screen(
                ExportDirectoryPicker(
                    title="Import Study",
                    purpose="Select the existing package containing manifest.json.",
                    confirm_label="Select package",
                ),
                self._picked,
            )
        elif event.button.id == "study-import-verify":
            self.source = (
                Path(self.query_one("#study-import-path", Input).value).expanduser().absolute()
            )
            self._load(apply=False)
        elif event.button.id == "study-import-apply" and self.preview is not None:
            self.app.push_screen(
                ExactConfirmation("Register portable evidence, not restart training", self.preview),
                self._confirmed,
            )

    def _picked(self, path: Path | None) -> None:
        if path is not None:
            self.query_one("#study-import-path", Input).value = str(path)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "study-import-path":
            self.preview = None
            self.query_one("#study-import-apply", Button).disabled = True

    def _confirmed(self, confirmed: bool | None) -> None:
        if confirmed and self.source is not None and self.preview is not None:
            self._load(apply=True)

    def _load(self, *, apply: bool) -> None:
        if self.source is None or self.busy:
            return
        source, app = self.source, self.app
        self.busy = True
        self.query_one("#study-import-path", Input).disabled = True
        for button in self.query(".workspace-actions Button"):
            button.disabled = True
        self.query_one("#study-import-status", Static).update(
            "Registering verified evidence…" if apply else "Verifying exact inventory and products…"
        )

        def run() -> None:
            try:
                value = self.services.import_study(source, apply=apply)
            except Exception as error:
                app.call_from_thread(self._finished, None, f"{type(error).__name__}: {error}")
            else:
                app.call_from_thread(self._finished, value, None)

        Thread(target=run, daemon=True, name="lambdaforge-study-import").start()

    def _finished(self, value: dict[str, Any] | None, error: str | None) -> None:
        self.busy = False
        if not self.is_mounted:
            return
        self.query_one("#study-import-path", Input).disabled = False
        for button in self.query(".workspace-actions Button"):
            button.disabled = False
        self.preview = value if value and not value["applied"] else None
        self.query_one("#study-import-apply", Button).disabled = self.preview is None
        self.query_one("#study-import-status", Static).update(
            f"Import failed · {error}"
            if error
            else "Evidence registered in Studies as a read-only snapshot; no computation launched."
            if value and value["applied"]
            else "Verified. Review the plan, then confirm import."
        )
        self.query_one("#study-import-plan", Static).update(
            structured_text(value, heading="PORTABLE STUDY") if value else "No valid import plan."
        )
