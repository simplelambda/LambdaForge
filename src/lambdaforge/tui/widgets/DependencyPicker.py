"""Select exact native historical evidence without reading artifact bytes or editing YAML."""

from __future__ import annotations

from threading import Thread
from typing import Any

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Select, Static, TextArea


class DependencyPicker(ModalScreen[dict[str, Any] | None]):
    """Human catalog labels are presentation only; returned markers are already pinned."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, services: Any) -> None:
        super().__init__()
        self.services = services
        self._rows: list[dict[str, Any]] = []
        self._reference: dict[str, Any] | None = None
        self._busy = False
        self._offset = 0
        self._choice: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal"):
            yield Static("Historical evidence & independent products", classes="modal-title")
            yield Static(
                "Choose an exact Execution or sealed product. No model bytes are downloaded. "
                "Imported evidence remains read-only; a result is not an independent product."
            )
            yield DataTable(id="dependency-table", cursor_type="row", zebra_stripes=True)
            with Horizontal(classes="modal-actions"):
                yield Button("Previous 100", id="dependency-previous")
                yield Button("Next 100", id="dependency-next")
            yield Static("Loading metadata…", id="dependency-status", markup=False)
            yield Select(
                [("Whole Execution evidence", "whole")],
                value="whole",
                allow_blank=False,
                id="dependency-attempt",
                disabled=True,
            )
            yield Select(
                [("Whole product (contracted content)", "whole")],
                value="whole",
                allow_blank=False,
                id="dependency-artifact",
                disabled=True,
            )
            with VerticalScroll():
                yield Static(id="dependency-details", markup=False)
                yield TextArea("", read_only=True, language="yaml", id="dependency-declaration")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="dependency-cancel")
                yield Button("Copy typed YAML input", id="dependency-copy", disabled=True)
                yield Button(
                    "Use exact reference", id="dependency-use", disabled=True, variant="primary"
                )

    def on_mount(self) -> None:
        self.query_one("#dependency-table", DataTable).add_columns(
            "Source", "Name", "State / contract", "Exact identity"
        )
        self._load()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _load(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._choice = None
        self._reference = None
        self.query_one("#dependency-attempt", Select).disabled = True
        self.query_one("#dependency-artifact", Select).disabled = True
        for key in ("copy", "use"):
            self.query_one(f"#dependency-{key}", Button).disabled = True
        self.query_one("#dependency-status", Static).update("Loading compact catalog metadata…")

        def read() -> None:
            try:
                rows = self.services.dependency_choices(offset=self._offset)
                self.app.call_from_thread(self._loaded, rows, None)
            except Exception as error:
                self.app.call_from_thread(self._loaded, [], error)

        Thread(target=read, daemon=True, name="lambdaforge-dependency-catalog").start()

    def _loaded(self, rows: list[dict[str, Any]], error: Exception | None) -> None:
        self._busy = False
        if not self.is_mounted:
            return
        self._rows, self._reference = rows, None
        table = self.query_one("#dependency-table", DataTable)
        table.clear()
        for index, row in enumerate(rows):
            table.add_row(row["kind"], row["label"], row["state"], row["identity"], key=str(index))
        self.query_one("#dependency-status", Static).update(
            f"Catalog failed: {error}"
            if error
            else f"{len(rows)} choices · Enter selects exact evidence, never the newest name."
        )
        for key in ("copy", "use"):
            self.query_one(f"#dependency-{key}", Button).disabled = True

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "dependency-table" or self._busy:
            return
        event.stop()
        row = self._rows[event.cursor_row]
        self._choice = row
        self._busy = True
        self._reference = None
        for key in ("copy", "use"):
            self.query_one(f"#dependency-{key}", Button).disabled = True
        self.query_one("#dependency-status", Static).update("Resolving exact identity…")

        def resolve() -> None:
            try:
                value = self.services.dependency_reference(row)
                loader = getattr(self.services, "dependency_details", None)
                details = loader(row) if callable(loader) else None
                self.app.call_from_thread(self._resolved, value, None, details)
            except Exception as error:
                self.app.call_from_thread(self._resolved, None, error)

        Thread(target=resolve, daemon=True, name="lambdaforge-dependency-pin").start()

    def _resolved(
        self,
        value: dict[str, Any] | None,
        error: Exception | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._busy = False
        if not self.is_mounted:
            return
        self._reference = value
        if details is not None:
            from lambdaforge.tui.viewmodels import structured_text

            self.query_one("#dependency-details", Static).update(
                structured_text(details, heading="SELECTED SOURCE · METADATA ONLY", limit=60)
            )
            options = [("Whole Execution evidence", "whole")]
            options.extend(
                (
                    f"{row['run_id']} / {row['attempt_id']} · {row['status']}",
                    f"{row['run_id']}/{row['attempt_number']}",
                )
                for row in details.get("attempts", ())
            )
            selector = self.query_one("#dependency-attempt", Select)
            selector.set_options(options)
            selector.value = "whole"
            selector.disabled = len(options) == 1
            artifact_options = [("Whole product (contracted content)", "whole")]
            product = details.get("product", {})
            artifact_options.extend(
                (
                    f"{row['name']} · {row.get('role', 'artifact')} · "
                    f"{row.get('size_bytes', '—')} bytes",
                    str(index),
                )
                for index, row in enumerate(product.get("artifacts", ()))
            )
            self._artifacts = list(product.get("artifacts", ()))
            artifact_selector = self.query_one("#dependency-artifact", Select)
            artifact_selector.set_options(artifact_options)
            artifact_selector.value = "whole"
            artifact_selector.disabled = len(artifact_options) == 1
        self.query_one("#dependency-status", Static).update(
            f"Reference unavailable: {error}"
            if error
            else "Exact reference fixed. Destination availability is checked at preparation."
        )
        self.query_one("#dependency-declaration", TextArea).load_text(
            yaml.safe_dump(value, sort_keys=False) if value else ""
        )
        for key in ("copy", "use"):
            self.query_one(f"#dependency-{key}", Button).disabled = value is None

    def on_select_changed(self, event: Select.Changed) -> None:
        if (
            event.select.id not in {"dependency-attempt", "dependency-artifact"}
            or self._busy
            or not self._choice
        ):
            return
        event.stop()
        if event.value is Select.BLANK:
            return
        choice = dict(self._choice)
        options = {}
        if event.select.id == "dependency-artifact" and event.value != "whole":
            options = {"artifact": self._artifacts[int(str(event.value))]["name"]}
        elif event.select.id == "dependency-attempt" and event.value != "whole":
            run, attempt = str(event.value).rsplit("/", 1)
            options = {"run": run, "attempt": int(attempt)}
        self._busy = True
        self.query_one("#dependency-status", Static).update("Pinning selected Run / Attempt…")
        for key in ("copy", "use"):
            self.query_one(f"#dependency-{key}", Button).disabled = True

        def resolve() -> None:
            try:
                value = self.services.dependency_reference(choice, **options)
                self.app.call_from_thread(self._resolved, value, None)
            except Exception as error:
                self.app.call_from_thread(self._resolved, None, error)

        Thread(target=resolve, daemon=True, name="lambdaforge-dependency-attempt-pin").start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        selected = event.button.id
        if selected == "dependency-cancel":
            self.dismiss(None)
        elif selected == "dependency-copy" and self._reference:
            self.app.copy_to_clipboard(yaml.safe_dump(self._reference, sort_keys=False))
            self.query_one("#dependency-status", Static).update("Typed YAML input copied.")
        elif selected == "dependency-use" and self._reference:
            self.dismiss(self._reference)
        elif selected in {"dependency-previous", "dependency-next"} and not self._busy:
            self._offset = max(0, self._offset + (100 if selected == "dependency-next" else -100))
            self._load()


class MaterializationDialog(ModalScreen[None]):
    """Metadata preview, one explicit confirmation and native compressed apply."""

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, services: Any, selector: str, *, kind: str) -> None:
        super().__init__()
        self.services, self.selector, self.kind = services, selector, kind
        self._busy = False
        self._plan: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card wide-modal"):
            yield Static("Materialize exact dependency", classes="modal-title")
            yield Static("Copies a verified compressed package. Does not execute the producer.")
            yield Select(
                [(name, name) for name in self.services.cluster_names()],
                prompt="Destination cluster",
                id="placement-target",
            )
            yield Static("Select destination and preview.", id="placement-status", markup=False)
            with Horizontal(classes="modal-actions"):
                yield Button("Close", id="placement-close")
                yield Button("Preview", id="placement-preview")
                yield Button(
                    "Transfer & verify…", id="placement-apply", disabled=True, variant="success"
                )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "placement-target":
            self._plan = None
            self.query_one("#placement-apply", Button).disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "placement-close":
            self.dismiss(None)
        elif event.button.id == "placement-preview":
            self._operation(False)
        elif event.button.id == "placement-apply" and self._plan:
            from lambdaforge.tui.screens.Workspace import ExactConfirmation

            self.app.push_screen(
                ExactConfirmation("Transfer exact dependency", self._plan),
                lambda accepted: self._operation(True) if accepted else None,
            )

    def _operation(self, apply: bool) -> None:
        target = self.query_one("#placement-target", Select).value
        if self._busy or target is Select.BLANK:
            return
        if apply and (self._plan is None or self._plan["cluster"] != target):
            return
        self._busy = True
        captured = dict(self._plan or {})
        self.query_one("#placement-status", Static).update(
            "Compressing, transferring and verifying exact bytes…"
            if apply
            else "Inspecting dependency metadata; no bytes transferred…"
        )
        for button in self.query(Button):
            if button.id != "placement-close":
                button.disabled = True
        self.query_one("#placement-target", Select).disabled = True

        def run() -> None:
            try:
                value = self.services.materialize_dependency(
                    str(captured["identity"]) if apply else self.selector,
                    str(target),
                    kind=self.kind,
                    apply=apply,
                    expected_evidence_id=captured.get("evidence_id") if apply else None,
                )
                if apply and self._plan and value["identity"] != self._plan["identity"]:
                    raise ValueError("Dependency identity changed since preview.")
                self.app.call_from_thread(self._ready, value, None)
            except Exception as error:
                self.app.call_from_thread(self._ready, None, error)

        Thread(target=run, daemon=True, name="lambdaforge-dependency-transfer").start()

    def _ready(self, value: dict[str, Any] | None, error: Exception | None) -> None:
        self._busy = False
        if not self.is_mounted:
            return
        from lambdaforge.tui.viewmodels import structured_text

        self._plan = value if value and not value.get("applied") else None
        self.query_one("#placement-status", Static).update(
            f"Transfer failed: {error}"
            if error
            else structured_text(value, heading="DEPENDENCY PLAN")
        )
        self.query_one("#placement-target", Select).disabled = False
        self.query_one("#placement-preview", Button).disabled = False
        self.query_one("#placement-apply", Button).disabled = self._plan is None
