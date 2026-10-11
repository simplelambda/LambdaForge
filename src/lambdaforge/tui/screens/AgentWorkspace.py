"""Agent integration status and explicit asset setup through the shared Python service."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Thread
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Input, Select, Static

from lambdaforge.agents import AgentServices, IntegrationService


class AgentWorkspace(Screen[None]):
    """No embedded chatbot or CLI subprocess; sensitive MCP tools remain unavailable."""

    BINDINGS = [("escape", "back", "Back")]
    DEFAULT_CSS = """
    AgentWorkspace { layout: vertical; padding: 1 2; }
    AgentWorkspace Horizontal { height: auto; min-height: 3; }
    AgentWorkspace Button { width: auto; margin-right: 1; }
    AgentWorkspace Select { width: 1fr; }
    AgentWorkspace Static { height: auto; margin: 1 0; }
    AgentWorkspace DataTable { height: 1fr; min-height: 6; }
    AgentWorkspace VerticalScroll { height: 1fr; }
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.service = AgentServices(root)
        self.integration = IntegrationService(root)
        self._busy = False
        self._plan: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("← Back", id="agents-back", flat=True)
            yield Button("Review agent requests…", id="agents-approvals")
            yield Static("Agents · pinned project, installed Skills and portable knowledge")
        yield Static("Loading integration status…", id="agents-status", markup=False)
        with Horizontal():
            yield Select(
                [("Portable Skills", "portable"), ("Codex example", "codex")],
                value="portable",
                allow_blank=False,
                id="agents-mode",
            )
            yield Select(
                [
                    ("No memory setup", "none"),
                    ("Markdown", "markdown"),
                    ("Obsidian vault", "obsidian"),
                ],
                value="none",
                allow_blank=False,
                id="agents-memory",
            )
            yield Button("Preview setup / sync", id="agents-preview")
        yield Static(
            "Setup preserves custom guides, Skills and all notes. Dependencies and global "
            "client configuration are never changed here. Agent note edits need a separate local "
            "approval and operator permission. Scientific mutations remain disabled; local HTTP "
            "requires explicit paired setup outside this panel.",
            markup=False,
        )
        yield DataTable(id="agents-table", cursor_type="row", zebra_stripes=True)
        with VerticalScroll():
            yield Static("Select a Skill to read its procedure.", id="agents-detail", markup=False)
            yield Input(placeholder="Search project knowledge", id="agents-search")
            yield Button("Search notes", id="agents-search-button")
            yield Static("", id="agents-search-results", markup=False)

    def on_mount(self) -> None:
        from lambdaforge.agents.integration import skill_metadata, skill_names
        from lambdaforge.agents.mcp_server import dependency_status

        status = self.integration.status()
        sdk = dependency_status()
        self.query_one("#agents-status", Static).update(
            f"Project {status['project_id']} · {status['installed_skills']} installed Skills · "
            f"MCP {'ready' if sdk['available'] else 'optional SDK missing/incompatible'} · "
            "default permissions READ / PLAN"
        )
        table = self.query_one("#agents-table", DataTable)
        table.add_columns("Skill", "Use when")
        for name in skill_names():
            table.add_row(name, skill_metadata(name)["description"], key=name)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        from lambdaforge.agents.integration import skill_metadata

        self.query_one("#agents-detail", Static).update(
            skill_metadata(str(event.row_key.value))["document"]
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agents-back":
            self.action_back()
        elif event.button.id == "agents-approvals":
            from lambdaforge.agents.approvals import ApprovalService
            from lambdaforge.tui.screens.AgentApprovals import AgentApprovals

            self.app.push_screen(AgentApprovals(ApprovalService(self.service.scope)))
        elif event.button.id == "agents-preview":
            if self._busy:
                return
            agents = str(self.query_one("#agents-mode", Select).value)
            knowledge = str(self.query_one("#agents-memory", Select).value)
            self._run(
                lambda: self.integration.preview(agents=agents, knowledge=knowledge), self._confirm
            )
        elif event.button.id == "agents-search-button":
            query = self.query_one("#agents-search", Input).value
            self._run(lambda: asdict(self.service.knowledge_search(query)), self._show_notes)

    def _confirm(self, plan: dict[str, Any]) -> None:
        from lambdaforge.tui.screens.Workspace import ExactConfirmation

        self._plan = plan
        counts = {
            kind: sum(e["action"] == kind for e in plan["files"])
            for kind in ("create", "update", "unchanged", "preserve_custom")
        }
        self.app.push_screen(
            ExactConfirmation(
                "Prepare project agent assets?",
                {
                    "project": plan["project_id"],
                    "files": counts,
                    "agents": plan["agents"],
                    "knowledge": plan["knowledge"],
                    "notes_modified": False,
                    "global_client_configuration_modified": False,
                },
            ),
            self._apply,
        )

    def _apply(self, approved: bool | None) -> None:
        plan = self._plan
        if approved and plan is not None:
            self._run(
                lambda: self.integration.apply(
                    plan_hash=plan["plan_hash"], agents=plan["agents"], knowledge=plan["knowledge"]
                ),
                lambda result: self.query_one("#agents-status", Static).update(
                    f"Setup finished · {len(result['custom_files_preserved'])} custom files kept. "
                    "Review generated instructions/client example; no science was launched."
                ),
            )

    def _show_notes(self, reply: dict[str, Any]) -> None:
        text = (
            reply["error"]["message"]
            if reply["status"] == "error"
            else "\n\n".join(
                f"{item['metadata']['title']} · {item['metadata']['kind']}\n{item['snippet']}"
                for item in reply["data"]["items"]
            )
            or "No matching notes."
        )
        self.query_one("#agents-search-results", Static).update(text)

    def _run(self, function: Any, callback: Any) -> None:
        if self._busy:
            return
        self._busy = True
        self.query_one("#agents-status", Static).update("Working on project metadata…")

        def work() -> None:
            try:
                result = function()
            except (ValueError, OSError, RuntimeError) as error:
                self.app.call_from_thread(self._failed, str(error))
            else:
                self.app.call_from_thread(self._finished, callback, result)

        Thread(target=work, daemon=True).start()

    def _finished(self, callback: Any, value: Any) -> None:
        self._busy = False
        if not self.is_mounted:
            return
        self.query_one("#agents-status", Static).update("Project metadata ready.")
        callback(value)

    def _failed(self, message: str) -> None:
        self._busy = False
        if not self.is_mounted:
            return
        self.query_one("#agents-status", Static).update("Operation failed · " + message[:1000])

    def action_back(self) -> None:
        self.app.pop_screen()
