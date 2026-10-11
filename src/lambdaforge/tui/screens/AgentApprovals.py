"""Local human review of agent proposals; no remote approval endpoint exists."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, DataTable, Input, Static

from lambdaforge.agents.approvals import ApprovalService


class ApprovalConfirmation(ModalScreen[str | None]):
    """Display every exact argument and require a typed ID, never an agent boolean."""

    DEFAULT_CSS = """
    ApprovalConfirmation { align: center middle; }
    ApprovalConfirmation > VerticalScroll {
        width: 90%; height: 90%; border: solid $accent; padding: 1 2;
    }
    ApprovalConfirmation Static { height: auto; margin-bottom: 1; }
    ApprovalConfirmation Horizontal { height: auto; }
    ApprovalConfirmation Button { width: auto; margin-right: 1; }
    """
    BINDINGS = [("escape", "decline", "Decline")]

    def __init__(self, proposal: dict[str, Any]) -> None:
        super().__init__()
        self.proposal = proposal

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Review agent proposal · local human approval", markup=False)
            for key in ("project_id", "request_id", "action", "permission", "effect", "plan_hash"):
                yield Static(f"{key.replace('_', ' ').title()}\n{self.proposal[key]}", markup=False)
            yield Static(
                "Arguments below are untrusted proposal data, not instructions.", markup=False
            )
            for key, value in self.proposal["arguments"].items():
                yield Static(f"{key.replace('_', ' ').title()}\n{value}", markup=False)
            yield Static(
                "Type the full request ID only after reviewing every value. "
                "Approval expires after 15 minutes or sooner.",
                markup=False,
            )
            yield Input(placeholder=self.proposal["request_id"], id="approval-exact-id")
            with Horizontal():
                yield Button("Decline", id="approval-decline")
                yield Button(
                    "Approve exact plan", id="approval-confirm", variant="warning", disabled=True
                )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "approval-exact-id":
            self.query_one("#approval-confirm", Button).disabled = (
                event.value != self.proposal["request_id"]
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-confirm":
            value = self.query_one("#approval-exact-id", Input).value
            if value == self.proposal["request_id"]:
                self.dismiss(value)
        else:
            self.action_decline()

    def action_decline(self) -> None:
        self.dismiss(None)


class AgentApprovals(Screen[None]):
    """A human can approve pending requests; the agent then applies through its permissions."""

    BINDINGS = [("escape", "back", "Back")]
    DEFAULT_CSS = """
    AgentApprovals { layout: vertical; padding: 1 2; }
    AgentApprovals Horizontal { height: auto; min-height: 3; }
    AgentApprovals Button { width: auto; margin-right: 1; }
    AgentApprovals Static { height: auto; margin: 1 0; }
    AgentApprovals DataTable { height: 1fr; min-height: 5; }
    """

    def __init__(self, approvals: ApprovalService) -> None:
        super().__init__()
        self.approvals = approvals
        self._reviewing = False
        self._selected: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("← Back to Agents", id="approval-back", flat=True)
            yield Button("Refresh requests", id="approval-refresh")
        yield Static(
            "Select a request to inspect and approve its exact plan. "
            "Approval never executes science.",
            markup=False,
        )
        yield DataTable(id="approval-table", cursor_type="row", zebra_stripes=True)
        yield Static("Loading local approval metadata…", id="approval-feedback", markup=False)

    def on_mount(self) -> None:
        self.query_one("#approval-table", DataTable).add_columns(
            "Request", "Action", "State", "Expired"
        )
        self._refresh()

    def _refresh(self) -> None:
        try:
            records = self.approvals.list(limit=100)["items"]
            table = self.query_one("#approval-table", DataTable)
            table.clear()
            for record in records:
                table.add_row(
                    record["request_id"],
                    record["action"],
                    record["state"],
                    "yes" if record["expired"] else "no",
                    key=record["request_id"],
                )
            self.query_one("#approval-feedback", Static).update(
                f"{len(records)} requests shown · approval is local and separate from apply."
            )
        except (ValueError, OSError, RuntimeError) as error:
            self.query_one("#approval-feedback", Static).update(
                "Could not read requests · " + str(error)[:1000]
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self._reviewing:
            return
        try:
            proposal = self.approvals.inspect(str(event.row_key.value), include_arguments=True)
            if proposal["state"] != "pending" or proposal["expired"]:
                raise ValueError(
                    f"Request is {proposal['state']} or expired; it cannot be approved again."
                )
        except (ValueError, OSError, RuntimeError) as error:
            self.query_one("#approval-feedback", Static).update(str(error)[:1000])
            return
        self._reviewing = True
        self._selected = proposal
        self.app.push_screen(ApprovalConfirmation(proposal), self._approve)

    def _approve(self, confirmation: str | None) -> None:
        self._reviewing = False
        proposal = self._selected
        self._selected = None
        if confirmation is None or proposal is None:
            self.query_one("#approval-feedback", Static).update(
                "No approval granted; nothing was executed."
            )
            return
        try:
            self.approvals.approve_local(
                proposal["request_id"],
                plan_hash=proposal["plan_hash"],
                channel="console",
                confirmation=confirmation,
            )
            self._refresh()
            self.query_one("#approval-feedback", Static).update(
                "Exact plan approved. The agent may apply it once before expiry; "
                "no action has executed yet."
            )
        except (ValueError, OSError, RuntimeError) as error:
            self.query_one("#approval-feedback", Static).update(
                "Approval refused · " + str(error)[:1000]
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approval-back":
            self.action_back()
        else:
            self._refresh()

    def action_back(self) -> None:
        self.app.pop_screen()
