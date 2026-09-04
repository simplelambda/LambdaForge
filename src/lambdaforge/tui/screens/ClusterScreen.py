"""Cluster profiles and operational actions."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label, LoadingIndicator, Static

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.widgets import ResourceDashboard


class ClusterScreen(DataScreen):
    """List configured execution targets and reveal operational profile details."""

    TITLE = "Clusters · connection, scheduler, environment and GPU policy"

    def __init__(self, loader: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loader, *args, **kwargs)
        self._selected_name: str | None = None
        self._rendering = False

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.TITLE, classes="screen-title"),
            Static("Loading…", id="screen-status", classes="status-line"),
            LoadingIndicator(id="screen-loading"),
            DataTable(id="screen-table", cursor_type="row", zebra_stripes=True),
            Static(
                "Select a cluster to see capacity, policy and live usage.",
                id="screen-detail",
                classes="detail-panel",
            ),
            ResourceDashboard(id="cluster-screen-dashboard"),
            Static(
                "Waiting for the first update",
                id="screen-freshness",
                classes="freshness-line",
            ),
        )

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(2.0, self._refresh_visible)

    def _refresh_visible(self) -> None:
        if self.display and self.app.screen is self.app.screen_stack[0]:
            self.reload()

    def populate(self, table: DataTable[Any], value: Any) -> None:
        self._rendering = True
        table.clear(columns=True)
        table.add_columns("Name", "Host", "Scheduler", "Environment", "GPU access", "Source")
        for item in value:
            profile = item.get("profile", {})
            table.add_row(
                str(profile.get("name", "-")),
                str(profile.get("host", "local")),
                str(profile.get("scheduler", "local")),
                str(
                    profile.get("environment", {}).get("provider", "managed")
                    if isinstance(profile.get("environment"), dict)
                    else profile.get("environment", "managed")
                ),
                str(
                    profile.get("gpu_access", {}).get("mode", "auto")
                    if isinstance(profile.get("gpu_access"), dict)
                    else "auto"
                ),
                str(item.get("source", "-")),
                key=str(profile.get("name", "-")),
            )
        row = next(
            (
                index
                for index, item in enumerate(value)
                if str(item.get("profile", {}).get("name", "")) == self._selected_name
            ),
            0,
        )
        if value:
            table.move_cursor(row=row)
        self._rendering = False
        self._show_selection(row)

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return "Cluster selection is unavailable."
        item = self.last_success[row]
        profile = item.get("profile", {})
        resources = item.get("resources", {})
        environment = profile.get("environment", "managed")
        if isinstance(environment, dict):
            environment = environment.get("provider", environment.get("mode", "managed"))
        return (
            f"{profile.get('name')}  ·  "
            f"{'ONLINE' if resources.get('online') else 'OFFLINE'}  ·  "
            f"{profile.get('scheduler')} scheduler\n"
            f"Connection    {profile.get('transport')} · "
            f"{profile.get('user') or 'current user'}@{profile.get('host', 'local')}\n"
            f"Workspace     {profile.get('workspace')}\n"
            f"Environment   {environment}\n"
            f"Authentication {item.get('authentication_status')}\n"
            "Enter/right opens operational details and safe actions."
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "screen-table" and not self._rendering:
            self._show_selection(event.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "screen-table":
            self._show_selection(event.cursor_row)
            self.open_row(event.cursor_row)

    def _show_selection(self, row: int) -> None:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return
        item = self.last_success[row]
        profile = item.get("profile", {})
        resources = item.get("resources", {})
        name = str(profile.get("name", "-"))
        self._selected_name = name
        self.query_one("#screen-detail", Static).update(self.detail(row))
        self.query_one("#cluster-screen-dashboard", ResourceDashboard).show_cluster(
            name,
            resources.get("observed", {}) if isinstance(resources, dict) else {},
            resources.get("personal", {}) if isinstance(resources, dict) else {},
            sample_id=(
                str(resources.get("observed_at_utc"))
                if isinstance(resources, dict) and resources.get("observed_at_utc")
                else None
            ),
        )

    def open_row(self, row: int) -> None:
        if isinstance(self.last_success, list) and 0 <= row < len(self.last_success):
            profile = self.last_success[row].get("profile", {})
            self.post_message(EntitySelected("cluster", str(profile.get("name", ""))))
