"""Cluster profiles and operational actions."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class ClusterScreen(DataScreen):
    """List configured execution targets and reveal operational profile details."""

    TITLE = "Clusters · connection, scheduler, environment and GPU policy"

    def populate(self, table: DataTable[Any], value: Any) -> None:
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
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return "Cluster selection is unavailable."
        item = self.last_success[row]
        profile = item.get("profile", {})
        return (
            f"Target: {profile.get('name')}\n"
            f"Connection: {profile.get('transport')} {profile.get('host', 'local')}\n"
            f"Scheduler: {profile.get('scheduler')}\n"
            f"Workspace: {profile.get('workspace')}\n"
            f"Authentication: {item.get('authentication_status')}\n\n"
            "Use Ctrl+P for test, doctor, bootstrap, credentials, resources and storage."
        )
