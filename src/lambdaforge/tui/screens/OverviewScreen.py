"""Research Console overview."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class OverviewScreen(DataScreen):
    """Refresh a compact cross-domain snapshot without blocking the UI loop."""

    TITLE = "Overview · what is running and what needs attention"

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(2.0, self._refresh_visible)

    def _refresh_visible(self) -> None:
        if self.display:
            self.reload()

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Kind", "Name", "Target", "State", "Progress / attention")
        work = value.get("work", {}).get("items", []) if isinstance(value, dict) else []
        for item in work:
            study = item.get("study")
            counts = study.get("counts", {}) if isinstance(study, dict) else {}
            waiting = counts.get("queued_runs", 0)
            progress = item.get("progress", "-")
            if waiting:
                progress = f"{progress} · {waiting} Run(s) waiting"
            table.add_row(
                "Study" if isinstance(study, dict) or item.get("study_expected") else "Work",
                str(item.get("name", "-")),
                str(item.get("cluster", "local")),
                str(item.get("state", "unknown")),
                str(progress),
            )
        clusters = value.get("clusters", []) if isinstance(value, dict) else []
        for item in clusters:
            table.add_row(
                "Cluster",
                str(item.get("cluster", "-")),
                str(item.get("cluster", "-")),
                "online" if item.get("online") else "unreachable",
                str(item.get("error") or item.get("summary") or ""),
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, dict):
            return "Overview details are unavailable."
        works = list(self.last_success.get("work", {}).get("items", ()))
        clusters = list(self.last_success.get("clusters", ()))
        values = [*works, *clusters]
        if not 0 <= row < len(values):
            return "Overview selection is unavailable."
        selected = values[row]
        if row < len(works):
            study = selected.get("study")
            counts = study.get("counts", {}) if isinstance(study, dict) else {}
            return (
                f"Work: {selected.get('name')}\nState: {selected.get('state')}\n"
                f"Cluster: {selected.get('cluster')}\nProgress: {selected.get('progress')}\n"
                f"Study Runs: active={counts.get('active_runs', 0)} "
                f"waiting={counts.get('queued_runs', 0)}"
            )
        return (
            f"Cluster: {selected.get('cluster')}\nOnline: {selected.get('online')}\n"
            f"Observed: {selected.get('observed', {})}\n"
            f"Available: {selected.get('available', {})}\n"
            f"Scheduler: {selected.get('scheduler')}\nError: {selected.get('error')}"
        )
