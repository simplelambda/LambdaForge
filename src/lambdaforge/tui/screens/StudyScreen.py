"""Study monitoring and analysis entry point."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected
from lambdaforge.tui.viewmodels import entity_key, format_value, objective_display_name


class StudyScreen(DataScreen):
    """Present adaptive/repeated-seed study state and censored evidence."""

    TITLE = "Studies · adaptive HPO, sweeps and repeated seeds"

    def __init__(self, loader: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(loader, *args, **kwargs)
        self._selected_key: str | None = None
        self._rendering = False

    def on_mount(self) -> None:
        super().on_mount()
        self.set_interval(2.0, self._refresh_visible)

    def _refresh_visible(self) -> None:
        if self.display and self.app.screen is self.app.screen_stack[0]:
            self.reload()

    @staticmethod
    def _studies(value: Any) -> list[dict[str, Any]]:
        return (
            [
                item
                for item in value.get("work", {}).get("items", [])
                if item.get("study_expected") or isinstance(item.get("study"), dict)
            ]
            if isinstance(value, dict)
            else []
        )

    def populate(self, table: DataTable[Any], value: Any) -> None:
        self._rendering = True
        table.clear(columns=True)
        table.add_columns(
            "Study", "State", "Objective", "Candidates", "Running", "Waiting", "Pruned"
        )
        studies = self._studies(value)
        for index, item in enumerate(studies):
            study = item.get("study")
            study = study if isinstance(study, dict) else {}
            counts = study.get("counts", {})
            objective = study.get("objective", {})
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("state", "unknown")),
                objective_display_name(objective),
                f"{counts.get('candidates', 0)}",
                f"{counts.get('active_runs', 0)}",
                f"{counts.get('queued_runs', 0)}",
                f"{counts.get('pruned_runs', 0)}",
                key=entity_key(item, kind="study", fallback_index=index),
            )
        row = next(
            (index for index, item in enumerate(studies) if self._key(item) == self._selected_key),
            0,
        )
        if studies:
            table.move_cursor(row=row)
            self._selected_key = self._key(studies[row])
        self._rendering = False
        self.query_one("#screen-detail", Static).update(self.detail(row))

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, dict):
            return "Study data is unavailable."
        studies = self._studies(self.last_success)
        if not 0 <= row < len(studies):
            return "Study selection is unavailable."
        item = studies[row]
        study = item.get("study") or {}
        counts = study.get("counts", {})
        objective = study.get("objective", {})
        candidates = study.get("candidates", [])
        comparable = [
            candidate
            for candidate in candidates
            if isinstance(candidate.get("selection_objective"), int | float)
        ]
        reverse = str(objective.get("mode", "max")) == "max"
        leader = (
            sorted(
                comparable,
                key=lambda candidate: float(candidate["selection_objective"]),
                reverse=reverse,
            )[0]
            if comparable
            else None
        )
        partial = [
            candidate
            for candidate in candidates
            if candidate.get("selection_objective") is None
            and isinstance(candidate.get("best_objective"), int | float)
        ]
        partial_leader = (
            sorted(
                partial,
                key=lambda candidate: float(candidate["best_objective"]),
                reverse=reverse,
            )[0]
            if partial
            else None
        )
        lines = [
            f"{item.get('name', 'Study')} · {str(item.get('state', 'unknown')).upper()} · "
            f"{item.get('cluster', 'local')}",
            f"Objective   {objective_display_name(objective)} ({objective.get('mode', '-')})",
            "Runs        "
            f"{counts.get('active_runs', 0)} active · {counts.get('queued_runs', 0)} waiting · "
            f"{counts.get('completed_runs', 0)} completed · {counts.get('pruned_runs', 0)} pruned",
            f"Trials      {counts.get('candidates', len(candidates))}",
            (
                f"Current lead  Trial {leader.get('trial')} · "
                f"{objective_display_name(objective)} "
                f"{format_value(leader.get('selection_objective'))}"
                if leader is not None
                else "Current lead  not comparable yet"
            ),
        ]
        admission = study.get("admission")
        current = admission.get("current") if isinstance(admission, dict) else None
        if isinstance(current, dict):
            status = str(current.get("status") or "monitoring")
            reason = current.get("reason")
            lines.append(f"Admission   {status}" + (f" · {reason}" if reason else ""))
        if partial_leader is not None:
            lines.append(
                f"Best partial/censored  Trial {partial_leader.get('trial')} · best observed "
                f"{format_value(partial_leader.get('best_objective'))}† · not a final seed mean"
            )
        lines.append("Enter/right opens Trials, seeds, HPO analysis, resources and logs.")
        return "\n".join(lines)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "screen-table" or self._rendering:
            return
        studies = self._studies(self.last_success)
        if 0 <= event.cursor_row < len(studies):
            self._selected_key = self._key(studies[event.cursor_row])
        self.query_one("#screen-detail", Static).update(self.detail(event.cursor_row))

    def open_row(self, row: int) -> None:
        if isinstance(self.last_success, dict):
            studies = self._studies(self.last_success)
            if 0 <= row < len(studies):
                self.post_message(EntitySelected("study", studies[row]))

    @staticmethod
    def _key(item: dict[str, Any]) -> str:
        return entity_key(item, kind="study")
