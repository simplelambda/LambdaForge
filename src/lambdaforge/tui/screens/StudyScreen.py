"""Study monitoring and analysis entry point."""

from __future__ import annotations

import json
from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class StudyScreen(DataScreen):
    """Present adaptive/repeated-seed study state and censored evidence."""

    TITLE = "Studies · adaptive HPO, sweeps and repeated seeds"

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns(
            "Study", "State", "Objective", "Candidates", "Running", "Waiting", "Pruned"
        )
        for item in value.get("work", {}).get("items", []):
            study = item.get("study")
            if not isinstance(study, dict):
                continue
            counts = study.get("counts", {})
            objective = study.get("objective", {})
            table.add_row(
                str(item.get("name", "-")),
                str(item.get("state", "unknown")),
                str(objective.get("metric", "utility")),
                f"{counts.get('candidates', 0)}",
                f"{counts.get('active_runs', 0)}",
                f"{counts.get('queued_runs', 0)}",
                f"{counts.get('pruned_runs', 0)}",
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, dict):
            return "Study data is unavailable."
        studies = [
            item
            for item in self.last_success.get("work", {}).get("items", [])
            if isinstance(item.get("study"), dict)
        ]
        if not 0 <= row < len(studies):
            return "Study selection is unavailable."
        study = studies[row]["study"]
        candidates = study.get("candidates", [])
        lines = ["Trials († = partial/censored; never a final seed mean)"]
        for candidate in candidates[:12]:
            value = candidate.get("selection_objective")
            if value is None:
                value = candidate.get("best_objective")
                suffix = "†" if candidate.get("partially_censored") else ""
            else:
                suffix = ""
            lines.append(
                f"Trial {candidate.get('trial')}  objective={value}{suffix}  "
                f"state={candidate.get('state')}  seeds={candidate.get('selection_seed_count', 0)}"
            )
        admission = study.get("admission", {}).get("current")
        if admission:
            lines.extend(("", "Admission", json.dumps(admission, indent=2)))
        return "\n".join(lines)
