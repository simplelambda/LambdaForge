"""Read-only local registry dashboard."""

from __future__ import annotations

import html
from pathlib import Path

from lambdaforge.registry.ExperimentRegistry import ExperimentRegistry


class LocalDashboard:
    """Generate a dependency-free static dashboard from registry records."""

    def build(self, root: str | Path, destination: str | Path) -> Path:
        """Write a read-only HTML table linking run directories and metrics."""
        records = ExperimentRegistry(root).query()
        rows = []
        for record in records:
            cells = (
                record["name"],
                record["variant"],
                record["seed"],
                record["status"],
                record["attempt_id"],
                record["metrics"],
                record["run_dir"],
            )
            rows.append(
                "<tr>"
                + "".join(f"<td><pre>{html.escape(str(value))}</pre></td>" for value in cells)
                + "</tr>"
            )
        document = (
            "<!doctype html><meta charset='utf-8'><title>LambdaForge registry</title>"
            "<h1>LambdaForge registry</h1><p>Read-only snapshot.</p><table><thead><tr>"
            "<th>Name</th><th>Variant</th><th>Seed</th><th>Status</th><th>Attempt</th>"
            "<th>Metrics</th><th>Run</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
        return path
