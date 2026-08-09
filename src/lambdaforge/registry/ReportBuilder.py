"""Objective Markdown/HTML report generation."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ReportBuilder:
    """Render supplied registry/comparison facts without inferred conclusions."""

    def write(
        self,
        comparison: Mapping[str, Any],
        path: str | Path,
        *,
        title: str = "LambdaForge comparison",
        make_plot: bool = True,
    ) -> Path:
        """Write Markdown or self-contained HTML according to destination suffix."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        groups = comparison.get("groups", {})
        rows = [
            f"| {label} | {value['count']} | {value['mean']:.8g} | {value['stdev']:.8g} |"
            for label, value in groups.items()
        ]
        markdown = (
            f"# {title}\n\nMetric: `{comparison.get('metric')}`\n\n"
            "| Group | N | Mean | SD |\n|---|---:|---:|---:|\n"
            + "\n".join(rows)
            + "\n\n## Machine-readable details\n\n```json\n"
            + json.dumps(comparison, indent=2, sort_keys=True)
            + "\n```\n"
        )
        plot = self._write_plot(groups, destination) if make_plot else None
        if plot is not None:
            markdown = markdown.replace(
                "## Machine-readable details",
                f"![Objective means and intervals]({plot.name})\n\n## Machine-readable details",
            )
        if destination.suffix.lower() == ".md":
            destination.write_text(markdown, encoding="utf-8")
        elif destination.suffix.lower() == ".html":
            destination.write_text(
                "<!doctype html><meta charset='utf-8'>"
                f"<title>{html.escape(title)}</title>"
                f"<body><pre>{html.escape(markdown)}</pre></body>",
                encoding="utf-8",
            )
        else:
            raise ValueError("Reports must use a .md or .html suffix.")
        return destination

    @staticmethod
    def _write_plot(groups: Mapping[str, Any], destination: Path) -> Path | None:
        """Write an objective mean/interval figure when groups are available."""
        if not groups:
            return None
        import matplotlib.pyplot as plt

        labels = tuple(groups)
        means = [float(groups[label]["mean"]) for label in labels]
        intervals = [groups[label]["confidence_interval"] for label in labels]
        errors = [
            [mean - float(interval[0]) for mean, interval in zip(means, intervals, strict=True)],
            [float(interval[1]) - mean for mean, interval in zip(means, intervals, strict=True)],
        ]
        figure, axis = plt.subplots(figsize=(max(5, len(labels) * 1.2), 3.5))
        axis.errorbar(labels, means, yerr=errors, fmt="o", capsize=4)
        axis.set_ylabel("Objective")
        axis.set_title("Group means and configured confidence intervals")
        figure.tight_layout()
        path = destination.with_name(f"{destination.stem}-means.png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path
