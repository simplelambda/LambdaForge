"""Immutable DatasetVersion browser."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen


class DatasetScreen(DataScreen):
    """Browse immutable dataset versions, placements and lineage."""

    TITLE = "Datasets · versions, members, placements and lineage"

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Dataset", "Members", "Size", "Locations", "Lineage")
        for item in value:
            placements = item.get("placements", [])
            table.add_row(
                f"{item.get('name')}@{item.get('version')}",
                str(item.get("sample_count", 0)),
                str(item.get("size_bytes", "-")),
                ", ".join(str(entry.get("cluster")) for entry in placements),
                ", ".join(map(str, item.get("lineage", []))),
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return "Dataset selection is unavailable."
        item = self.last_success[row]
        return (
            f"{item.get('name')}@{item.get('version')}\n"
            f"Content identity: {item.get('dataset_id')}\n"
            f"Members: {item.get('sample_count')}\n"
            f"Partitions: {item.get('partitions', {})}\n"
            f"Lineage: {item.get('lineage', [])}\n\n"
            "Lifecycle actions preserve preview/apply semantics and require confirmation."
        )
