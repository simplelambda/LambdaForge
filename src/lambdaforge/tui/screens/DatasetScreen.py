"""Immutable DatasetVersion browser."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected


class DatasetScreen(DataScreen):
    """Browse immutable dataset versions, placements and lineage."""

    TITLE = "Datasets · versions, members, placements and lineage"

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Dataset", "Members", "Size", "Locations", "Parents")
        for item in value:
            placements = item.get("placements", [])
            table.add_row(
                f"{item.get('name')}@{item.get('version')}",
                str(item.get("sample_count", 0)),
                self._bytes(self._dataset_size(item)),
                ", ".join(str(entry.get("cluster")) for entry in placements) or "none",
                str(len(item.get("lineage", ()))),
                key=f"{item.get('name')}@{item.get('version')}",
            )

    def detail(self, row: int) -> str:
        if not isinstance(self.last_success, list) or not 0 <= row < len(self.last_success):
            return "Dataset selection is unavailable."
        item = self.last_success[row]
        placements = item.get("placements", ())
        locations = ", ".join(
            str(entry.get("cluster", "unknown")) for entry in placements if isinstance(entry, dict)
        )
        partitions = item.get("partitions", {})
        lineage = item.get("lineage", ())
        lineage = lineage if isinstance(lineage, list | tuple) else ()
        return (
            f"{item.get('name')}@{item.get('version')}  ·  IMMUTABLE\n"
            f"Scale        {item.get('sample_count', 0)} members · "
            f"{self._bytes(self._dataset_size(item))}\n"
            f"Organization {len(partitions) if isinstance(partitions, dict) else 0} partitions · "
            f"{len(lineage)} lineage inputs\n"
            f"Locations    {locations or 'not materialized'}\n"
            f"Content ID   {item.get('content_id', item.get('dataset_id', 'unavailable'))}\n"
            "Enter/right opens bounded members, statistics, integrity and lineage views."
        )

    def open_row(self, row: int) -> None:
        if isinstance(self.last_success, list) and 0 <= row < len(self.last_success):
            self.post_message(EntitySelected("dataset", self.last_success[row]))

    @staticmethod
    def _bytes(value: Any) -> str:
        if not isinstance(value, int | float):
            return "unknown"
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(size) < 1024 or unit == "TiB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return "unknown"

    @staticmethod
    def _dataset_size(item: dict[str, Any]) -> int | None:
        direct = item.get("size_bytes")
        if isinstance(direct, int):
            return direct
        for placement in item.get("placements", ()):
            if isinstance(placement, dict) and isinstance(placement.get("size_bytes"), int):
                return int(placement["size_bytes"])
        return None
