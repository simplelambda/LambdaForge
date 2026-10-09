"""Immutable DatasetVersion browser."""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable, Static

from lambdaforge.tui.screens.Base import DataScreen, EntitySelected


class DatasetScreen(DataScreen):
    """Browse immutable dataset versions, placements and lineage."""

    TITLE = "Datasets · versions, members, placements and lineage"

    def _loaded(self, value: Any) -> None:
        super()._loaded(value)
        warnings = {warning for item in value for warning in item.get("discovery_warnings", ())}
        conflicts = any(item.get("inventory_conflict") for item in value)
        if warnings or conflicts:
            self.query_one("#screen-status", Static).update(
                "Identity conflicts detected · inspect the selected dataset details."
                if conflicts
                else "Partial inventory · remote discovery warnings in dataset details."
            )

    def populate(self, table: DataTable[Any], value: Any) -> None:
        table.clear(columns=True)
        table.add_columns("Dataset", "Members", "Size", "Locations", "Identity")
        for item in value:
            placements = item.get("placements", [])
            table.add_row(
                f"{item.get('name')}@{item.get('version')}",
                str(item.get("sample_count", 0)),
                self._bytes(self._dataset_size(item)),
                ", ".join(str(entry.get("cluster")) for entry in placements) or "none",
                ("REFERENCE · " if item.get("project_reference") else "")
                + ("CONFLICT · " if item.get("inventory_conflict") else "")
                + str(item.get("dataset_id", ""))[-12:],
                key=f"{item.get('name')}@{item.get('version')}:{item.get('dataset_id')}",
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
            + (
                "CONFLICT: the same version has different content across indexes. "
                "Open Manage copies to explicitly choose a reference or retire one exact copy.\n"
                if item.get("inventory_conflict")
                else ""
            )
            + (
                "PROJECT REFERENCE: used for future dataset resolution, not an equivalence claim.\n"
                if item.get("project_reference")
                else ""
            )
            + (
                "Discovery incomplete: " + "; ".join(item.get("discovery_warnings", ())) + "\n"
                if item.get("discovery_warnings")
                else ""
            )
            + "Enter/right opens bounded members, statistics, integrity and lineage views."
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
