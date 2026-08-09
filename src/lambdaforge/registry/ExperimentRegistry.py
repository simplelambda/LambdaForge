"""Read-only registry built on canonical result artifacts."""

from __future__ import annotations

import csv
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.registry.RegistryQuery import RegistryQuery


class ExperimentRegistry:
    """Query existing results without introducing a second mutable database."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def query(self, query: RegistryQuery | None = None) -> tuple[dict[str, Any], ...]:
        """Return deterministically ordered records enriched from their config snapshots."""
        query = query or RegistryQuery()
        output: list[dict[str, Any]] = []
        for record in ResultCatalog(self.root).records():
            payload = record.to_dict()
            config = self._config(Path(record.run_dir) / "config.yaml")
            metadata = self._mapping(config.get("metadata"))
            result_metadata = self._mapping(record.result.get("metadata"))
            metadata = {**metadata, **result_metadata}
            tags = self._tags(config, metadata)
            payload.update({"tags": tags, "metadata": metadata, "config": config})
            if query.statuses and payload["status"] not in query.statuses:
                continue
            if query.names and payload["name"] not in query.names:
                continue
            if query.fingerprint and payload["config_fingerprint"] != query.fingerprint:
                continue
            if not set(query.tags).issubset(tags):
                continue
            if any(metadata.get(key) != value for key, value in query.metadata.items()):
                continue
            output.append(payload)
        return tuple(output)

    def export(self, path: str | Path, *, query: RegistryQuery | None = None) -> Path:
        """Export query results as JSON, CSV or optional Parquet."""
        destination = Path(path)
        records = self.query(query)
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix == ".json":
            destination.write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        elif suffix == ".csv":
            fields = (
                "attempt_id",
                "name",
                "variant",
                "seed",
                "status",
                "config_fingerprint",
                "run_dir",
                "tags",
                "metrics",
            )
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    writer.writerow(
                        {
                            key: json.dumps(record[key], sort_keys=True)
                            if key in {"tags", "metrics"}
                            else record.get(key)
                            for key in fields
                        }
                    )
        elif suffix == ".parquet":
            try:
                pandas = importlib.import_module("pandas")
            except ImportError as error:
                raise ImportError("Parquet export requires pandas and a Parquet engine.") from error
            pandas.DataFrame(
                [
                    {
                        **record,
                        "config": json.dumps(record["config"]),
                        "metadata": json.dumps(record["metadata"]),
                        "metrics": json.dumps(record["metrics"]),
                        "tags": json.dumps(record["tags"]),
                    }
                    for record in records
                ]
            ).to_parquet(destination, index=False)
        else:
            raise ValueError("Registry export path must end in .json, .csv or .parquet.")
        return destination

    @staticmethod
    def _config(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _tags(config: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str, ...]:
        raw = metadata.get("tags", config.get("tags", ()))
        if isinstance(raw, str):
            raw = (raw,)
        return tuple(sorted(str(value) for value in raw)) if isinstance(raw, (list, tuple)) else ()
