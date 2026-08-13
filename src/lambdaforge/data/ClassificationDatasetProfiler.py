"""Explicit-schema classification statistics profiler."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetProfiler import DatasetProfiler
from lambdaforge.data.DatasetRecord import DatasetRecord


class ClassificationDatasetProfiler(DatasetProfiler):
    """Count declared target values in CSV files; never infer a target by name."""

    def profile(
        self, root: Path, record: DatasetRecord, schema: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del record
        if schema.get("task") != "classification" or not schema.get("target"):
            raise ValueError("Classification profiling requires task=classification and target.")
        target = str(schema["target"])
        labels = tuple(
            str(value) for value in schema.get("class_labels", schema.get("classes", ()))
        )
        explicit = schema.get("file")
        pattern = str(schema.get("files", "*.csv"))
        paths = (
            ((root / str(explicit)).resolve(),)
            if explicit is not None
            else tuple(sorted(root.rglob(pattern)))
        )
        counts: Counter[str] = Counter()
        missing = 0
        for path in paths:
            if not path.is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
                raise ValueError(f"Unsafe or missing classification data file: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if target not in row:
                        raise KeyError(f"Declared classification target {target!r} is absent.")
                    value = row[target]
                    if value is None or not str(value).strip():
                        missing += 1
                    else:
                        counts[str(value)] += 1
        ordered = labels or tuple(sorted(counts))
        total = sum(counts.values())
        positive = tuple(value for value in (counts[label] for label in ordered) if value > 0)
        return {
            "class_distribution": {label: counts[label] for label in ordered},
            "class_proportions": {
                label: counts[label] / total if total else 0.0 for label in ordered
            },
            "missing_targets": missing,
            "imbalance_ratio": max(positive) / min(positive) if positive else None,
        }
