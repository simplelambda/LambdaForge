"""Cross-experiment semantic and metric comparison."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.configuration.ConfigurationDiff import ConfigurationDiff


class ExperimentComparator:
    """Compute objective tables/effects without generating scientific conclusions."""

    def compare(
        self,
        groups: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        metric: str,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Compare selected registry records grouped by an explicit user label."""
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one.")
        critical = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2)
        table: dict[str, dict[str, Any]] = {}
        configs: dict[str, Mapping[str, Any]] = {}
        for label, records in groups.items():
            values = [
                float(record["metrics"][metric])
                for record in records
                if metric in record.get("metrics", {})
            ]
            if not values:
                raise ValueError(f"Group {label!r} has no numeric metric {metric!r}.")
            mean = statistics.fmean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            half_width = critical * stdev / len(values) ** 0.5
            table[label] = {
                "count": len(values),
                "mean": mean,
                "stdev": stdev,
                "confidence_level": confidence_level,
                "confidence_interval": [mean - half_width, mean + half_width],
                "values": values,
            }
            first = next(iter(records), None)
            configs[label] = first.get("config", {}) if first else {}
        labels = tuple(table)
        effects = {
            f"{left} - {right}": table[left]["mean"] - table[right]["mean"]
            for index, left in enumerate(labels)
            for right in labels[index + 1 :]
        }
        config_diffs = (
            {
                f"{labels[0]} vs {label}": ConfigurationDiff().compare(
                    configs[labels[0]], configs[label]
                )
                for label in labels[1:]
            }
            if labels
            else {}
        )
        return {
            "metric": metric,
            "groups": table,
            "mean_effects": effects,
            "configuration_diffs": config_diffs,
        }
