"""Pure bounded view-model transformations for the Research Console."""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Mapping, Sequence
from typing import Any


def epoch_rows(curves: Mapping[str, Any]) -> list[tuple[int, dict[str, float]]]:
    """Pivot scalar curves into one deterministic row per observed epoch."""
    rows: dict[int, dict[str, float]] = {}
    for raw_name, raw_points in curves.items():
        if not isinstance(raw_points, Sequence) or isinstance(raw_points, str | bytes):
            continue
        for point in raw_points:
            if not isinstance(point, Mapping):
                continue
            step, value = point.get("step"), point.get("value")
            if (
                isinstance(step, int)
                and not isinstance(step, bool)
                and isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                rows.setdefault(step, {})[str(raw_name)] = float(value)
    return [(step, rows[step]) for step in sorted(rows)]


def visible_metrics(
    curves: Mapping[str, Any],
    chart_filter: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> list[str]:
    """Select chart names without hiding metrics from tables or epoch detail."""
    names = sorted(map(str, curves))
    includes = [str(value) for value in chart_filter.get("include", ())]
    excludes = [str(value) for value in chart_filter.get("exclude", ())]
    if includes:
        names = [name for name in names if any(fnmatch.fnmatch(name, rule) for rule in includes)]
    if excludes:
        names = [
            name for name in names if not any(fnmatch.fnmatch(name, rule) for rule in excludes)
        ]
    priorities = ("val_", "train_loss", "loss")
    names.sort(
        key=lambda name: (
            next((i for i, p in enumerate(priorities) if name.startswith(p)), 9),
            name,
        )
    )
    return names[:limit] if limit is not None else names


def format_duration(value: Any) -> str:
    """Format optional seconds without inventing precision."""
    if not isinstance(value, int | float) or value < 0:
        return "unavailable"
    seconds = int(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_value(value: Any) -> str:
    """Render compact scientific scalar or explicit unavailability."""
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def structured_text(value: Any, *, heading: str | None = None, limit: int = 160) -> str:
    """Render bounded nested evidence as human labels instead of serialized JSON."""
    lines: list[str] = [heading] if heading else []

    def append(item: Any, depth: int, label: str | None = None) -> None:
        if len(lines) >= limit:
            return
        prefix = "  " * depth
        readable = label.replace("_", " ").title() if label else None
        if isinstance(item, Mapping):
            if readable:
                lines.append(f"{prefix}{readable}")
            if not item:
                lines.append(f"{prefix}  None recorded")
            for key, nested in item.items():
                append(nested, depth + (1 if readable else 0), str(key))
            return
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            if readable:
                lines.append(f"{prefix}{readable}  ·  {len(item)} item(s)")
            if not item:
                lines.append(f"{prefix}  None recorded")
            for index, nested in enumerate(item, 1):
                if isinstance(nested, Mapping):
                    lines.append(f"{prefix}  Item {index}")
                    append(nested, depth + 2)
                elif isinstance(nested, Sequence) and not isinstance(nested, str | bytes):
                    append(nested, depth + 1, f"Item {index}")
                else:
                    lines.append(f"{prefix}  • {format_value(nested)}")
            return
        lines.append(f"{prefix}{readable or 'Value':24} {format_value(item)}")

    append(value, 0)
    if len(lines) >= limit:
        lines.append(f"… bounded to {limit} display lines")
    return "\n".join(lines) or "No persisted evidence."


def confirmation_text(preview: Mapping[str, Any], *, limit: int = 200) -> str:
    """Present an exact mutation preview as a warning with semantic sections."""
    preferred_sections = (
        ("will_change", "WILL CHANGE"),
        ("will_remove", "WILL REMOVE"),
        ("will_preserve", "WILL PRESERVE"),
        ("placements", "AFFECTED PLACEMENTS"),
        ("reasons", "SAFETY NOTES"),
    )
    section_keys = {key for key, _ in preferred_sections}
    context = {key: value for key, value in preview.items() if key not in section_keys}
    lines = [
        "This action changes persisted state.",
        "Review the exact target and scope before applying it.",
    ]
    if context:
        lines.extend(("", structured_text(context, heading="TARGET AND OPERATION", limit=limit)))
    for key, heading in preferred_sections:
        value = preview.get(key)
        if value in (None, (), [], {}):
            continue
        lines.extend(("", structured_text(value, heading=heading, limit=limit)))
    return "\n".join(lines)


_METRIC_NAMES = {
    "__lambdaforge_utility__": "Composite selection score",
    "lambdaforge_utility": "Composite selection score",
    "epoch_time_s": "Epoch time (s)",
    "validation_time_s": "Validation time (s)",
    "max_gpu_reserved_memory": "Peak reserved GPU memory",
    "gpu_peak_reserved_mb": "Peak reserved GPU memory (MiB)",
    "gpu_reserved_mb": "Reserved GPU memory (MiB)",
    "gpu_mem_mb": "GPU memory (MiB)",
    "cpu_percent": "CPU usage (%)",
    "ram_mb": "RAM (MiB)",
}

_ACRONYMS = {
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "f1": "F1",
    "gpu": "GPU",
    "mcc": "MCC",
    "ram": "RAM",
}


def metric_display_name(name: Any, aliases: Mapping[str, Any] | None = None) -> str:
    """Return a concise human label while preserving the raw metric as its identity."""
    raw = str(name)
    if aliases and raw in aliases:
        return str(aliases[raw])
    if raw in _METRIC_NAMES:
        return _METRIC_NAMES[raw]
    prefix = ""
    remainder = raw
    if remainder.startswith("val_"):
        remainder = remainder[4:]
    elif remainder.startswith("train_"):
        prefix, remainder = "Train ", remainder[6:]
    elif remainder.startswith("test_"):
        prefix, remainder = "Test ", remainder[5:]
    elif remainder.startswith("max_"):
        prefix, remainder = "Peak ", remainder[4:]
    words = [_ACRONYMS.get(word, word) for word in remainder.split("_") if word]
    label = " ".join(words) or raw
    if not prefix and words and words[0] not in _ACRONYMS.values():
        label = label[0].upper() + label[1:]
    return prefix + label


def objective_display_name(objective: Mapping[str, Any] | Any) -> str:
    """Describe a scalar or composite Study objective without leaking internal keys."""
    if not isinstance(objective, Mapping):
        return metric_display_name(objective)
    components = objective.get("components")
    metric = objective.get("metric", "__lambdaforge_utility__")
    if isinstance(components, Sequence) and not isinstance(components, str | bytes):
        return "Composite selection score"
    return metric_display_name(metric)


def entity_key(
    item: Mapping[str, Any],
    *,
    kind: str = "work",
    fallback_index: int | None = None,
) -> str:
    """Return a collision-safe UI identity without treating a display name as an ID."""
    if kind == "cluster":
        return f"cluster:{item.get('cluster', '')}"
    for field in ("work_id", "execution_id", "primary_job_id"):
        value = item.get(field)
        if value not in (None, ""):
            return f"{kind}:{field}:{value}"
    legacy = "\x1f".join(
        str(item.get(field, ""))
        for field in ("name", "cluster", "scientific_revision", "created_at_utc")
    )
    suffix = f":{fallback_index}" if fallback_index is not None else ""
    return f"{kind}:legacy:{legacy}{suffix}"


__all__ = [
    "epoch_rows",
    "format_duration",
    "format_value",
    "confirmation_text",
    "entity_key",
    "metric_display_name",
    "objective_display_name",
    "structured_text",
    "visible_metrics",
]
