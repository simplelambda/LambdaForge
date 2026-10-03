"""Versioned, declarative metric semantics; never an executable metric registry."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import itertools
import json
import math
import string
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.metrics.MetricRegistry import MetricRegistry

SEMANTICS_VERSION = 1
_FIELDS = {
    "label",
    "description",
    "category",
    "tags",
    "role",
    "unit",
    "range",
    "direction",
    "scale",
    "split",
    "phase",
    "priority",
    "visibility",
    "discovery",
    "aliases",
    "aggregation",
    "derived_from",
    "transformation",
    "expected_to_vary",
    "practical_scale",
    "notes",
}
_QUESTIONS = {
    "relationship",
    "parameter_screen",
    "metric_family",
    "tradeoff",
    "category_summary",
    "consistency",
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return copy.deepcopy(dict(value))


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be a list of names.")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{label} contains an invalid name.")
    return list(value)


def _metadata(value: Any, label: str) -> dict[str, Any]:
    data = _mapping(value, label)
    if set(data) - _FIELDS:
        raise ValueError(f"Unknown {label} field(s): {sorted(set(data) - _FIELDS)}.")
    for field in (
        "label",
        "description",
        "category",
        "role",
        "unit",
        "phase",
        "transformation",
        "notes",
    ):
        if field in data and not isinstance(data[field], str):
            raise ValueError(f"{label}.{field} must be text.")
    if "category" in data and any(not part.strip() for part in data["category"].split("/")):
        raise ValueError(f"{label}.category must contain nonempty hierarchy components.")
    for field in ("tags", "aliases", "derived_from"):
        if field in data:
            data[field] = _names(data[field], f"{label}.{field}")
    choices = {
        "direction": {"min", "max", "unknown"},
        "scale": {"linear", "log"},
        "split": {"train", "validation", "test", "unspecified"},
        "visibility": {"primary", "normal", "advanced", "hidden"},
        "aggregation": {"selected_epoch", "latest", "best", "terminal", "mean", "unspecified"},
    }
    for field, allowed in choices.items():
        if field in data and data[field] not in allowed:
            raise ValueError(f"{label}.{field} must be one of {sorted(allowed)}.")
    for field in ("discovery", "expected_to_vary"):
        if field in data and not isinstance(data[field], bool):
            raise ValueError(f"{label}.{field} must be boolean.")
    for field in ("priority", "practical_scale"):
        if field in data and (
            not _finite(data[field])
            or data[field] < 0
            or (field == "practical_scale" and data[field] == 0)
        ):
            raise ValueError(
                f"{label}.{field} must be a finite "
                f"{'positive' if field == 'practical_scale' else 'nonnegative'} number."
            )
    if "range" in data:
        bounds = data["range"]
        if (
            not isinstance(bounds, list | tuple)
            or len(bounds) != 2
            or not all(map(_finite, bounds))
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"{label}.range must contain two increasing finite bounds.")
    return data


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def resolve_semantics(
    declared: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    objective: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve class defaults then YAML overrides, validating without constructing a Work."""
    raw: dict[str, Any] = {}
    for source in (declared, overrides):
        if source is None:
            continue
        source = _mapping(source, "analysis")
        unknown = set(source) - {"metrics", "defaults", "families", "questions", "discovery"}
        if unknown:
            raise ValueError(f"Unknown analysis fields: {sorted(unknown)}.")
        for name, value in source.items():
            if name in {"metrics", "families", "discovery"}:
                previous = _mapping(raw.get(name, {}), f"analysis.{name}")
                for key, entry in _mapping(value, f"analysis.{name}").items():
                    if (
                        name == "metrics"
                        and isinstance(previous.get(key), Mapping)
                        and isinstance(entry, Mapping)
                    ):
                        previous[key] = {**previous[key], **entry}
                    else:
                        previous[key] = entry
                raw[name] = previous
            else:
                raw[name] = value
    metrics = {
        str(name): _metadata(value, f"analysis.metrics.{name}")
        for name, value in _mapping(raw.get("metrics", {}), "analysis.metrics").items()
    }
    defaults = raw.get("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError("analysis.defaults must be a list.")
    for item in defaults:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"pattern", "metadata"}
            or not isinstance(item["pattern"], str)
        ):
            raise ValueError("Every analysis default requires pattern and metadata.")
        metadata = _metadata(item["metadata"], "analysis.defaults.metadata")
        if metadata.get("aliases"):
            raise ValueError("Metric aliases belong to named metrics, not pattern defaults.")
    families = _mapping(raw.get("families", {}), "analysis.families")
    owners: dict[str, str] = {}
    for name, value in families.items():
        family = _mapping(value, f"analysis.families.{name}")
        if set(family) - {"label", "description", "dimensions", "members", "template"}:
            raise ValueError(f"Unknown metric family fields in {name}.")
        dimensions = _mapping(family.get("dimensions", {}), f"family {name} dimensions")
        for dimension, rule in dimensions.items():
            rule = _mapping(rule, f"dimension {dimension}")
            if set(rule) - {"kind", "values"} or rule.get("kind", "categorical") not in {
                "numeric",
                "ordered",
                "categorical",
            }:
                raise ValueError(f"Invalid family dimension {dimension}.")
            if not isinstance(rule.get("values"), list) or not rule["values"]:
                raise ValueError(f"Family dimension {dimension} requires nonempty values.")
            if len({json.dumps(v, sort_keys=True) for v in rule["values"]}) != len(rule["values"]):
                raise ValueError(f"Duplicate dimension values in {dimension}.")
            if rule.get("kind") == "numeric" and not all(map(_finite, rule["values"])):
                raise ValueError(f"Numeric dimension {dimension} requires finite numbers.")
        members = _mapping(family.get("members", {}), f"family {name} members")
        if "template" in family:
            template = family["template"]
            if not isinstance(template, str) or not dimensions:
                raise ValueError(f"Family {name} needs a text template and dimensions.")
            if len(template) > 256 or any(
                field not in dimensions or spec or conversion
                for _, field, spec, conversion in string.Formatter().parse(template)
                if field is not None
            ):
                raise ValueError("Family templates accept only bare dimension placeholders.")
            if math.prod(len(rule["values"]) for rule in dimensions.values()) > 4096:
                raise ValueError("Family templates may resolve at most 4096 members.")
            for values in itertools.product(*(rule["values"] for rule in dimensions.values())):
                coordinate = dict(zip(dimensions, values, strict=True))
                try:
                    metric = template.format_map(coordinate)
                except (ValueError, KeyError, AttributeError, IndexError) as error:
                    raise ValueError(f"Invalid template for family {name}.") from error
                if metric in members:
                    raise ValueError(f"Duplicate template member {metric}.")
                members[metric] = coordinate
        family["members"] = members
        coordinates: set[str] = set()
        for metric, coordinate in members.items():
            coordinate = _mapping(coordinate, f"family {name} member {metric}")
            if set(coordinate) != set(dimensions) or any(
                coordinate[d] not in dimensions[d]["values"] for d in dimensions
            ):
                raise ValueError(f"Invalid dimensional membership for {metric} in {name}.")
            identity = json.dumps(coordinate, sort_keys=True)
            if identity in coordinates or metric in owners:
                raise ValueError(f"Ambiguous family membership for {metric}.")
            coordinates.add(identity)
            owners[metric] = name
            metrics.setdefault(metric, {})
        families[name] = family
    aliases: dict[str, str] = {}
    for name, metadata in metrics.items():
        if not name.strip():
            raise ValueError("Metric names cannot be empty.")
        for alias in metadata.get("aliases", []):
            if alias in metrics or alias in aliases:
                raise ValueError(f"Duplicate or canonical metric alias {alias!r}.")
            aliases[alias] = name

    def visit(name: str, path: set[str]) -> None:
        if name in path:
            raise ValueError(f"Cyclic metric lineage at {name!r}.")
        for parent in metrics[name].get("derived_from", []):
            if parent not in metrics:
                raise ValueError(f"Unknown lineage metric {parent!r}.")
            visit(parent, path | {name})

    for name in metrics:
        visit(name, set())
    questions = raw.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("analysis.questions must be a list.")
    question_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict) or question.get("kind") not in _QUESTIONS:
            raise ValueError("Invalid analysis question kind.")
        if set(question) - {
            "id",
            "kind",
            "x",
            "y",
            "metrics",
            "parameters",
            "family",
            "category",
            "expected",
            "optional",
            "label",
            "priority",
        }:
            raise ValueError("Unknown analysis question fields.")
        question_identity = question.get("id")
        if (
            not isinstance(question_identity, str)
            or not question_identity
            or question_identity in question_ids
        ):
            raise ValueError("Analysis questions need unique nonempty ids.")
        identity = question_identity
        question_ids.add(identity)
        if "label" in question and not isinstance(question["label"], str):
            raise ValueError(f"Question {identity}.label must be text.")
        priority = question.get("priority", 0)
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int | float)
            or not math.isfinite(priority)
            or priority < 0
        ):
            raise ValueError(f"Question {identity}.priority must be finite and non-negative.")
        if question.get("kind") in {"relationship", "consistency"} and not all(
            isinstance(question.get(k), str) for k in ("x", "y")
        ):
            raise ValueError(f"Question {identity} requires x and y metrics.")
        if question["kind"] == "metric_family" and "family" not in question:
            raise ValueError(f"Question {identity} requires a family.")
        if question["kind"] == "category_summary" and not isinstance(question.get("category"), str):
            raise ValueError(f"Question {identity} requires a category.")
        if question["kind"] == "tradeoff" and len(question.get("metrics", [])) < 2:
            raise ValueError(f"Question {identity} requires at least two metrics.")
        if "expected" in question and question["expected"] not in {"positive", "negative", "equal"}:
            raise ValueError(f"Invalid expected relationship in {identity}.")
        if "optional" in question and not isinstance(question["optional"], bool):
            raise ValueError(f"Question {identity}.optional must be boolean.")
        if "family" in question and question["family"] not in families:
            raise ValueError(f"Unknown question family {question['family']}.")
        for field in ("metrics", "parameters"):
            if field in question:
                _names(question[field], f"question {identity}.{field}")
        references = [question[k] for k in ("x", "y") if k in question] + question.get(
            "metrics", []
        )
        for reference in references:
            if (
                reference != "selection_objective"
                and reference not in metrics
                and reference not in aliases
                and MetricRegistry.resolve(reference) is None
                and not question.get("optional", False)
            ):
                raise ValueError(
                    f"Unknown analysis question metric {reference!r}; "
                    "declare it or mark the question optional."
                )
    discovery = _mapping(raw.get("discovery", {}), "analysis.discovery")
    if set(discovery) - {"enabled", "max_metrics", "max_pairs", "resamples"}:
        raise ValueError("Unknown analysis.discovery rules.")
    if "enabled" in discovery and not isinstance(discovery["enabled"], bool):
        raise ValueError("discovery.enabled must be boolean.")
    for name, bounds in {
        "max_metrics": (2, 256),
        "max_pairs": (1, 512),
        "resamples": (32, 2048),
    }.items():
        if name in discovery and (
            type(discovery[name]) is not int or not bounds[0] <= discovery[name] <= bounds[1]
        ):
            raise ValueError(f"discovery.{name} must be an integer in {bounds}.")
    result = {
        "semantics_version": SEMANTICS_VERSION,
        "metrics": metrics,
        "defaults": defaults,
        "families": families,
        "questions": questions,
        "discovery": discovery,
        "registry_defaults": MetricRegistry.catalog_defaults(),
    }
    if objective:
        names = [objective["metric"]] if objective.get("metric") else []
        names += list(objective.get("metrics", {}))
        names += list(objective.get("constraints", {}))
        catalogue = MetricCatalog.resolve(result, set(metrics) | set(names))

        def test_derived(name: str, path: set[str]) -> bool:
            name = aliases.get(name, name)
            return name not in path and (
                catalogue["metrics"].get(name, {}).get("split") == "test"
                or any(
                    test_derived(parent, path | {name})
                    for parent in catalogue["metrics"].get(name, {}).get("derived_from", [])
                )
            )

        if any(test_derived(name, set()) for name in names):
            raise ValueError(
                "Test-split metrics (including derived metrics) cannot govern "
                "objective or constraints."
            )
    result["identity"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return result


class MetricCatalog:
    """Resolve frozen declarations over recorded names; no observed-value inference of meaning."""

    @staticmethod
    def resolve(semantics: Mapping[str, Any], names: set[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        declared = semantics.get("metrics", {})
        for name in sorted(names | set(declared)):
            known = semantics.get("registry_defaults", {}).get(name, {})
            split = next(
                (
                    split
                    for prefix, split in (
                        ("test_", "test"),
                        ("val_", "validation"),
                        ("validation_", "validation"),
                        ("train_", "train"),
                    )
                    if name.startswith(prefix)
                ),
                "unspecified",
            )
            metadata = {
                "name": name,
                "label": name.replace("_", " "),
                "description": "",
                "category": split if split != "unspecified" else "uncategorized",
                "split": split,
                "unit": "unknown",
                "direction": known.get("mode", "unknown"),
                "aggregation": "unspecified",
                "visibility": "normal",
                "priority": 0,
                "discovery": True,
                "aliases": [],
                "tags": [],
                "derived_from": [],
                "source": "inferred",
            }
            if name.startswith("resource."):
                metadata.update(
                    category="resources",
                    role="resource",
                    direction="min",
                    unit="seconds" if name.endswith("seconds") else "bytes",
                    aggregation="terminal",
                    visibility="advanced",
                )
            if "range" in known:
                metadata["range"] = known["range"]
            for rule in semantics.get("defaults", []):
                if fnmatch.fnmatchcase(name, rule["pattern"]):
                    metadata.update(rule["metadata"])
                    metadata["source"] = "pattern"
            if name in declared:
                metadata.update(declared[name])
                metadata["source"] = "declared"
            if name == "selection_objective":
                metadata.update(
                    label="Selection objective",
                    role="objective",
                    visibility="primary",
                    priority=100,
                    aggregation="selected_epoch",
                )
            if name == "__lambdaforge_utility__":
                metadata.update(
                    label="Composite selection score",
                    role="objective",
                    visibility="hidden",
                    discovery=False,
                )
            result[name] = metadata
        for family_name, family in semantics.get("families", {}).items():
            for name, coordinate in family.get("members", {}).items():
                result[name].update(family=family_name, dimensions=coordinate)

        def contains_test(name: str, seen: set[str]) -> bool:
            return name not in seen and (
                result.get(name, {}).get("split") == "test"
                or any(
                    contains_test(parent, seen | {name})
                    for parent in result.get(name, {}).get("derived_from", [])
                )
            )

        for name, metadata in result.items():
            metadata["contains_test_evidence"] = contains_test(name, set())
        return {
            "catalog_version": 1,
            "metrics": result,
            "families": copy.deepcopy(semantics.get("families", {})),
        }
