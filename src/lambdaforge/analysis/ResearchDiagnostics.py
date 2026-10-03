"""Cheap inspection signals, explicitly separate from inferential discovery."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def rank_findings(
    findings: list[dict[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Rank presentation evidence and select a diverse inbox; retain every raw finding."""
    representatives = {
        name: (group["representative"], len(group["members"]))
        for group in groups
        for name in group["members"]
    }
    for finding in findings:
        penalties = [
            1 / representatives[name][1]
            for name in finding.get("variables", [])
            if name in representatives
            and representatives[name][0] != name
            and name not in finding.get("parameter_names", [])
        ]
        penalty = min(penalties, default=1.0)
        finding["ranking_components"] = {
            **finding.get("reliability", {}).get("components", {}),
            "base_priority": finding["priority"],
            "redundancy_penalty": penalty,
        }
        finding["ranking_score"] = finding["priority"] * penalty
    findings.sort(key=lambda f: (-f["ranking_score"], f["id"]))
    inbox: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for finding in findings:
        variables = tuple(
            sorted(
                "parameter:" + name
                if name in finding.get("parameter_names", [])
                else representatives.get(name, (name, 1))[0]
                for name in finding.get("variables", [])
            )
        )
        identity = (finding["kind"], variables) if variables else (finding["id"],)
        if identity not in seen:
            inbox.append(finding)
            seen.add(identity)
        if len(inbox) == 8:
            break
    return inbox


def inspection_signals(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    relationships: Sequence[Mapping[str, Any]],
    correlation: Callable[[Any, Any], float],
    ranks: Callable[[Any], Any],
) -> list[dict[str, Any]]:
    """Flag robust outliers and simple context sign reversals, never assign confidence."""
    result: list[dict[str, Any]] = []
    names = sorted(
        catalog["metrics"], key=lambda name: (-catalog["metrics"][name]["priority"], name)
    )[:64]
    for name in names:
        observations = [
            (row["trial"], row["values"][name]) for row in rows if name in row["values"]
        ]
        if len(observations) < 8:
            continue
        values = np.asarray([value for _, value in observations], dtype=float)
        low, high = np.quantile(values, [0.25, 0.75])
        width = high - low
        if not width:
            continue
        outliers = [
            trial
            for trial, value in observations
            if value < low - 1.5 * width or value > high + 1.5 * width
        ]
        if outliers:
            result.append(
                {
                    "id": f"outliers:{name}",
                    "kind": "outlier",
                    "origin": "exploratory",
                    "title": f"Unusual observed values: {catalog['metrics'][name]['label']}",
                    "summary": f"{len(outliers)} Trials lie beyond descriptive 1.5-IQR whiskers.",
                    "variables": [name],
                    "support": len(values),
                    "priority": 0.2,
                    "evidence": {
                        "trial_ids": outliers[:20],
                        "iqr_bounds": [float(low), float(high)],
                        "total_outliers": len(outliers),
                    },
                    "reliability": {
                        "status": "inspection_only",
                        "method": "Tukey 1.5-IQR whiskers",
                    },
                    "caveats": [
                        "An unusual value is not a measurement error "
                        "or statistically confirmed anomaly."
                    ],
                    "recommended_view": {
                        "x": "trial",
                        "y": "metric:"
                        + ("__selection__" if name == "selection_objective" else name),
                    },
                }
            )
    parameters = sorted({name for row in rows for name in row["parameters"]})
    for relation in sorted(relationships, key=lambda r: (-abs(r["effect"]), r["x"], r["y"]))[:8]:
        left, right = relation["x"], relation["y"]
        if (
            relation.get("x_kind") != "parameter"
            or left not in parameters
            or relation["method"] != "Spearman"
        ):
            continue
        for context in parameters:
            if context == left:
                continue
            groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                x, y, c = (
                    row["parameters"].get(left),
                    row["values"].get(right),
                    row["parameters"].get(context),
                )
                if (
                    isinstance(x, int | float)
                    and not isinstance(x, bool)
                    and y is not None
                    and c is not None
                ):
                    groups[str(c)].append((float(x), float(y)))
            # Do not invent bins or run an unbounded context search over continuous values.
            if not 2 <= len(groups) <= 6:
                continue
            evidence: list[dict[str, Any]] = []
            for group, paired in sorted(groups.items()):
                if len(paired) >= 6:
                    xvalues, yvalues = np.asarray(paired).T
                    if len(np.unique(xvalues)) > 1 and len(np.unique(yvalues)) > 1:
                        evidence.append(
                            {
                                "group": group,
                                "candidates": len(paired),
                                "spearman": correlation(ranks(xvalues), ranks(yvalues)),
                            }
                        )
            scores = [row["spearman"] for row in evidence]
            if scores and min(scores) < 0 < max(scores):
                result.append(
                    {
                        "id": f"context:{left}:{right}:{context}",
                        "kind": "context_reversal",
                        "origin": "exploratory",
                        "title": f"Context-sensitive response: {left} → {right}",
                        "summary": f"Observed association changes sign across {context} levels.",
                        "variables": [left, right, context],
                        "support": sum(row["candidates"] for row in evidence),
                        "priority": 0.2,
                        "evidence": {"context": context, "groups": evidence},
                        "reliability": {
                            "status": "inspection_only",
                            "method": "descriptive stratified Spearman",
                        },
                        "caveats": [
                            "No adjusted context-level inference; "
                            "other parameters remain uncontrolled.",
                            "A sign change may reflect noise, selection bias "
                            "or a genuine interaction.",
                        ],
                        "recommended_view": {
                            "x": "param:" + left,
                            "y": "metric:"
                            + ("__selection__" if right == "selection_objective" else right),
                        },
                    }
                )
    return result
