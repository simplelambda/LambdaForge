"""Deterministic, bounded retrospective discovery over existing candidate evidence.

This service has no scheduler dependency. Association diagnostics are not calibrated
scientific confidence, causal evidence, or new objective values.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from lambdaforge.analysis.MetricCatalog import MetricCatalog, resolve_semantics
from lambdaforge.analysis.ResearchDiagnostics import inspection_signals, rank_findings
from lambdaforge.hpo.ParameterSpace import ParameterSpace

RESEARCH_VERSION = 1


def finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def candidate_values(candidate: Mapping[str, Any]) -> dict[str, float]:
    """No partial score, survivor-only means, or incomparable rung mixtures."""
    if not finite(candidate.get("mean")) or candidate.get("censored_observations", 0):
        return {}
    values: dict[str, float] = {"selection_objective": float(candidate["mean"])}
    for field in ("objective_components", "diagnostic_metrics"):
        for name, item in candidate.get(field, {}).items():
            value = item.get("mean") if isinstance(item, Mapping) else item
            if finite(value):
                values[str(name)] = float(str(value))
    runs = [
        run
        for run in candidate.get("runs", [])
        if run.get("phase") != "confirmation"
        and finite(run.get("final_objective"))
        and not run.get("censored")
    ]
    rungs = {
        (run.get("fidelity", {}).get("target"), run.get("fidelity", {}).get("maximum"))
        for run in runs
    }
    if len(rungs) <= 1:
        extra: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            for name, value in run.get("metrics", {}).items():
                if finite(value):
                    extra[name].append(float(value))
        for name, observations in extra.items():
            values.setdefault(name, float(np.mean(observations)))
    cost = candidate.get("resource_cost", {}).get("intrinsic_per_comparable_run", {})
    values.update(
        {
            f"resource.{name}": float(value)
            for name, value in cost.items()
            if finite(value) and name != "matched_run_count"
        }
    )
    return values


def _ranks(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2)[inverse]


def _correlation(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    left, right = left - np.mean(left), right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _adjust(records: list[dict[str, Any]], total: int) -> None:
    """BY adjustment includes unconfirmed screened hypotheses as p=1, not only winners."""
    harmonic = sum(1 / i for i in range(1, total + 1))
    bound = 1.0
    ranked = sorted(records, key=lambda record: record["p_value"])
    for index in reversed(range(len(ranked))):
        bound = min(bound, ranked[index]["p_value"] * total * harmonic / (index + 1))
        ranked[index]["adjusted_p"] = bound


class ResearchAnalysis:
    """One backend authority for metric health, exploratory findings and family evidence."""

    @classmethod
    def compute(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        *,
        fingerprint: str,
        semantics: Mapping[str, Any] | None = None,
        status: str = "final",
        parameter_names: Sequence[str] = (),
        parameter_space: Mapping[str, Any] | None = None,
        existing_findings: Sequence[Mapping[str, Any]] = (),
        objective: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        semantics = semantics or resolve_semantics()
        rows: list[dict[str, Any]] = [
            {
                "trial": candidate.get("trial"),
                "parameters": dict(candidate.get("parameters", {})),
                "values": candidate_values(candidate),
            }
            for candidate in candidates
        ]
        observed = {name for row in rows for name in row["values"]}
        observed |= {
            str(name)
            for candidate in candidates
            for field in ("diagnostic_metrics", "objective_components")
            for name in candidate.get(field, {})
        }
        # Keep missing/pruned-only metrics discoverable, but never assign them a final score.
        observed |= {
            str(name)
            for candidate in candidates
            for run in candidate.get("runs", [])
            for name in run.get("metrics", {})
        }
        catalog = MetricCatalog.resolve(semantics, observed | {"selection_objective"})
        if objective:
            catalog["metrics"]["selection_objective"]["direction"] = objective["mode"]
            if "range" in objective:
                catalog["metrics"]["selection_objective"]["range"] = objective["range"]
        geometry = ParameterSpace.from_schema(
            parameter_space, tuple(row["parameters"] for row in rows)
        )
        descriptors = {descriptor.name: descriptor for descriptor in geometry.descriptors}
        analysis_rows = [
            {
                **row,
                "parameters": {
                    name: descriptors[name].normalize(value)
                    if name in descriptors and descriptors[name].kind in {"continuous", "integer"}
                    else value
                    for name, value in row["parameters"].items()
                },
            }
            for row in rows
        ]
        aliases = {
            alias: name
            for name, metadata in catalog["metrics"].items()
            for alias in metadata["aliases"]
        }
        for row in rows:
            for alias, name in aliases.items():
                if alias in row["values"] and name not in row["values"]:
                    row["values"][name] = row["values"].pop(alias)
        profiles = cls._profiles(rows, candidates, catalog)
        eligible = sorted(
            (
                name
                for name, profile in profiles.items()
                if profile["informative"]
                and catalog["metrics"][name]["discovery"]
                and (status == "final" or not catalog["metrics"][name]["contains_test_evidence"])
            ),
            key=lambda name: (-catalog["metrics"][name]["priority"], name),
        )
        rules = {
            "enabled": True,
            "max_metrics": 64,
            "max_pairs": 128,
            "resamples": 256,
            **semantics.get("discovery", {}),
        }
        selected = eligible[: rules["max_metrics"]]
        redundancy = cls._redundancy(rows, selected, catalog)
        relationships: list[dict[str, Any]] = []
        question_status: list[dict[str, Any]] = []
        if rules["enabled"]:
            hypotheses: dict[tuple[str, str, str], dict[str, Any]] = {}
            for question in semantics.get("questions", []):
                pairs: list[tuple[str, str]] = []
                if question["kind"] in {"relationship", "consistency"}:
                    pairs = [(question["x"], question["y"])]
                elif question["kind"] == "parameter_screen":
                    pairs = list(
                        itertools.product(
                            question.get("parameters", parameter_names),
                            question.get("metrics", selected),
                        )
                    )
                elif question["kind"] == "tradeoff":
                    pairs = list(itertools.combinations(question.get("metrics", []), 2))
                available = 0
                for left, right in pairs:
                    kind = "parameter" if question["kind"] == "parameter_screen" else "metric"
                    left = aliases.get(left, left) if kind == "metric" else left
                    right = aliases.get(right, right)
                    known_left = (
                        left in parameter_names if kind == "parameter" else left in profiles
                    )
                    if known_left and right in profiles:
                        hypotheses[(left, right, kind)] = {
                            "origin": "configured",
                            "question": question["id"],
                            "expected": question.get("expected"),
                            "x_kind": kind,
                        }
                        available += 1
                question_status.append(
                    {
                        "id": question["id"],
                        "kind": question["kind"],
                        "status": "available"
                        if available or question["kind"] in {"metric_family", "category_summary"}
                        else "unavailable",
                        "reason": None
                        if available
                        else "No comparable referenced observations or a summary-only question.",
                    }
                )
            # Interleave parameter and metric questions; neither type monopolizes the
            # budget, and all authored parameters get a turn before a second metric.
            automatic = itertools.zip_longest(
                (
                    (parameter, metric, "parameter")
                    for metric in selected
                    for parameter in parameter_names
                ),
                ((left, right, "metric") for left, right in itertools.combinations(selected, 2)),
            )
            for pair in itertools.chain.from_iterable(automatic):
                if len(hypotheses) >= rules["max_pairs"]:
                    break
                if pair is None:
                    continue
                left, right, kind = pair
                if pair not in hypotheses:
                    hypotheses[pair] = {"origin": "exploratory", "x_kind": kind}
            for (left, right, kind), purpose in list(hypotheses.items())[: rules["max_pairs"]]:
                if status != "final" and any(
                    catalog["metrics"].get(name, {}).get("contains_test_evidence")
                    for name in ((left, right) if kind == "metric" else (right,))
                ):
                    continue
                record = cls._relationship(analysis_rows, left, right, fingerprint, 0, purpose)
                if record is not None:
                    relationships.append(record)
            shortlist = sorted(
                relationships,
                key=lambda record: (
                    record.get("origin") != "configured",
                    -abs(record["effect"]),
                    record["x"],
                    record["y"],
                ),
            )[:32]
            for record in shortlist:
                measured = cls._relationship(
                    analysis_rows,
                    record["x"],
                    record["y"],
                    fingerprint,
                    rules["resamples"],
                    {
                        name: record[name]
                        for name in ("origin", "question", "expected", "x_kind")
                        if name in record
                    },
                )
                if measured is not None:
                    record.update(measured)
            _adjust(relationships, max(1, min(len(hypotheses), rules["max_pairs"])))
        findings = cls._findings(profiles, relationships, catalog)
        if rules["enabled"]:
            diagnostic_catalog = {"metrics": {name: catalog["metrics"][name] for name in selected}}
            findings.extend(
                inspection_signals(rows, diagnostic_catalog, relationships, _correlation, _ranks)
            )
        families = cls._families(rows, catalog)
        for summary in question_status:
            question = next(q for q in semantics["questions"] if q["id"] == summary["id"])
            matches = [r for r in relationships if r.get("question") == summary["id"]]
            if question["kind"] == "metric_family":
                points = families.get(question["family"], {}).get("points", [])
                support = sum(p["support"] > 0 for p in points)
                summary.update(
                    family=question["family"], observed_members=support, total_members=len(points)
                )
            elif question["kind"] == "category_summary":
                names = [
                    name
                    for name, m in catalog["metrics"].items()
                    if m["category"] == question["category"]
                    or m["category"].startswith(question["category"] + "/")
                ]
                support = sum(profiles[name]["finite_candidates"] > 0 for name in names)
                summary.update(
                    category=question["category"],
                    observed_metrics=support,
                    total_metrics=len(names),
                )
            else:
                support = len(matches)
                summary.update(comparable_relationships=support)
            summary["status"] = "available" if support else "insufficient_support"
            summary["reason"] = (
                "Descriptive summary; no new hypothesis inference."
                if question["kind"] in {"metric_family", "category_summary"} and support
                else "Comparable recorded evidence is available; reliability is separate."
                if support
                else "Missing/comparable evidence or a bounded discovery budget prevents analysis."
            )
        for item in existing_findings:
            findings.append(
                {
                    "id": "study:"
                    + str(item["kind"])
                    + ":"
                    + hashlib.sha256(str(item["title"]).encode()).hexdigest()[:12],
                    "kind": item["kind"],
                    "origin": "persisted_study_analysis",
                    "title": item["title"],
                    "summary": item["statement"],
                    "support": item.get("reliability_components", {}).get("support"),
                    "reliability": {
                        "status": item.get("reliability"),
                        "method": "existing Study Analysis",
                        "components": item.get("reliability_components", {}),
                    },
                    "evidence": {"observations": item.get("evidence", [])},
                    "caveats": [item.get("implication", "")],
                    "priority": 0.5 if item.get("severity") == "warning" else 0.3,
                    "recommended_view": None,
                }
            )
        for item in findings:
            item.setdefault(
                "category",
                "study" if item["origin"] == "persisted_study_analysis" else "relationships",
            )
            item.setdefault("variables", [])
            item.setdefault("status", item["reliability"]["status"])
            item.setdefault("effect", None)
            item.setdefault("practical_significance", "Not calibrated; descriptive evidence only.")
        inbox = rank_findings(findings, redundancy)
        categories: dict[str, Any] = {}
        for name, metadata in catalog["metrics"].items():
            entry = categories.setdefault(
                metadata["category"], {"metrics": [], "informative": 0, "constant": 0}
            )
            entry["metrics"].append(name)
            entry["informative"] += int(profiles[name]["informative"])
            entry["constant"] += int(profiles[name]["constant"])
        return {
            "research_version": RESEARCH_VERSION,
            "semantics": dict(semantics),
            "metric_catalog": catalog,
            "profile_identity": semantics.get("identity"),
            "metric_profiles": profiles,
            "rows": rows,
            "redundancy_groups": redundancy,
            "relationships": relationships,
            "families": families,
            "categories": categories,
            "questions": question_status,
            "findings": findings,
            "inbox_findings": inbox,
            "methodology": {
                "rules_version": "bounded-association-by-v1",
                "rules": rules,
                "screened_metric_count": len(selected),
                "excluded_metric_count": len(eligible) - len(selected),
                "expensive_relationship_count": sum(r["resamples"] > 0 for r in relationships),
                "expensive_relationship_limit": 32,
                "inference_unit": "candidate",
                "resampling": "deterministic paired-value permutation and candidate bootstrap",
                "multiplicity": "Benjamini-Yekutieli; unconfirmed hypotheses count as p=1",
                "limitations": [
                    "Adaptive sampling is not randomized causal evidence; adjusted p-values "
                    "are exploratory diagnostics, not confirmatory guarantees.",
                    "Candidate bootstrap stability is not shared-seed scientific confidence.",
                    "Terminal diagnostic metrics may be latest rather than selected-checkpoint "
                    "values; unspecified aggregation stays explicit.",
                    "Missing, censored and mixed-fidelity candidates never become exact "
                    "completed evidence.",
                    "Pair budget is deterministic; not every possible relationship is examined.",
                ],
            },
        }

    @staticmethod
    def _profiles(
        rows: list[dict[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        catalog: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = {}
        all_seeds = {
            run["seed"]
            for candidate in candidates
            for run in candidate.get("runs", [])
            if "seed" in run and run.get("phase") != "confirmation"
        }
        for name, metadata in catalog["metrics"].items():
            values = np.asarray(
                [row["values"][name] for row in rows if name in row["values"]], dtype=float
            )
            support = len(values)
            distinct = len(np.unique(values))
            spread = float(np.ptp(values)) if support else None
            scale = metadata.get("practical_scale")
            if scale is None and "range" in metadata:
                scale = metadata["range"][1] - metadata["range"][0]
            if scale is None and support:
                scale = float(np.max(np.abs(values)))
            near = bool(
                support >= 2
                and distinct > 1
                and scale
                and spread is not None
                and spread / scale <= 1e-6
            )
            constant = support >= 2 and distinct == 1
            reason = (
                "missing"
                if not support
                else "insufficient_support"
                if support < 2
                else "constant"
                if constant
                else "near_constant"
                if near
                else "variable"
            )
            warnings = []
            coverage = support / len(rows) if rows else 0
            if support and coverage < 0.5:
                warnings.append("low_coverage")
            if metadata["aggregation"] == "unspecified":
                warnings.append("aggregation_unspecified")
            if metadata.get("expected_to_vary") and constant:
                warnings.append("expected_variation_not_observed")
            bounds = metadata.get("range")
            if bounds and any(value < bounds[0] or value > bounds[1] for value in values):
                warnings.append("outside_declared_range")
            observed_runs = [
                run
                for candidate in candidates
                for run in candidate.get("runs", [])
                if finite(run.get("final_objective"))
                and not run.get("censored")
                and run.get("phase") != "confirmation"
                and (
                    name == "selection_objective"
                    or finite(run.get("metrics", {}).get(name))
                    or any(finite(run.get("metrics", {}).get(a)) for a in metadata["aliases"])
                )
            ]
            result[name] = {
                "finite_candidates": support,
                "total_candidates": len(rows),
                "missing_candidates": len(rows) - support,
                "coverage": support / len(rows) if rows else 0,
                "low_coverage": bool(support and coverage < 0.5),
                "unique_values": distinct,
                "min": float(np.min(values)) if support else None,
                "max": float(np.max(values)) if support else None,
                "mean": float(np.mean(values)) if support else None,
                "median": float(np.median(values)) if support else None,
                "sd": float(np.std(values, ddof=1)) if support >= 2 else None,
                "robust_spread": float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                if support
                else None,
                "run_support": len(observed_runs),
                "seed_support": len({run["seed"] for run in observed_runs if "seed" in run}),
                "seed_coverage": len({run["seed"] for run in observed_runs if "seed" in run})
                / len(all_seeds)
                if all_seeds
                else None,
                "run_coverage": len(observed_runs) / sum(len(c.get("runs", [])) for c in candidates)
                if any(c.get("runs") for c in candidates)
                else None,
                "constant": constant,
                "near_constant": near,
                "informative": reason == "variable",
                "reason": reason,
                "warnings": warnings,
                "relative_spread_scale": scale,
            }
        return result

    @staticmethod
    def _redundancy(
        rows: list[dict[str, Any]], names: list[str], catalog: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        assigned: set[str] = set()
        for left in names:
            if left in assigned or catalog["metrics"][left].get("family"):
                continue
            members = [left]
            for right in names:
                if right == left or right in assigned or catalog["metrics"][right].get("family"):
                    continue
                paired = [
                    (row["values"][left], row["values"][right])
                    for row in rows
                    if left in row["values"] and right in row["values"]
                ]
                if len(paired) < 6:
                    continue
                x, y = np.asarray(paired, dtype=float).T
                if abs(_correlation(_ranks(x), _ranks(y))) >= 0.995:
                    members.append(right)
                    assigned.add(right)
            if len(members) > 1:
                groups.append(
                    {
                        "cluster_id": "redundancy:"
                        + hashlib.sha256("\0".join(members).encode()).hexdigest()[:12],
                        "representative": left,
                        "members": members,
                        "method": "absolute Spearman >= .995 on >=6 paired candidates; "
                        "descriptive only",
                    }
                )
            assigned.add(left)
        return groups

    @staticmethod
    def _relationship(
        rows: list[dict[str, Any]],
        left: str,
        right: str,
        fingerprint: str,
        resamples: int,
        purpose: dict[str, Any],
    ) -> dict[str, Any] | None:
        paired = [
            (
                row["parameters"].get(left)
                if purpose.get("x_kind") == "parameter"
                else row["values"].get(left),
                row["values"].get(right),
            )
            for row in rows
        ]
        paired = [(x, y) for x, y in paired if x is not None and finite(y)]
        if len(paired) < 3:
            return None
        labels = [str(x) for x, _ in paired]
        numeric = all(finite(x) for x, _ in paired)
        x = (
            np.asarray([x for x, _ in paired], dtype=float)
            if numeric
            else np.asarray([sorted(set(labels)).index(x) for x in labels], dtype=float)
        )
        y = np.asarray([y for _, y in paired], dtype=float)
        observed_equal = bool(numeric and np.array_equal(x, y))
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            return None
        if numeric:
            raw_x = x.copy()
            x = _ranks(x)
            y = _ranks(y)
            normalized_x = (raw_x - np.mean(raw_x)) / np.std(raw_x)
            design = np.column_stack([np.ones(len(x)), normalized_x, normalized_x**2])
            inverse = np.linalg.pinv(design)
            diagonal = 1 - np.sum(design * inverse.T, axis=1)
            quadratic_ready = len(x) >= 8 and np.min(diagonal) > 0.05

            def quadratic(values: np.ndarray[Any, Any]) -> float:
                baseline = float(np.sum((values - np.mean(values)) ** 2))
                if not quadratic_ready or not baseline:
                    return 0.0
                residuals = (values - design @ (inverse @ values)) / diagonal
                return max(0.0, 1 - float(np.dot(residuals, residuals)) / baseline)

            def statistic(values: np.ndarray[Any, Any]) -> float:
                return max(_correlation(x, values) ** 2, quadratic(values))

            effect = _correlation(x, y)
            gain = quadratic(y)
            method = "Spearman" if effect**2 >= gain else "quadratic rank-response LOO gain"
            if method != "Spearman":
                effect = gain
        else:
            indices = [np.flatnonzero(x == level) for level in np.unique(x)]

            def statistic(values: np.ndarray[Any, Any]) -> float:
                total = float(np.sum((values - np.mean(values)) ** 2))
                return (
                    float(
                        sum(len(i) * (np.mean(values[i]) - np.mean(values)) ** 2 for i in indices)
                        / total
                    )
                    if total
                    else 0.0
                )

            effect = statistic(y)
            method = "categorical eta-squared"
        seed = int(hashlib.sha256(f"{fingerprint}:{left}:{right}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        observed = statistic(y)
        count = sum(statistic(rng.permutation(y)) >= observed - 1e-12 for _ in range(resamples))
        stability: float | None = None
        if numeric and method == "Spearman":
            scores = []
            for _ in range(min(resamples, 128)):
                ids = rng.integers(0, len(x), len(x))
                if len(np.unique(x[ids])) > 1 and len(np.unique(y[ids])) > 1:
                    scores.append(_correlation(x[ids], y[ids]))
            stability = (
                float(np.mean([np.sign(score) == np.sign(effect) for score in scores]))
                if scores
                else None
            )
        return {
            "x": left,
            "y": right,
            "effect": effect,
            "method": method,
            "support": len(x),
            "p_value": (count + 1) / (resamples + 1),
            "stability": stability,
            "resamples": resamples,
            "shape": "monotonic"
            if method == "Spearman"
            else "nonlinear"
            if numeric
            else "categorical",
            "observed_equal": observed_equal,
            **purpose,
        }

    @staticmethod
    def _findings(
        profiles: Mapping[str, Any],
        relationships: Sequence[Mapping[str, Any]],
        catalog: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        result = []

        def ancestors(name: str, seen: set[str]) -> set[str]:
            if name in seen:
                return set()
            parents = set(catalog["metrics"].get(name, {}).get("derived_from", []))
            return parents | {
                ancestor for parent in parents for ancestor in ancestors(parent, seen | {name})
            }

        for name, profile in profiles.items():
            if (
                "expected_variation_not_observed" in profile["warnings"]
                and profile["finite_candidates"] >= 3
            ):
                result.append(
                    {
                        "id": f"invariant:{name}",
                        "kind": "invariant",
                        "origin": "configured",
                        "title": f"Expected variation absent: {catalog['metrics'][name]['label']}",
                        "summary": f"Identical values across {profile['finite_candidates']} "
                        "candidates; check instrumentation or declared semantics.",
                        "variables": [name],
                        "support": profile["finite_candidates"],
                        "priority": 1.0,
                        "reliability": {
                            "status": "descriptive",
                            "method": "exact observed equality, not an inference",
                        },
                        "caveats": ["Constant evidence does not establish a universal invariant."],
                        "recommended_view": {"x": "trial", "y": f"metric:{name}"},
                    }
                )
        for record in relationships:
            if not record["resamples"]:
                continue
            left, right = record["x"], record["y"]
            right_parents, left_parents = ancestors(right, set()), ancestors(left, set())
            structural = (
                left in right_parents or right in left_parents or bool(right_parents & left_parents)
            ) and record.get("x_kind") != "parameter"
            support_weight = min(1.0, record["support"] / 20)
            coverage_weight = record["support"] / max(1, profiles[right]["total_candidates"])
            strength = abs(record["effect"])
            stable = record.get("stability")
            score = (
                strength
                * support_weight
                * coverage_weight
                * (stable if stable is not None else 0.5)
                * (0.1 if structural else 1)
            )
            reliable = record["support"] >= 8 and record["adjusted_p"] <= 0.05
            expectation = record.get("expected")
            contradiction = (
                (
                    expectation == "positive"
                    and record["method"] == "Spearman"
                    and record["effect"] < 0
                )
                or (
                    expectation == "negative"
                    and record["method"] == "Spearman"
                    and record["effect"] > 0
                )
                or (expectation == "equal" and not record["observed_equal"])
            )
            result.append(
                {
                    "id": f"association:{left}:{right}"
                    + (":parameter" if record.get("x_kind") == "parameter" else ""),
                    "kind": "expectation_mismatch" if contradiction else "association",
                    "origin": record["origin"],
                    "title": (
                        left
                        if record.get("x_kind") == "parameter"
                        else catalog["metrics"][left]["label"]
                    )
                    + " → "
                    + catalog["metrics"][right]["label"],
                    "summary": f"{record['method']}: {record['effect']:.3g}; "
                    f"{record['support']} paired candidates.",
                    "variables": [left, right],
                    "parameter_names": [left] if record.get("x_kind") == "parameter" else [],
                    "metric_names": [right]
                    if record.get("x_kind") == "parameter"
                    else [left, right],
                    "effect": record["effect"],
                    "support": record["support"],
                    "stability": stable,
                    "priority": score,
                    "structural_relationship": structural,
                    "expected_relationship": expectation,
                    "reliability": {
                        "status": "supported_exploration" if reliable else "preliminary",
                        "adjusted_p": record["adjusted_p"],
                        "method": "bounded permutation + BY diagnostic",
                        "components": {
                            "support_weight": support_weight,
                            "coverage_weight": coverage_weight,
                            "structural_novelty": 0.1 if structural else 1,
                            "effect_strength": strength,
                            "sign_stability": stable,
                        },
                    },
                    "evidence": dict(record),
                    "caveats": [
                        "Other parameters are uncontrolled; association is not causation.",
                        "No fresh independent confirmation of this exploratory relationship.",
                    ]
                    + (
                        ["Declared metric lineage may mechanically explain this relationship."]
                        if structural
                        else []
                    ),
                    "recommended_view": {
                        "x": ("param:" if record.get("x_kind") == "parameter" else "metric:")
                        + ("__selection__" if left == "selection_objective" else left),
                        "y": "metric:"
                        + ("__selection__" if right == "selection_objective" else right),
                    },
                }
            )
        return sorted(result, key=lambda item: (-item["priority"], item["id"]))

    @staticmethod
    def _families(rows: list[dict[str, Any]], catalog: Mapping[str, Any]) -> dict[str, Any]:
        result = {}
        for name, family in catalog["families"].items():
            points = []
            for metric, dimensions in family["members"].items():
                values = [row["values"][metric] for row in rows if metric in row["values"]]
                points.append(
                    {
                        "metric": metric,
                        "dimensions": dimensions,
                        "support": len(values),
                        "mean": float(np.mean(values)) if values else None,
                        "sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                    }
                )
            result[name] = {
                **family,
                "points": points,
                "interpretation": "Descriptive candidate means; not a paired intervention "
                "or seed uncertainty.",
            }
        return result
