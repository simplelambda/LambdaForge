"""Explicit project-owned scientific verification; exact content remains authoritative."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.data.DatasetArtifact import DatasetArtifact
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.index import DatasetMember
from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.work.snapshot import validate_path


@dataclass(frozen=True)
class DatasetComparisonContext:
    """One matching member, validated roots and explicit variable-specific policy."""

    left_root: Path
    right_root: Path
    left: DatasetMember
    right: DatasetMember
    policy: Mapping[str, Any]


class DatasetComparison:
    """Compare sealed bytes before invoking an explicitly trusted project verifier.

    A verifier returns equivalent: bool, checked_assets: list[str], and optional
    JSON details. Every changed asset must be covered. LambdaForge never rounds
    arrays, guesses operational fields or writes a registry during comparison.
    """

    @classmethod
    def compare(
        cls,
        left: str | Path,
        right: str | Path,
        *,
        verifier: Callable[[DatasetComparisonContext], Mapping[str, Any]] | None = None,
        verifier_id: str | None = None,
        policy: Mapping[str, Any] | None = None,
        scientific_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        roots = tuple(Path(item).expanduser().absolute() for item in (left, right))
        for root in roots:
            validate_path(root / "dataset-artifact.json")
        artifacts = tuple(
            DatasetArtifact.read_json(root / "dataset-artifact.json") for root in roots
        )
        checked = tuple(
            DatasetOperations.verify(root, artifact.dataset_id)
            for root, artifact in zip(roots, artifacts, strict=True)
        )
        comparison_ids: list[str | None] = []
        for artifact in artifacts:
            assertion = (scientific_contracts or {}).get(artifact.content_id)
            identifier = artifact.scientific_id
            if assertion is not None:
                asserted_id = DatasetArtifact.scientific_contract_id(assertion)
                if identifier is not None and identifier != asserted_id:
                    raise ValueError(
                        "Explicit contract cannot override persisted scientific identity."
                    )
                identifier = asserted_id
            comparison_ids.append(identifier)
        report: dict[str, Any] = {
            "comparison_version": 1,
            "left": {
                "root": str(roots[0]),
                "content_id": artifacts[0].content_id,
                "scientific_id": artifacts[0].scientific_id,
                "comparison_scientific_id": comparison_ids[0],
            },
            "right": {
                "root": str(roots[1]),
                "content_id": artifacts[1].content_id,
                "scientific_id": artifacts[1].scientific_id,
                "comparison_scientific_id": comparison_ids[1],
            },
            "byte_equal": artifacts[0].content_id == artifacts[1].content_id,
            "scientifically_equivalent": None,
            "status": "unresolved",
            "verifier": verifier_id,
            "policy": dict(policy or {}),
            "members": [],
            "registration_changed": False,
            "project_contract_assertions": dict(scientific_contracts or {}),
        }
        if any(not result["valid"] for result in checked):
            return {
                **report,
                "status": "invalid",
                "byte_equal": False,
                "errors": [result["errors"] for result in checked],
            }
        if comparison_ids[0] != comparison_ids[1]:
            if comparison_ids[0] is None or comparison_ids[1] is None:
                return {**report, "reason": "Both references need an explicit scientific_identity."}
            return {
                **report,
                "status": "different",
                "scientifically_equivalent": False,
                "reason": "Declared scientific contracts differ, even if asset bytes match.",
            }
        # Global assets have no per-member verifier: never silently tolerate differences.
        if (
            artifacts[0].target_schema != artifacts[1].target_schema
            or artifacts[0].global_assets != artifacts[1].global_assets
        ):
            return {
                **report,
                "status": "different",
                "scientifically_equivalent": False,
                "reason": "Target schema or exact global assets differ.",
            }
        if report["byte_equal"]:
            return {**report, "status": "exact", "scientifically_equivalent": True}
        if comparison_ids[0] is None or comparison_ids[1] is None:
            return {
                **report,
                "reason": "Both reconstructions need an explicit scientific_identity.",
            }
        indices = tuple(DatasetOperations._index(root) for root in roots)
        members = tuple({member.member_id: member for member in index} for index in indices)
        if members[0].keys() != members[1].keys():
            return {
                **report,
                "status": "different",
                "scientifically_equivalent": False,
                "reason": "Member identifiers differ.",
            }
        differences: list[tuple[DatasetMember, DatasetMember, set[str]]] = []
        for key, a in members[0].items():
            b = members[1][key]
            if (
                a.partitions != b.partitions
                or a.targets != b.targets
                or a.metadata != b.metadata
                or a.assets.keys() != b.assets.keys()
            ):
                return {
                    **report,
                    "status": "different",
                    "scientifically_equivalent": False,
                    "reason": f"Exact partitions/targets/metadata/asset names differ for {key}.",
                }
            changed = {
                name
                for name in a.assets
                if a.assets[name].identity_dict() != b.assets[name].identity_dict()
            }
            if changed:
                if any(
                    a.assets[name].kind == "uri" or b.assets[name].kind == "uri" for name in changed
                ):
                    return {**report, "reason": "Changed URI assets require local materialization."}
                differences.append((a, b, changed))
        if verifier is None:
            return {
                **report,
                "reason": "Content differs; supply a trusted project verifier and policy.",
            }
        if not verifier_id or not policy:
            raise ValueError(
                "Scientific verification requires verifier_id and explicit variable policy."
            )
        normalized = json.loads(json.dumps(dict(policy), allow_nan=False))
        for variable, options in normalized.items():
            if not isinstance(options, dict) or not options.get("method"):
                raise ValueError(f"Verification policy for {variable} requires an explicit method.")
            for name in ("atol", "rtol"):
                if name in options and (
                    isinstance(options[name], bool)
                    or not isinstance(options[name], int | float)
                    or not math.isfinite(options[name])
                    or options[name] < 0
                ):
                    raise ValueError(f"Invalid {name} for {variable}.")
        equivalent = True
        for a, b, changed in differences:
            result = dict(
                verifier(
                    DatasetComparisonContext(
                        roots[0], roots[1], a, b, FrozenJsonMapping(normalized)
                    )
                )
            )
            json.dumps(result, allow_nan=False)
            if not isinstance(result.get("equivalent"), bool):
                raise ValueError("Project verifier must return an explicit equivalent boolean.")
            covered = result.get("checked_assets")
            if not isinstance(covered, list) or not changed.issubset(covered):
                raise ValueError("Project verifier did not check every changed member asset.")
            equivalent &= result["equivalent"]
            report["members"].append({"member": a.member_id, **result})
        if any(
            not DatasetOperations.verify(root, artifact.dataset_id)["valid"]
            for root, artifact in zip(roots, artifacts, strict=True)
        ):
            raise ValueError("Dataset bytes changed while the project verifier was running.")
        return {
            **report,
            "policy": normalized,
            "status": "equivalent" if equivalent else "different",
            "scientifically_equivalent": equivalent,
            "scope": "all members; project verification of every changed asset; not byte equality",
        }
