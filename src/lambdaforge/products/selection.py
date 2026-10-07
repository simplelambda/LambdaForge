"""Deterministic post-Study model selection over explicit, checkpoint-bound artifact evidence.

This does not refit HPO, change its winner, or infer that latest/best Run metrics describe an
arbitrary checkpoint. The project must register the scored snapshot and its metrics explicitly.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from lambdaforge.ImmutableJson import FrozenJsonMapping
from lambdaforge.products.models import ProductArtifact, ProductContract, StudyProduct
from lambdaforge.products.registry import _encoded
from lambdaforge.work.managed import fingerprint


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Rank registered model snapshots; constraints use metrics of that same snapshot."""

    rank_by: str
    artifact: str = "model"
    mode: str = "max"
    group_by: tuple[str, ...] = ()
    top_k: int = 1
    constraints: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    tie_policy: str = "stable"
    practical_margin: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group_by, tuple | list):
            raise ValueError("Selection group_by must be an array of parameter names.")
        for text in (self.rank_by, self.artifact, *self.group_by):
            if not isinstance(text, str) or not text.strip() or len(text) > 256:
                raise ValueError(
                    "Selection names must be non-empty text of at most 256 characters."
                )
        if not isinstance(self.group_by, tuple | list) or len(set(self.group_by)) != len(
            self.group_by
        ):
            raise ValueError("Selection group_by must contain unique parameter names.")
        object.__setattr__(self, "group_by", tuple(sorted(self.group_by)))
        if self.mode not in {"min", "max"} or self.tie_policy not in {
            "stable",
            "include_equivalent",
        }:
            raise ValueError(
                "Selection requires mode=min|max and tie_policy=stable|include_equivalent."
            )
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("Selection top_k must be a positive integer.")
        if self.practical_margin is not None and (
            not _finite(self.practical_margin)
            or self.practical_margin < 0
            or self.tie_policy != "include_equivalent"
        ):
            raise ValueError(
                "Selection practical_margin requires include_equivalent "
                "and a finite non-negative value."
            )
        if not isinstance(self.constraints, Mapping):
            raise ValueError("Selection constraints must be a metric-bound mapping.")
        constraints = dict(self.constraints)
        for metric, bounds in constraints.items():
            if (
                not isinstance(metric, str)
                or not metric
                or not isinstance(bounds, Mapping)
                or not bounds
                or set(bounds) - {"min", "max"}
                or any(not _finite(value) for value in bounds.values())
            ):
                raise ValueError(
                    "Selection constraints require metric names and finite min/max bounds."
                )
            if bounds.get("min", -math.inf) > bounds.get("max", math.inf):
                raise ValueError("Selection constraint minimum exceeds maximum.")
        object.__setattr__(self, "constraints", FrozenJsonMapping(constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_by": self.rank_by,
            "artifact": self.artifact,
            "mode": self.mode,
            "group_by": list(self.group_by),
            "top_k": self.top_k,
            "constraints": dict(self.constraints),
            "tie_policy": self.tie_policy,
            "practical_margin": self.practical_margin,
            "rank_source": "registered-artifact-metrics",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SelectionPolicy:
        if not isinstance(value, Mapping):
            raise ValueError("Selection policy must be an object.")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown selection policy fields: {sorted(unknown)}.")
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Sealed immutable product and explicit selected file sources; publication is separate."""

    product: StudyProduct
    sources: tuple[tuple[str, Path], ...]


def select_models(
    source: Mapping[str, Any],
    execution_dir: Path,
    policy: SelectionPolicy,
    *,
    name: str,
    contract: str,
) -> ModelSelection:
    """Select latest completed full-fidelity snapshots, reading bytes of selected models only.

    Failure/pruning history remains in the source. Missing/incomplete/pruned Runs are ineligible,
    not fabricated objective values. A failed auxiliary Run does not erase other valid snapshots.
    Fleet paths outside this exact local Execution are deliberately rejected until attested import.
    """
    root = execution_dir.absolute()
    if root.resolve() != root or root.name != source.get("execution_id") or not root.is_dir():
        raise ValueError("Model selection requires its exact non-symbolic owned Execution root.")
    latest: dict[tuple[str, str, str, bytes], Mapping[str, Any]] = {}
    for row in source.get("runs", ()):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("run_id"), str)
            or type(row.get("attempt_number")) is not int
            or row["attempt_number"] < 1
        ):
            raise ValueError("Model selection requires valid authoritative Run/Attempt records.")
        key = (
            str(row.get("name", "")),
            row["run_id"],
            str(row.get("study_phase") or "search"),
            _encoded(row.get("fidelity") or {}),
        )
        previous = latest.get(key)
        if previous is None or row["attempt_number"] > previous["attempt_number"]:
            latest[key] = row
        elif row["attempt_number"] == previous["attempt_number"] and _encoded(row) != _encoded(
            previous
        ):
            raise ValueError("Conflicting records for the same logical Run/Attempt.")
    groups: dict[bytes, list[tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]]] = {}
    exclusions = []
    for row in latest.values():
        fidelity = row.get("fidelity") or {}
        if (
            row.get("status") != "succeeded"
            or row.get("pruned")
            or row.get("termination_type", "completed") != "completed"
            or (fidelity and int(fidelity["target"]) < int(fidelity["maximum"]))
        ):
            exclusions.append({"run_id": row["run_id"], "reason": "not-completed-full-fidelity"})
            continue
        artifacts = [
            artifact
            for artifact in row.get("artifacts", ())
            if isinstance(artifact, Mapping) and artifact.get("name") == policy.artifact
        ]
        if len(artifacts) != 1:
            exclusions.append({"run_id": row["run_id"], "reason": "missing-scored-model-artifact"})
            continue
        artifact = artifacts[0]
        if artifact.get("role") not in {"model", "checkpoint"}:
            raise ValueError(
                "Model selection artifact must explicitly declare model/checkpoint role."
            )
        metadata = artifact.get("metadata") or {}
        metrics = metadata.get("metrics", {}) if isinstance(metadata, Mapping) else {}
        if not isinstance(metrics, Mapping) or not _finite(metrics.get(policy.rank_by)):
            exclusions.append(
                {"run_id": row["run_id"], "reason": "missing-artifact-bound-ranking-metric"}
            )
            continue
        if any(
            not _finite(metrics.get(metric))
            or any(
                float(metrics[metric]) < bound if side == "min" else float(metrics[metric]) > bound
                for side, bound in bounds.items()
            )
            for metric, bounds in policy.constraints.items()
        ):
            exclusions.append(
                {"run_id": row["run_id"], "reason": "artifact-bound-constraint-not-satisfied"}
            )
            continue
        parameters = row.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("Selected Run parameters must be an authoritative JSON object.")
        group = {
            key: {"active": key in parameters, "value": parameters.get(key)}
            for key in policy.group_by
        }
        groups.setdefault(_encoded(group), []).append((row, artifact, group))
    selected = []
    for group_key in sorted(groups):
        rows = sorted(
            groups[group_key],
            key=lambda value: (
                (-1 if policy.mode == "max" else 1)
                * float(value[1]["metadata"]["metrics"][policy.rank_by]),
                str(value[0]["run_id"]),
                str(value[0].get("study_phase") or "search"),
            ),
        )
        chosen = rows[: policy.top_k]
        if policy.tie_policy == "include_equivalent" and chosen:
            boundary = float(chosen[-1][1]["metadata"]["metrics"][policy.rank_by])
            chosen.extend(
                row
                for row in rows[policy.top_k :]
                if abs(float(row[1]["metadata"]["metrics"][policy.rank_by]) - boundary)
                <= (policy.practical_margin or 0)
            )
        selected.extend(chosen)
    if not selected:
        reasons = sorted({row["reason"] for row in exclusions})
        raise ValueError(
            f"No eligible scored model snapshots: {reasons}. "
            "Register model artifacts with their own metadata.metrics; "
            "latest/best Run metrics cannot establish checkpoint identity."
        )
    product_artifacts = []
    source_files = []
    models = []
    inputs = {}
    for ordinal, (row, artifact, group) in enumerate(selected, 1):
        attempt = Path(str(row.get("run_dir", ""))).absolute()
        expected = root / "runs" / row["run_id"] / "attempts" / str(row.get("attempt_id", ""))
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", row["run_id"])
            or not re.fullmatch(r"attempt-[0-9]+", str(row.get("attempt_id", "")))
            or attempt != expected
            or attempt.resolve() != attempt
            or not attempt.is_relative_to(root / "runs")
        ):
            raise ValueError("Selected model Attempt path is outside its owned Execution.")
        relative = Path(str(artifact.get("path", "")))
        path = attempt / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.resolve() != path
            or not path.is_relative_to(attempt)
        ):
            raise ValueError("Selected model artifact path is unsafe or symbolic.")
        if not path.is_file():
            raise ValueError("Selected model artifact is missing or is not a regular file.")
        before = path.stat()
        if fingerprint(path) != (
            artifact.get("sha256"),
            artifact.get("size_bytes"),
        ):
            raise ValueError(
                "Selected model artifact is missing or differs from its exact persisted bytes."
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Selected model bytes changed while their snapshot was being sealed.")
        artifact_name = f"model-{ordinal}"
        destination = f"models/{artifact_name}/{path.name}"
        product_artifacts.append(
            ProductArtifact(
                artifact_name, destination, digest.hexdigest(), path.stat().st_size, "model"
            )
        )
        source_files.append((artifact_name, path))
        models.append(
            {
                "run_id": row["run_id"],
                "attempt_id": row["attempt_id"],
                "name": row.get("name"),
                "attempt_number": row["attempt_number"],
                "seed": row.get("seed"),
                "phase": row.get("study_phase"),
                "parameters": row.get("parameters", {}),
                "group": group,
                "artifact": artifact_name,
                "metrics": artifact["metadata"]["metrics"],
                "step": artifact["metadata"].get("step"),
                "rationale": "ranked eligible registered snapshot within its explicit group",
            }
        )
        for raw_input in row.get("inputs", ()):
            meaning = {
                key: raw_input.get(key)
                for key in ("name", "kind", "content_id", "sha256", "size_bytes")
            }
            if not meaning["sha256"] and not meaning["content_id"]:
                raise ValueError("Selected model input lacks authoritative exact content identity.")
            inputs[_encoded(meaning)] = meaning
    objective = (source.get("summary") or {}).get("objective")
    evidence = {
        "execution_id": source["execution_id"],
        "runs": list(latest.values()),
        "selection": policy.to_dict(),
    }
    product = StudyProduct(
        name,
        "ModelSet",
        ProductContract(contract, ("inputs", "objective", "design", "selection")),
        {
            "source_study": source["execution_id"],
            "models": models,
            "excluded": sorted(exclusions, key=lambda row: (row["run_id"], row["reason"])),
            "selection_scope": "eligible-observed-artifact-snapshots",
            "epoch_continuation_claimed": False,
        },
        {
            "inputs": [inputs[key] for key in sorted(inputs)],
            "objective": objective,
            "design": (source.get("summary") or {}).get("study_design"),
            "selection": policy.to_dict(),
        },
        {
            "execution_id": source["execution_id"],
            "evidence_fingerprint": "sha256:" + hashlib.sha256(_encoded(evidence)).hexdigest(),
            "config_fingerprint": source.get("scientific_fingerprint"),
            "source_status": source.get("status"),
        },
        tuple(product_artifacts),
    )
    return ModelSelection(product, tuple(source_files))
