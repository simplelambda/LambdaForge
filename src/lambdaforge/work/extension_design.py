"""Exact native fixed-sweep deltas, including sparse conditional designs."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.config import RunDefinition, WorkConfig


def plain(value: Any) -> Any:
    """Detach immutable native JSON without numeric coercion or tolerance."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def exact(value: Any) -> str:
    """Native JSON numeric/type semantics: true, 1 and 1.0 are not interchangeable."""
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_id(parameters: Mapping[str, Any]) -> str:
    """Stable across reordering, not a Run fingerprint or a historical Trial ordinal."""
    return "candidate-" + ScientificIdentity.from_payload(
        {"candidate_identity_version": 1, "parameters": plain(parameters)}
    ).digest.removeprefix("sha256:")


def cell_id(candidate: str, seed: int) -> str:
    return "cell-" + ScientificIdentity.from_payload(
        {"extension_cell_version": 1, "candidate_id": candidate, "seed": seed}
    ).digest.removeprefix("sha256:")


def fixed_definition(config: WorkConfig) -> RunDefinition:
    """Single native fixed sweep with explicit seeds and scalar dimensions."""
    if len(config.levels) != 1 or len(config.levels[0].runs) != 1 or "steps" in config.raw:
        raise ValueError("Study extension requires one Work, not composed levels.")
    definition = config.levels[0].runs[0]
    design = definition.study_design
    if definition.search_policy is not None or design is None or design.kind != "sweep":
        raise ValueError(
            "Study extension supports fixed sweeps, not adaptive HPO or repeated designs."
        )
    if design.replication != "fixed" or not isinstance(config.raw.get("seeds"), (list, tuple)):
        raise ValueError("Study extension requires fixed replication with explicit seeds.")
    if not isinstance(config.raw.get("sweep"), Mapping):
        raise ValueError("Use an explicit sweep.space design for Study extension.")
    if definition.objective is None:
        raise ValueError("Study extension requires an explicit scientific objective.")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in definition.seeds):
        raise ValueError("Study extension requires exact explicit integer seeds.")
    if any(
        isinstance(value, (Mapping, list, tuple))
        for variant in definition.variants
        for value in variant.values()
    ):
        raise ValueError("Structured sweep dimensions require a declared extension contract.")
    return definition


def candidate_rows(definition: RunDefinition) -> list[dict[str, Any]]:
    rows = [
        {
            "candidate_id": candidate_id(variant),
            "original_trial": index,
            "parameters": plain(variant),
        }
        for index, variant in enumerate(definition.variants, 1)
    ]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate normalized candidates make extension provenance ambiguous.")
    return rows


@dataclass(frozen=True)
class FixedExtensionComparison:
    """Immutable serialized comparison, with an exact native delta document."""

    serialized: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.serialized)


def compare_fixed_designs(
    previous: WorkConfig,
    proposed: WorkConfig,
    *,
    fixed_identities: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> FixedExtensionComparison:
    """Compare required cells exactly, independently of display/Trial indices.

    Input *content* compatibility is verified by the domain service. This comparator checks
    normalized science and shape; it never reads evidence, edits a file or consumes a seed.
    """
    old, new = fixed_definition(previous), fixed_definition(proposed)
    assert old.study_design is not None and new.study_design is not None
    for label, left, right in (
        ("Work class", old.work_class, new.work_class),
        ("fixed parameters/inputs", *(fixed_identities or (old.parameters, new.parameters))),
        ("objective", old.objective, new.objective),
        ("metric semantics", old.analysis_semantics, new.analysis_semantics),
        ("product contracts", previous.raw.get("products"), proposed.raw.get("products")),
        ("sweep reference", old.study_design.reference, new.study_design.reference),
    ):
        if exact(left) != exact(right):
            raise ValueError(f"Incompatible Study extension: changed {label}.")
    old_dimensions, new_dimensions = set(old.study_design.space), set(new.study_design.space)
    if old_dimensions != new_dimensions:
        raise ValueError("Extension cannot add/remove a dimension; only its values may grow.")
    for name in sorted(old_dimensions):
        for policy in ("when", "scale", "type"):
            default = "linear" if policy == "scale" else None
            left = old.study_design.space[name].get(policy, default)
            right = new.study_design.space[name].get(policy, default)
            if exact(left) != exact(right):
                raise ValueError(
                    f"Incompatible Study extension: changed {name!r} {policy} contract."
                )
    sparse = any("when" in rule for rule in new.study_design.space.values()) or any(
        key in config.raw["sweep"]
        for config in (previous, proposed)
        for key in ("include", "candidates")
    )
    old_rows, new_rows = candidate_rows(old), candidate_rows(new)
    old_by_id = {row["candidate_id"]: row for row in old_rows}
    new_by_id = {row["candidate_id"]: row for row in new_rows}
    if not set(old_by_id).issubset(new_by_id):
        raise ValueError(
            "Old normalized candidates are missing or changed. Numeric/type differences are "
            "not equivalence; no float tolerance is applied."
        )
    old_seeds = [cast(int, seed) for seed in old.seeds]
    new_seeds = [cast(int, seed) for seed in new.seeds]
    if new_seeds[: len(old_seeds)] != old_seeds:
        raise ValueError(
            "Old explicit seeds and their coordinates must remain an unchanged prefix."
        )
    if exact(plain(old.seed_metadata)) != exact(plain(new.seed_metadata[: len(old_seeds)])):
        raise ValueError("Explicit seed namespace/coordinates changed; extension is not recovery.")
    changed: dict[str, list[Any]] = {}
    for name in sorted(old_dimensions):
        earlier = {exact(row["parameters"][name]) for row in old_rows if name in row["parameters"]}
        observed: dict[str, Any] = {
            exact(row["parameters"][name]): row["parameters"][name] for row in new_rows
            if name in row["parameters"]
        }
        if not earlier.issubset(observed):
            raise ValueError(f"Old values of {name!r} were removed or changed.")
        added = [value for key, value in observed.items() if key not in earlier]
        if added:
            changed[name] = added
    added_seeds = new_seeds[len(old_seeds) :]
    new_candidate_ids = set(new_by_id) - set(old_by_id)
    if (not sparse and len(changed) > 1) or new_candidate_ids and added_seeds:
        raise ValueError(
            "Extend one dimension OR explicit seeds at a time. Use successive revisions for "
            "mixed changes; no overlapping Cartesian replay is permitted."
        )
    reused, pending = [], []
    for row in new_rows:
        for seed in new_seeds:
            cell = {
                "cell_id": cell_id(row["candidate_id"], seed),
                "candidate_id": row["candidate_id"],
                "parameters": row["parameters"],
                "seed": seed,
                "proposed_trial": row["original_trial"],
            }
            if row["candidate_id"] in old_by_id and seed in old_seeds:
                reused.append(
                    {**cell, "original_trial": old_by_id[row["candidate_id"]]["original_trial"]}
                )
            else:
                pending.append(cell)
    delta = copy.deepcopy(plain(proposed.raw))
    if new_candidate_ids and sparse:
        # A conditional delta is generally not a Cartesian rectangle. Persist its full
        # authored geometry and only the exact new candidates; inactive keys stay absent.
        delta["sweep"] = {
            "space": plain(new.study_design.space),
            "candidates": [row["parameters"] for row in new_rows
                           if row["candidate_id"] in new_candidate_ids],
        }
        delta["seeds"] = old_seeds
        delta.pop("replicates", None)
        kind = "candidates"
    elif changed:
        name = next(iter(changed))
        # Reify the native normalized dimensions, including native finite range samples.
        delta["sweep"] = {
            "space": {
                dimension: {"values": values}
                for dimension, values in (
                    (
                        dimension,
                        changed[dimension]
                        if dimension in changed
                        else list(
                            {
                                exact(row["parameters"][dimension]): row["parameters"][dimension]
                                for row in new_rows
                            }.values()
                        ),
                    )
                    for dimension in sorted(new_dimensions)
                )
            }
        }
        delta["seeds"] = old_seeds
        delta.pop("replicates", None)
        kind = "values"
    elif added_seeds:
        delta["seeds"] = added_seeds
        delta.pop("replicates", None)
        kind = "seeds"
    else:
        delta = None
        kind = "no-op"
    # Native publication is not automatically repeated on a post-hoc delta. Retain the full
    # author declaration in the comparison, and require a separate native product decision.
    if delta is not None:
        delta.pop("products", None)
        native_delta = WorkConfig.from_mapping(delta, source=proposed.source)
        actual = {
            cell_id(candidate_id(variant), cast(int, seed))
            for variant in native_delta.levels[0].runs[0].variants
            for seed in native_delta.levels[0].runs[0].seeds
        }
        expected = {cell["cell_id"] for cell in pending}
        if actual != expected or actual & {cell["cell_id"] for cell in reused}:
            raise ValueError("Native delta does not exactly match the disjoint new evidence cells.")
    return FixedExtensionComparison(
        exact(
            {
                "extension_comparison_version": 1,
                "kind": kind,
                "added_values": changed,
                "added_seeds": added_seeds,
                "reused_cells": reused,
                "new_cells": pending,
                "counts": {
                    "reused": len(reused),
                    "new": len(pending),
                    "total": len(new_rows) * len(new_seeds),
                },
                "delta_configuration": delta,
                "proposed_configuration": plain(proposed.raw),
                "proposed_normalized": proposed.resolved_configuration(),
                "operational_changes": {
                    "resources": {
                        "previous": old.resources.to_dict(),
                        "proposed": new.resources.to_dict(),
                    },
                    "execution": {
                        "previous": old.execution_policy.to_dict(),
                        "proposed": new.execution_policy.to_dict(),
                    },
                },
                "interpretation": "exploratory post-hoc extension; no automatic confirmatory claim",
            }
        )
    )
