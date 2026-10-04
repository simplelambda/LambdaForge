"""Verified path relocation for concrete native invocations, not another input resolver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.PreparedWork import PreparedWork
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work.config import WorkConfig
from lambdaforge.work.managed import CANONICAL_FINGERPRINT_ALGORITHM, canonical_fingerprint


def _files(value: Any) -> dict[str, None]:
    """Collect only explicit typed markers; never guess that a plain string is a path."""
    output: dict[str, None] = {}
    if isinstance(value, Mapping):
        if set(value) == {"file"}:
            output[str(value["file"])] = None
        else:
            for child in value.values():
                output.update(_files(child))
    elif isinstance(value, list | tuple):
        for child in value:
            output.update(_files(child))
    return output


def prepared_input_bindings(
    source: Path,
    prepared: PreparedWork,
    equivalence: ExecutionEquivalence,
    *,
    invocations: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Verify the existing bundle, retaining scientific identity across machine-local paths.

    Inputs are staged by ExecutionBundleBuilder only. These bindings contain expected content,
    not permission to create/synchronize a remote mirror. Workers revalidate actual bytes.
    Remote existing environments are not attested by a caller saying they are ready: production
    preparation here requires an immutable managed environment identity.
    """
    manifest = json.loads(prepared.bundle.manifest_path.read_text(encoding="utf-8"))
    if manifest["code_identity"].get("provider") == "unversioned":
        raise ValueError("Prepared shards require identifiable consumer code.")
    if ScientificIdentity.from_payload(manifest["code_identity"]).digest != equivalence.code:
        raise ValueError("Prepared consumer code differs from the coordinator stratum.")
    if (
        prepared.profile.environment != "managed"
        or not prepared.bundle.environment_id
        or prepared.bundle.environment_id == "existing"
        or prepared.bundle.environment_id != equivalence.environment
    ):
        raise ValueError("Prepared shard needs the exact immutable managed environment identity.")
    config = WorkConfig.from_yaml(source)
    if len(config.levels) != 1 or len(config.levels[0].runs) != 1:
        raise ValueError("A prepared shard belongs to one Study definition.")
    if invocations is not None and any(
        value["definition"]["work_class"] != config.levels[0].runs[0].work_class
        for value in invocations.values()
    ):
        raise ValueError("Native invocations differ from the prepared Work class.")
    original = config.levels[0].runs[0].parameters

    # Dataset placement must be attested by its registry/content identity, not only by having
    # the same authored NAME@VERSION. That distributed placement integration is still gated.
    def contains_dataset(value: Any) -> bool:
        if isinstance(value, Mapping):
            return set(value) == {"dataset"} or any(
                contains_dataset(child) for child in value.values()
            )
        return isinstance(value, list | tuple) and any(contains_dataset(child) for child in value)

    if contains_dataset(original):
        raise ValueError("Distributed dataset placement attestation is not integrated yet.")
    staged = yaml.safe_load(prepared.bundle.config_path.read_text(encoding="utf-8")).get("with", {})
    pairs: dict[str, str] = {}

    def pair(left: Any, right: Any) -> None:
        if isinstance(left, Mapping):
            if set(left) == {"file"}:
                if not isinstance(right, Mapping) or set(right) != {"file"}:
                    raise ValueError("Bundle lost a declared file input.")
                old, new = str(left["file"]), str(right["file"])
                if old in pairs and pairs[old] != new:
                    raise ValueError("Bundle gives conflicting locations to one input.")
                pairs[old] = new
            else:
                if not isinstance(right, Mapping) or set(left) != set(right):
                    raise ValueError("Bundle changed scientific parameter keys.")
                for key in left:
                    pair(left[key], right[key])
        elif isinstance(left, list | tuple):
            if not isinstance(right, list | tuple) or len(left) != len(right):
                raise ValueError("Bundle changed a scientific parameter sequence.")
            for first, second in zip(left, right, strict=True):
                pair(first, second)
        elif left != right:
            raise ValueError("Bundle changed a scientific parameter value.")

    pair(original, staged)
    bindings: list[dict[str, Any]] = []
    for configured, destination in pairs.items():
        path = Path(configured)
        path = path if path.is_absolute() else source.parent / path
        digest, size = canonical_fingerprint(path)
        remote = PurePosixPath(destination)
        remote = remote if remote.is_absolute() else PurePosixPath(prepared.work_dir) / remote
        bindings.append(
            {
                "configured": configured,
                "destination": str(remote),
                "sha256": digest,
                "size_bytes": size,
                "algorithm": CANONICAL_FINGERPRINT_ALGORITHM,
            }
        )
    identity = [
        {key: item[key] for key in ("configured", "sha256", "size_bytes", "algorithm")}
        for item in sorted(bindings, key=lambda item: item["configured"])
    ]
    if ScientificIdentity.from_payload({"file_inputs": identity}).digest != equivalence.inputs:
        raise ValueError("Prepared input content differs from the coordinator stratum.")
    return bindings


def relocate_file_inputs(value: Any, bindings: list[dict[str, Any]]) -> Any:
    """Revalidate exact bytes before replacing typed file locations in native parameters."""
    locations: dict[str, str] = {}
    for item in bindings:
        if set(item) != {"configured", "destination", "sha256", "size_bytes", "algorithm"}:
            raise ValueError("Invalid prepared input binding fields.")
        configured, destination = item["configured"], item["destination"]
        if configured in locations or item["algorithm"] != CANONICAL_FINGERPRINT_ALGORITHM:
            raise ValueError("Prepared input binding must be unique and canonical.")
        path = Path(destination)
        if not path.is_absolute() or any(child.is_symlink() for child in (path, *path.parents)):
            raise ValueError("Prepared input destination must be absolute and non-symlinked.")
        digest, size = canonical_fingerprint(path)
        if digest != item["sha256"] or size != item["size_bytes"]:
            raise ValueError(f"Prepared input bytes changed: {configured}.")
        locations[configured] = destination
    if set(_files(value)) - locations.keys():
        raise ValueError("Prepared invocation contains an unbound file input.")

    def replace(item: Any) -> Any:
        if isinstance(item, Mapping):
            if set(item) == {"file"}:
                return {"file": locations[str(item["file"])]}
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, list | tuple):
            return [replace(child) for child in item]
        return item

    return replace(value)
