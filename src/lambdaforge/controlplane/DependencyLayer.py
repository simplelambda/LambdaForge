"""Dependency contracts independent of immutable execution wheel payloads."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from lambdaforge.controlplane.EnvironmentPublication import PUBLICATION_CONTRACT


class DependencyLayer:
    """Identify a frozen dependency resolution; exact execution wheels remain separate.

    The first build resolves dependencies with the native installer and retains its
    bootstrap wheels. Later execution layers shadow those wheels, never upgrade the
    base. Changing a wheel's dependency contract invalidates the base identity.
    """

    @staticmethod
    def policy(
        wheels: Sequence[Path],
        policy: Mapping[str, Any],
        *,
        offline_wheels: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        contracts: list[dict[str, Any]] = []
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                names = [
                    name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
                ]
                if len(names) != 1:
                    raise ValueError(f"Wheel requires one dependency metadata record: {wheel.name}")
                metadata = BytesParser().parsebytes(archive.read(names[0]))
            contracts.append(
                {
                    "name": canonicalize_name(str(metadata["Name"])),
                    "requires_python": str(metadata.get("Requires-Python", "")),
                    "requires_dist": sorted(metadata.get_all("Requires-Dist", [])),
                    "provides_extra": sorted(metadata.get_all("Provides-Extra", [])),
                }
            )
        # Operational discovery notes, activation paths and reuse actions are not ABI.
        normalized = json.loads(json.dumps(policy))
        runtime = normalized.get("python_runtime")
        if isinstance(runtime, dict):
            for key in ("action", "ready", "executable", "tls_trust"):
                runtime.pop(key, None)
        torch = normalized.get("pytorch")
        if isinstance(torch, dict):
            for key in ("reason", "driver_version", "compute_capabilities", "require_cuda"):
                torch.pop(key, None)
        native = normalized.get("native_environment")
        if isinstance(native, dict):
            for key in (
                "specification_name",
                "specification_sha256",
                "specification_kind",
                "requested_packages",
                "required_executables",
            ):
                native.pop(key, None)
        payload = {
            "version": 1,
            "contracts": sorted(contracts, key=lambda value: value["name"]),
            "resolution_policy": normalized,
            "offline_wheels": sorted(offline_wheels, key=lambda item: str(item["name"])),
        }
        legacy = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload.update(version=2, publication_contract=PUBLICATION_CONTRACT)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return {
            "identity": f"deps-{digest[:24]}",
            "legacy_identity": f"deps-{legacy[:24]}",
            **payload,
        }

    @staticmethod
    def previous_identity(layer: Mapping[str, Any]) -> str:
        """Validate migration provenance instead of trusting a filesystem selector."""
        payload = {key: layer[key] for key in ("contracts", "resolution_policy", "offline_wheels")}
        payload["version"] = 1
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        expected = f"deps-{digest[:24]}"
        if layer.get("legacy_identity") != expected:
            raise ValueError("Invalid previous dependency publication identity.")
        return expected
