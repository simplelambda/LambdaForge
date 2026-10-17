"""Retained-runtime attestation for same-placement fixed Study extensions."""

from __future__ import annotations

from typing import Any


def runtime_contract(environment: dict[str, Any]) -> dict[str, Any]:
    """Compare numerical provenance, not timestamps, machine paths or GPU ordinals."""
    import json

    if environment.get("plugins") or environment.get("external_tools"):
        raise ValueError("Extension cannot attest plugin/external-tool numerical contracts.")
    python, platform = environment.get("python", {}), environment.get("platform", {})
    packages, torch = environment.get("packages"), environment.get("torch", {})
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Historical dependency provenance is missing.")
    devices = sorted(
        {
            json.dumps(
                {
                    key: device.get(key)
                    for key in (
                        "name",
                        "capability",
                        "total_memory_bytes",
                    )
                },
                sort_keys=True,
            )
            for device in torch.get("devices", [])
        }
    )
    return {
        "framework": environment.get("lambdaforge_version"),
        "python": {key: python.get(key) for key in ("version", "implementation")},
        "platform": {key: platform.get(key) for key in ("system", "machine", "processor")},
        "packages": packages,
        "torch": {key: torch.get(key) for key in ("version", "cuda_version", "cudnn_version")},
        "device_types": devices,
    }


def inspect_retained_runtime(
    prefix: str, environment_id: str, contracts: list[dict[str, Any]], require_sparse: bool = False
) -> dict[str, Any]:
    """Run with the retained Python, without Torch imports or CUDA device initialization."""
    import hashlib
    import json
    import platform
    import sys
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    root = Path(prefix)
    marker = root / ".lambdaforge-environment.json"
    if (
        not root.is_absolute()
        or root.resolve() != root
        or Path(sys.prefix) != root
        or marker.resolve() != marker
        or marker.stat().st_size > 256 * 1024
    ):
        raise ValueError("Retained extension environment is missing, symbolic or unowned.")
    raw = marker.read_bytes()
    value = json.loads(raw)
    if value.get("environment_id") != environment_id or not value.get("packages"):
        raise ValueError("Retained managed environment identity is missing or changed.")
    if not contracts or any(contract != contracts[0] for contract in contracts):
        raise ValueError("Historical evidence has mixed numerical/runtime contracts.")
    expected = contracts[0]
    if expected["python"] != {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    } or expected["platform"] != {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }:
        raise ValueError("Retained Python/platform differs from the historical runtime.")
    for name, required in {**expected["packages"], "lambdaforge": expected["framework"]}.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        if actual != required:
            raise ValueError(f"Retained dependency {name!r} changed; extension was not started.")
    if require_sparse:
        from lambdaforge.work import config

        if getattr(config, "FIXED_SWEEP_POINTS_VERSION", 0) != 1:
            raise ValueError(
                "Retained LambdaForge runtime lacks native sparse sweep support. "
                "No extension was submitted; this immutable environment cannot execute "
                "the reviewed conditional delta. Do not edit it in place or rerun the full Study."
            )
    return {
        "prefix": prefix,
        "python": str(root / "bin/python"),
        "environment_id": environment_id,
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "contract": expected,
        **({"sparse_sweeps": True} if require_sparse else {}),
    }


def check_worker_runtime(expected: dict[str, Any]) -> None:
    """Check actual granted GPU hardware inside the native site/scheduler launcher."""
    from lambdaforge.EnvironmentManifest import EnvironmentManifest

    actual = runtime_contract(EnvironmentManifest.capture().to_dict())
    if actual != expected:
        raise ValueError(
            "Extension numerical environment/hardware differs from its reviewed origin; "
            "no new scientific Run was started. Use a separate Study for incompatible evidence."
        )
