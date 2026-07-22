"""Typed capture of the software and hardware environment for one run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from lambdaforge.LambdaForge import LambdaForge
from lambdaforge.plugins.PluginDescriptor import PluginDescriptor


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    """Serializable provenance for software, hardware, Git and used plugins."""

    created_at_utc: str
    lambdaforge_version: str
    python: dict[str, Any]
    platform: dict[str, Any]
    packages: dict[str, str | None]
    torch: dict[str, Any]
    git: dict[str, Any]
    environment: dict[str, str]
    plugins: tuple[PluginDescriptor, ...] = ()

    TRACKED_PACKAGES = (
        "lightning",
        "pytorch-lightning",
        "mlflow",
        "tensorboard",
        "tensorboardX",
        "wandb",
        "jsonschema",
        "matplotlib",
        "numpy",
        "psutil",
        "PyYAML",
        "ruamel.yaml",
        "torch",
        "torchmetrics",
    )

    @classmethod
    def capture(
        cls,
        repository: str | Path | None = None,
        *,
        plugins: Iterable[PluginDescriptor] = (),
    ) -> EnvironmentManifest:
        """Capture current provenance without mutating process or device state."""
        plugin_snapshot = tuple(sorted(set(plugins), key=PluginDescriptor.sort_key))
        return cls(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            lambdaforge_version=LambdaForge.VERSION,
            python={
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            platform={
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            packages={name: cls._package_version(name) for name in cls.TRACKED_PACKAGES},
            torch=cls._torch_info(),
            git=cls._git_info(Path(repository or Path.cwd())),
            environment={
                key: os.environ[key] for key in ("CUDA_VISIBLE_DEVICES",) if key in os.environ
            },
            plugins=plugin_snapshot,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible mapping."""
        payload = asdict(self)
        payload["plugins"] = [descriptor.to_dict() for descriptor in self.plugins]
        return payload

    def with_plugins(self, plugins: Iterable[PluginDescriptor]) -> EnvironmentManifest:
        """Return the same environment snapshot with deterministic run usage."""
        plugin_snapshot = tuple(sorted(set(plugins), key=PluginDescriptor.sort_key))
        return replace(self, plugins=plugin_snapshot)

    def write(self, path: str | Path) -> Path:
        """Atomically write a stable UTF-8 JSON artifact and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    @staticmethod
    def _torch_info() -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()
        devices: list[dict[str, Any]] = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        return {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": len(devices),
            "devices": devices,
        }

    @classmethod
    def _git_info(cls, repository: Path) -> dict[str, Any]:
        root = cls._git(repository, "rev-parse", "--show-toplevel")
        if root is None:
            return {"available": False}
        status = cls._git(Path(root), "status", "--porcelain")
        return {
            "available": True,
            "root": root,
            "commit": cls._git(Path(root), "rev-parse", "HEAD"),
            "branch": cls._git(Path(root), "branch", "--show-current"),
            "dirty": bool(status),
        }

    @staticmethod
    def _git(repository: Path, *arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()
