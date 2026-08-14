"""Pinned, verified micromamba and offline Python package artifacts."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path
from uuid import uuid4


class MicromambaArtifactStore:
    """Cache one pinned official micromamba executable after SHA-256 verification."""

    VERSION = "2.8.1-0"
    RELEASE_ROOT = "https://github.com/mamba-org/micromamba-releases/releases/download/" + VERSION
    ARTIFACTS = {
        "linux-64": (
            "micromamba-linux-64",
            "9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82",
        ),
        "linux-aarch64": (
            "micromamba-linux-aarch64",
            "e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a",
        ),
        "linux-ppc64le": (
            "micromamba-linux-ppc64le",
            "321c822aaf4c2922cf3e653cbe8b44a86e28cda8f81f8fac3c0804744c7baf75",
        ),
    }
    MAXIMUM_BYTES = 64 * 1024 * 1024

    def __init__(self, root: str | Path = ".lambdaforge/control/runtime-installers") -> None:
        self.root = Path(root).resolve()

    def artifact(self, platform_tag: str) -> tuple[Path, str]:
        """Download a known release locally and return its verified exact bytes."""
        if platform_tag not in self.ARTIFACTS:
            raise RuntimeError(
                f"Managed micromamba installation does not support {platform_tag!r}; configure "
                "an existing compatible Python or Conda-family manager."
            )
        filename, expected = self.ARTIFACTS[platform_tag]
        destination = self.root / self.VERSION / filename
        if destination.is_file() and self._sha256(destination) == expected:
            return destination, expected
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        request = urllib.request.Request(
            f"{self.RELEASE_ROOT}/{filename}", headers={"User-Agent": "LambdaForge-runtime"}
        )
        try:
            digest = hashlib.sha256()
            size = 0
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.MAXIMUM_BYTES:
                        raise RuntimeError(
                            "Micromamba artifact exceeded the bounded download size."
                        )
                    digest.update(chunk)
                    out.write(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(
                    f"Micromamba checksum mismatch for {filename}; the artifact was not staged."
                )
            temporary.chmod(0o755)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, expected

    def offline_packages(self, platform_tag: str, python_version: str) -> Path:
        """Prefetch an exact target-platform Python solve for an offline remote cluster."""
        executable, _ = self.artifact(self._local_platform_tag())
        key = hashlib.sha256(
            f"{self.VERSION}:{platform_tag}:python={python_version}:pip".encode()
        ).hexdigest()[:20]
        destination = self.root / "packages" / key
        marker = destination / ".complete"
        if marker.is_file():
            return destination / "root" / "pkgs"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        root = temporary / "root"
        prefix = temporary / "solve-prefix"
        temporary.mkdir(parents=True)
        environment = dict(
            os.environ,
            MAMBA_ROOT_PREFIX=str(root),
            CONDA_PKGS_DIRS=str(root / "pkgs"),
            CONDARC=os.devnull,
            MAMBARC=os.devnull,
        )
        try:
            completed = subprocess.run(
                (
                    str(executable),
                    "create",
                    "--yes",
                    "--download-only",
                    "--prefix",
                    str(prefix),
                    "--platform",
                    platform_tag,
                    "--override-channels",
                    "--channel",
                    "conda-forge",
                    f"python={python_version}",
                    "pip",
                ),
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                env=environment,
            )
            if completed.returncode:
                raise RuntimeError(
                    "Could not prefetch the managed Python packages for an offline cluster: "
                    f"{completed.stderr.strip()}"
                )
            (temporary / ".complete").write_text("verified package cache\n", encoding="utf-8")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination / "root" / "pkgs"

    @classmethod
    def _local_platform_tag(cls) -> str:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        if system != "linux":
            raise RuntimeError(
                "Offline target-package prefetch currently requires a Linux controller."
            )
        suffix = {
            "x86_64": "64",
            "amd64": "64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
            "ppc64le": "ppc64le",
        }.get(architecture)
        if suffix is None:
            raise RuntimeError(f"Unsupported controller architecture: {architecture!r}.")
        return f"linux-{suffix}"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
