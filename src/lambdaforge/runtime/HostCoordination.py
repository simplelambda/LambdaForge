"""User-owned host coordination roots; never a GPU allocation or scientific identity."""

from __future__ import annotations

import hashlib
import os
import socket
import stat
from pathlib import Path


def host_identity() -> str:
    """Scope shared-home metadata to this host without publishing machine identifiers."""
    machine = Path("/etc/machine-id")
    identity = socket.gethostname()
    if machine.is_file():
        identity += machine.read_text(encoding="ascii")[:1024]
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def boot_identity() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.is_file():
        return path.read_text(encoding="ascii").strip()
    import psutil

    return str(psutil.boot_time())


def secure_root(path: Path) -> Path:
    """Reject linked, foreign or publicly writable coordination, rather than adopting it."""
    path = path.expanduser().absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Host coordination must not traverse symbolic links.")
    if os.name == "posix":
        for parent in path.parents:
            if not parent.exists():
                continue
            mode = parent.stat().st_mode
            if mode & 0o022 and not mode & stat.S_ISVTX:
                raise ValueError("Host coordination ancestor is writable by other users.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or (
        os.name == "posix" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077)
    ):
        raise ValueError("Host coordination requires a private user-owned directory (0700).")
    return path


def coordination_root(category: str) -> Path:
    """Same UID/host, independent of project or remapped CUDA ordinals."""
    if category not in {"gpu-observation", "gpu-admission"}:
        raise ValueError("Unknown host coordination category.")
    base = Path(
        os.environ.get(
            "LAMBDAFORGE_HOST_COORDINATION_ROOT", Path.home() / ".cache/lambdaforge/hosts"
        )
    )
    return secure_root(base / host_identity() / category)
