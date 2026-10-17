"""Host-scoped outer leases, with conservative inspection of pre-scoping records."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from lambdaforge.runtime.HostCoordination import host_identity, secure_root


def lease_directory(root: Path) -> Path:
    """Keep the old lock namespace while separating shared-home host records."""
    if root.absolute() != root.resolve():
        raise ValueError("Resource leases must not traverse symbolic paths.")
    root.mkdir(parents=True, exist_ok=True)
    return secure_root(root / "hosts" / host_identity())


def gpu_lease_name(device_uuid: str) -> str:
    """UUID/MIG identity, never CUDA or NVML ordinal, owns the lease key."""
    if not device_uuid.startswith(("GPU-", "MIG-")) or len(device_uuid) > 256:
        raise ValueError("A resolved GPU UUID is required for a host lease.")
    return "gpu-" + hashlib.sha256(device_uuid.encode()).hexdigest()[:32]


def read_lease(path: Path) -> dict[str, Any]:
    """Bound metadata reads and reject corrupt/unowned objects without deleting them."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > 65536
            or os.name == "posix"
            and info.st_uid != os.getuid()
        ):
            raise ValueError("Unsafe or oversized resource lease ownership evidence.")
        value = json.loads(stream.read(65537))
    if not isinstance(value, Mapping):
        raise ValueError("Resource lease has corrupt ownership evidence.")
    return dict(value)


def lease_records(root: Path, scoped: Path, prefix: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Read only this host plus legacy records; never inspect another host's PIDs."""
    for directory in (root, scoped):
        with os.scandir(directory) as entries:
            for index, entry in enumerate(entries):
                if index >= 10000:
                    raise ValueError("Resource lease listing exceeds bounded metadata policy.")
                if entry.name.startswith(prefix) and entry.name.endswith(".json"):
                    path = Path(entry.path)
                    value = read_lease(path)
                    if value.get("host") not in {None, host_identity()}:
                        continue
                    yield path, value
