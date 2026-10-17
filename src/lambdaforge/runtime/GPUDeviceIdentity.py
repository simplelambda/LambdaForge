"""Pin CUDA-visible ordinals to UUIDs once, separately from ordinary NVML measurements.

UUID grants need no CUDA call. Legacy numeric grants require a one-time bounded driver
enumeration (cuInit/cuDeviceGetUuid_v2), never a context, allocation or Torch import.
Never assume NVML indices equal CUDA ordinals, including when FASTEST_FIRST is inherited.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.runtime.GPUObservation import GPUObservationCache
from lambdaforge.runtime.HostCoordination import boot_identity, coordination_root
from lambdaforge.work.atomic import atomic_write_json


def visible_devices(
    devices: Sequence[Mapping[str, Any]], count: int
) -> tuple[Mapping[str, Any], ...]:
    """Return only the original process grant, in its actual CUDA order."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = tuple(t.strip() for t in (raw or "").split(",") if t.strip())
    if raw is not None and (
        not tokens or any(t.lower() in {"-1", "none", "void", "nodevfiles"} for t in tokens)
    ):
        raise RuntimeError("GPU visibility is empty/disabled; no device can be admitted.")
    if tokens and all(t.startswith(("GPU-", "MIG-")) for t in tokens):
        ids = tokens[:count]
    else:
        import psutil

        scope = json.dumps(
            [
                boot_identity(),
                raw,
                os.environ.get("CUDA_DEVICE_ORDER"),
                os.environ.get("LAMBDAFORGE_JOB_ID"),
                os.getpid(),
                psutil.Process().create_time(),
            ]
        )
        key = hashlib.sha256(scope.encode()).hexdigest()
        root = coordination_root("gpu-observation")
        path = root / f"identity-{key}.json"
        with CrossProcessFileLock(
            root / f"identity-{key}.lock",
            shared=False,
            timeout_seconds=7,
            poll_interval_seconds=0.01,
        ):
            if path.is_symlink():
                raise ValueError("GPU identity cache must not be a symbolic link.")
            if path.is_file():
                if path.stat().st_size > 65536:
                    raise ValueError("GPU identity cache is oversized.")
                cached = json.loads(path.read_text())
                if cached.get("error") and time.time() < cached.get("retry_after", 0):
                    raise RuntimeError("Visible GPU identity probe is in backoff.")
                ids = tuple(cached.get("visible_uuids", ()))
            else:
                ids = ()
            if not ids:
                try:
                    decoded = GPUObservationCache().visible_identity()
                except Exception:
                    atomic_write_json(
                        path, {"error": "unavailable", "retry_after": time.time() + 30}
                    )
                    raise
                if (
                    not isinstance(decoded, list)
                    or len(decoded) > 256
                    or not all(isinstance(t, str) and t.startswith("GPU-") for t in decoded)
                ):
                    raise ValueError("Invalid visible GPU identity response.")
                ids = tuple(decoded)
                atomic_write_json(path, {"visible_uuids": ids})
        ids = ids[:count]
    if len(ids) != count:
        raise RuntimeError(
            "Fewer visible GPU identities than requested; refusing to broaden grant."
        )
    selected = []
    for token in ids:
        matches = [
            row
            for row in devices
            if (
                isinstance(row.get("uuid"), str)
                and (
                    row["uuid"].startswith(token)
                    or row["uuid"].removeprefix("MIG-") == token.removeprefix("GPU-")
                    or token in row.get("aliases", ())
                )
            )
        ]
        if len(matches) != 1 or matches[0] in selected:
            raise RuntimeError("GPU grant UUID is missing/ambiguous; admission paused.")
        selected.append(matches[0])
    return tuple(selected)


def driver_visible_uuids() -> list[str]:
    """Read device identities only. Never create/retain a CUDA compute context."""
    import ctypes
    import uuid

    class CUuuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    query = driver.cuDeviceGetUuid_v2
    query.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]
    if driver.cuInit(0) != 0:
        raise RuntimeError("CUDA identity enumeration unavailable.")
    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or not 0 <= count.value <= 256:
        raise RuntimeError("CUDA identity enumeration invalid.")
    values = []
    for index in range(count.value):
        device, identifier = ctypes.c_int(), CUuuid()
        if driver.cuDeviceGet(ctypes.byref(device), index) or query(
            ctypes.byref(identifier), device
        ):
            raise RuntimeError("CUDA visible UUID cannot be read safely.")
        values.append("GPU-" + str(uuid.UUID(bytes=bytes(identifier.bytes))))
    return values


if __name__ == "__main__":
    print(json.dumps(driver_visible_uuids()))
