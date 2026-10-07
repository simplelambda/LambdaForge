"""Filesystem-specific, conservative storage commitments for direct Jobs.

Reservations are future commitments, not measured usage. They are shared across projects
on a host and released only on verified owner death or explicit supervisor completion.
"""

from __future__ import annotations

import errno
import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.models import atomic_json


class StorageAdmission:
    """Observe and atomically reserve one physical filesystem; never pool free volumes."""

    @classmethod
    @contextmanager
    def transaction(
        cls, destination: Path, requested_bytes: int, *, purpose: str
    ) -> Iterator[None]:
        """Commit temporary copy space on its destination volume, not the Run volume.

        Copies are additional physical work, even when the enclosing Job already owns
        a workspace commitment. No lease is needed outside a managed worker: standalone
        use retains its ordinary filesystem semantics. Failure never publishes bytes.
        """
        import json
        from uuid import uuid4

        import psutil

        raw = os.environ.get("LAMBDAFORGE_STORAGE_POLICY")
        if not raw:
            yield
            return
        destination = destination.absolute()
        if any(part.is_symlink() for part in (destination, *destination.parents)):
            raise ValueError("Storage transaction destination cannot be symbolic.")
        policy = json.loads(raw)
        descriptor = {
            **policy,
            "run_root": str(destination.parent),
            "transaction_purpose": purpose,
            "transaction_destination": str(destination),
        }
        owner = ProcessIdentity.create(
            os.getpid(), os.getpgrp(), psutil.Process().cmdline(), f"job-storage-{uuid4().hex}"
        )
        try:
            observed = cls.reserve(descriptor, owner, requested_bytes)
            if not observed.get("admitted"):
                from lambdaforge.controlplane.StorageOperations import StorageOperations

                cache = cls.filesystem(Path(descriptor["cache_root"]), descriptor)
                deficit = max(
                    0,
                    requested_bytes
                    + int(observed.get("reserved_bytes", 0))
                    + observed["safety_bytes"]
                    - observed["free_bytes"],
                )
                if cache["device"] == observed["device"] and deficit:
                    try:
                        StorageOperations.gc(
                            {**policy, "automatic_gc": True, "target_reclaim_bytes": deficit},
                            {},
                            apply=True,
                        )
                    except (OSError, TimeoutError):
                        pass
                    observed = cls.reserve(descriptor, owner, requested_bytes)
                if not observed.get("admitted"):
                    raise OSError(
                        errno.ENOSPC,
                        f"Storage {purpose} blocked on destination filesystem: "
                        f"requested={requested_bytes}, free={observed['free_bytes']}, "
                        f"reserved={observed.get('reserved_bytes', 0)}, "
                        f"safety={observed['safety_bytes']} bytes ({observed['reason']}). "
                        "No partial publication was committed; free space or choose "
                        "another volume.",
                        str(destination),
                    )
            yield
        finally:
            cls.release(descriptor, owner)

    @staticmethod
    def filesystem(path: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
        probe = path.absolute()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        stats = os.statvfs(probe)
        total = stats.f_blocks * stats.f_frsize
        free = stats.f_bavail * stats.f_frsize
        # Only safety belongs to the physical probe. GC's internal zero-age override
        # must not be revalidated as an authored retention profile here.
        policy = ClusterStoragePolicy.from_mapping(
            {"safety": descriptor.get("safety", {})}, workspace=str(probe)
        )
        safety = max(
            policy.safety_min_free_bytes, int(total * policy.safety_min_free_percent / 100)
        )
        usable = free - safety
        pressure = (
            "CRITICAL"
            if free <= safety // 2 or (stats.f_files > 0 and stats.f_favail == 0)
            else "HARD_PRESSURE"
            if usable < 0
            else "SOFT_PRESSURE"
            if free < 2 * safety
            else "NORMAL"
        )
        return {
            "device": str(probe.stat().st_dev),
            "capacity_bytes": total,
            "free_bytes": free,
            "used_bytes": total - free,
            "safety_bytes": safety,
            "free_inodes": stats.f_favail,
            "inode_accounting": stats.f_files > 0,
            "pressure": pressure,
        }

    @classmethod
    def observe(
        cls, descriptor: Mapping[str, Any], *, owner_job_id: str | None = None
    ) -> dict[str, Any]:
        """Small read-only root/lease observation; never recursively inventories content."""
        from lambdaforge.controlplane.StorageOperations import StorageOperations

        roots = StorageOperations._roots(descriptor)
        volumes: dict[str, Any] = {}
        for name in ("state", "environments", "job_workspaces", "datasets"):
            observed = cls.filesystem(roots[name], descriptor)
            volume = volumes.setdefault(
                str(observed["device"]),
                {
                    **observed,
                    "roots": [],
                    "reserved_bytes": 0,
                    "reservations": 0,
                    "ownership_unresolved": False,
                },
            )
            volume["roots"].append(name)
        for device, volume in volumes.items():
            for path in (cls._lease_root(descriptor) / device).glob("job-*.json"):
                record = StorageOperations._read_json(path)
                if record is None:
                    volume["ownership_unresolved"] = True
                    continue
                if record.get("host") != socket.gethostname():
                    # A missing local PID says nothing about an owner on a shared mount.
                    volume["ownership_unresolved"] = True
                    continue
                try:
                    owner = ProcessIdentity.from_mapping(record["owner"])
                    cls._validate_owner(owner)
                    requested = record["requested_bytes"]
                    if (
                        isinstance(requested, bool)
                        or not isinstance(requested, int)
                        or requested < 0
                    ):
                        raise ValueError("Corrupt storage commitment")
                    if cls._dead(owner.pid):
                        continue
                    if not owner.matches():
                        volume["ownership_unresolved"] = True
                    if owner.job_id == owner_job_id and owner.matches():
                        continue  # This owner's existing commitment is not a second reservation.
                    volume["reserved_bytes"] += requested
                    volume["reservations"] += 1
                except (KeyError, TypeError, ValueError):
                    volume["ownership_unresolved"] = True
            remaining = volume["free_bytes"] - volume["reserved_bytes"] - volume["safety_bytes"]
            volume["admissible_bytes"] = max(0, remaining)
            if volume["ownership_unresolved"]:
                volume["admissible_bytes"] = 0
            if remaining < 0 and volume["pressure"] != "CRITICAL":
                volume["pressure"] = "HARD_PRESSURE"
            elif remaining < volume["safety_bytes"] and volume["pressure"] == "NORMAL":
                volume["pressure"] = "SOFT_PRESSURE"
        return volumes

    @classmethod
    @contextmanager
    def worker_lease(cls, requested_bytes: int) -> Iterator[None]:
        """Use the same authority in scheduler-owned native workers (no second runner)."""
        import json
        import sys
        import time

        import psutil

        from lambdaforge.controlplane.StorageOperations import StorageOperations

        raw = os.environ.get("LAMBDAFORGE_STORAGE_POLICY")
        if not raw or os.environ.get("LAMBDAFORGE_STORAGE_RESERVED") == "1":
            yield
            return
        descriptor = json.loads(raw)
        owner = ProcessIdentity.create(
            os.getpid(), os.getpgrp(), psutil.Process().cmdline(), os.environ["LAMBDAFORGE_JOB_ID"]
        )
        last_gc = float("-inf")
        previous_message = ""
        try:
            while True:
                observed = cls.reserve(descriptor, owner, requested_bytes)
                if observed.get("admitted"):
                    break
                message = (
                    f"Storage admission waiting: requested={requested_bytes}, "
                    f"free={observed['free_bytes']}, reserved={observed.get('reserved_bytes')}, "
                    f"safety={observed['safety_bytes']} bytes ({observed['reason']})."
                )
                if message != previous_message:
                    print(message, file=sys.stderr, flush=True)
                    previous_message = message
                if time.monotonic() - last_gc >= 60:
                    last_gc = time.monotonic()
                    cache = cls.filesystem(Path(descriptor["cache_root"]), descriptor)
                    deficit = max(
                        0,
                        requested_bytes
                        + int(observed.get("reserved_bytes", 0))
                        + observed["safety_bytes"]
                        - observed["free_bytes"],
                    )
                    try:
                        StorageOperations.gc(
                            {
                                **descriptor,
                                "automatic_gc": True,
                                "target_reclaim_bytes": deficit
                                if cache["device"] == observed["device"]
                                else 0,
                            },
                            {},
                            apply=True,
                        )
                    except (OSError, TimeoutError):
                        pass
                time.sleep(2)
            old = os.environ.get("LAMBDAFORGE_STORAGE_RESERVED")
            os.environ["LAMBDAFORGE_STORAGE_RESERVED"] = "1"
            try:
                yield
            finally:
                if old is None:
                    os.environ.pop("LAMBDAFORGE_STORAGE_RESERVED", None)
                else:
                    os.environ["LAMBDAFORGE_STORAGE_RESERVED"] = old
        finally:
            cls.release(descriptor, owner)

    @classmethod
    def reserve(
        cls, descriptor: Mapping[str, Any], owner: ProcessIdentity, requested_bytes: int
    ) -> dict[str, Any]:
        """Treat another admission transaction as a queue, never a failed scientific Job."""
        try:
            return cls._reserve_locked(descriptor, owner, requested_bytes)
        except TimeoutError:
            return {
                **cls.filesystem(Path(str(descriptor["run_root"])), descriptor),
                "admitted": False,
                "reason": "storage-lease-busy",
                "requested_bytes": requested_bytes,
            }

    @classmethod
    def _reserve_locked(
        cls, descriptor: Mapping[str, Any], owner: ProcessIdentity, requested_bytes: int
    ) -> dict[str, Any]:
        from lambdaforge.controlplane.StorageOperations import StorageOperations

        if (
            isinstance(requested_bytes, bool)
            or not isinstance(requested_bytes, int)
            or requested_bytes < 0
        ):
            raise ValueError("Storage commitment cannot be negative.")
        cls._validate_owner(owner)
        StorageOperations._roots(descriptor)
        run_root = Path(str(descriptor["run_root"]))
        observation = cls.filesystem(run_root, descriptor)
        root = cls._lease_root(descriptor) / str(observation["device"])
        if root.is_symlink():
            raise ValueError("Storage volume lease root cannot be symbolic.")
        root.mkdir(parents=True, exist_ok=True)
        with CrossProcessFileLock(
            root / ".lock", shared=False, timeout_seconds=10, poll_interval_seconds=0.05
        ):
            reserved = 0
            count = 0
            for path in root.glob("job-*.json"):
                record = StorageOperations._read_json(path)
                if not record:
                    # Unknown commitment is not permission to overcommit.
                    return {**observation, "admitted": False, "reason": "unresolved-storage-lease"}
                if record.get("host") != socket.gethostname():
                    return {**observation, "admitted": False, "reason": "unresolved-storage-host"}
                identity = record.get("owner")
                if not isinstance(identity, Mapping):
                    return {**observation, "admitted": False, "reason": "unresolved-storage-owner"}
                try:
                    prior = ProcessIdentity.from_mapping(identity)
                    cls._validate_owner(prior)
                    commitment = record["requested_bytes"]
                    if (
                        isinstance(commitment, bool)
                        or not isinstance(commitment, int)
                        or commitment < 0
                    ):
                        raise ValueError("Negative commitment")
                except (KeyError, TypeError, ValueError):
                    return {**observation, "admitted": False, "reason": "corrupt-storage-lease"}
                if not prior.matches():
                    # matches() returning false alone can mean an inaccessible process.
                    if cls._dead(prior.pid):
                        path.unlink()
                        continue
                    return {**observation, "admitted": False, "reason": "unresolved-storage-owner"}
                if prior.job_id != owner.job_id:
                    reserved += int(record["requested_bytes"])
                    count += 1
                elif prior.to_dict() != owner.to_dict():
                    return {**observation, "admitted": False, "reason": "storage-owner-conflict"}
            observation = cls.filesystem(run_root, descriptor)
            remaining = observation["free_bytes"] - reserved - requested_bytes
            admitted = remaining >= observation["safety_bytes"] and (
                not observation["inode_accounting"] or observation["free_inodes"] > 0
            )
            if admitted:
                atomic_json(
                    root / f"{owner.job_id}.json",
                    {
                        "storage_lease_version": 1,
                        "host": socket.gethostname(),
                        "owner": owner.to_dict(),
                        "requested_bytes": requested_bytes,
                        "device": observation["device"],
                        "run_root": str(run_root),
                        "purpose": descriptor.get("transaction_purpose", "job-workspace"),
                        "destination": descriptor.get("transaction_destination"),
                    },
                )
            return {
                **observation,
                "reserved_bytes": reserved,
                "reservations": count,
                "requested_bytes": requested_bytes,
                "admissible_bytes": max(0, remaining),
                "admitted": admitted,
                "reason": "admitted" if admitted else "storage-admission",
            }

    @classmethod
    def release(cls, descriptor: Mapping[str, Any], owner: ProcessIdentity) -> None:
        from lambdaforge.controlplane.StorageOperations import StorageOperations

        cls._validate_owner(owner)
        root = cls._lease_root(descriptor)
        for path in root.glob(f"*/{owner.job_id}.json"):
            if any(part.is_symlink() for part in (path, *path.parents)):
                continue
            with CrossProcessFileLock(
                path.parent / ".lock",
                shared=False,
                timeout_seconds=10,
                poll_interval_seconds=0.05,
            ):
                record = StorageOperations._read_json(path)
                if (
                    record
                    and record.get("host") == socket.gethostname()
                    and record.get("owner") == owner.to_dict()
                ):
                    path.unlink(missing_ok=True)

    @staticmethod
    def _lease_root(descriptor: Mapping[str, Any]) -> Path:
        root = (
            Path(str(descriptor.get("lease_root") or descriptor["state_root"])) / "storage-leases"
        )
        if any(part.is_symlink() for part in (root, *root.parents)):
            raise ValueError("Storage lease root cannot be symbolic.")
        return root

    @staticmethod
    def _validate_owner(owner: ProcessIdentity) -> None:
        import re

        if re.fullmatch(r"job-[a-z0-9-]+", owner.job_id) is None or owner.pid <= 0:
            raise ValueError("Invalid owned storage lease identity.")

    @staticmethod
    def _dead(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False
