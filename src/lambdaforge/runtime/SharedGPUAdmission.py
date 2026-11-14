"""Short host-wide check/start coordination, never a whole-GPU reservation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from lambdaforge.controlplane.ProcessIdentity import ProcessIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.runtime.HostCoordination import boot_identity, host_identity, secure_root
from lambdaforge.work.atomic import atomic_write_text

T = TypeVar("T")


def shared_gpu_launch(
    root: Path,
    token: str,
    launch: Callable[[], T | None],
    *,
    stagger_seconds: float,
    startup_bytes: int = 0,
    metadata: Mapping[str, object] | None = None,
    observed_pid_bytes: Mapping[int, int] | None = None,
    capacity_check: Callable[[int], bool] | None = None,
    startup_seconds: float = 60.0,
    fairness_wait_seconds: float = 30.0,
    waiter_ttl_seconds: float = 120.0,
    physical_free_bytes: int | None = None,
    on_wait: Callable[[str], None] | None = None,
) -> T | None:
    """Serialize a fresh physical check and launch; contention leaves the Run queued.

    All projects on a direct shared host use the same lease root and physical device token.
    Other GPUs remain independent. The OS releases the lock on death; the bounded last-start
    timestamp prevents sibling controllers racing while a new worker initializes CUDA.
    """
    if not token.startswith(("GPU-", "MIG-")):
        raise ValueError(
            "Shared GPU admission requires a stable physical/MIG UUID, not an ordinal."
        )
    if (
        startup_bytes < 0
        or not all(
            math.isfinite(value) and value > 0
            for value in (startup_seconds, fairness_wait_seconds, waiter_ttl_seconds)
        )
        or stagger_seconds < 0
    ):
        raise ValueError("Invalid GPU startup commitment/stagger.")
    root = secure_root(root)
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()
    lock = CrossProcessFileLock(
        root / f"admission-{key}.lock",
        shared=False,
        timeout_seconds=0.01,
        poll_interval_seconds=0.005,
    )
    try:
        lock.acquire()
    except TimeoutError:
        return None
    try:
        stamp = root / f"admission-{key}.json"
        if stamp.is_symlink():
            raise ValueError("Shared GPU admission metadata must not be a symlink.")
        value = {}
        if stamp.exists():
            if stamp.stat().st_size > 1024 * 1024:
                raise ValueError("GPU admission ledger is oversized.")
            value = json.loads(stamp.read_text(encoding="utf-8"))
        if value.get("boot") != boot_identity():
            value = {}
        controller = ProcessIdentity.create(
            os.getpid(),
            os.getpgrp() if os.name == "posix" else os.getpid(),
            (),
            str(os.environ.get("LAMBDAFORGE_JOB_ID", "controller")),
        )
        own_study = (metadata or {}).get("study")
        remaining, retained = _startup_remaining(
            value.get("commitments", []),
            controller,
            own_study,
            observed_pid_bytes or {},
        )
        ledger = {
            **value,
            "version": 3,
            "token": token,
            "boot": boot_identity(),
            "commitments": retained,
        }
        owner_key = _register_waiter(
            ledger, controller, metadata or {}, startup_bytes, ttl=waiter_ttl_seconds
        )
        reason = None
        previous = float(value.get("started_monotonic", float("-inf")))
        elapsed = time.monotonic() - previous
        if 0 <= elapsed < stagger_seconds:
            reason = "shared_startup_stagger"
        elif physical_free_bytes is not None and _fairness_peer(
            ledger["waiters"],
            owner_key,
            max(0, physical_free_bytes - remaining),
            now=time.time(),
            minimum_wait=fairness_wait_seconds,
        ):
            reason = "fairness_older_feasible_study"
        if reason:
            ledger["last_wait"] = {"reason": reason, "at": time.time()}
            atomic_write_text(stamp, json.dumps(ledger))
            if on_wait is not None:
                on_wait(reason)
            return None
        if capacity_check is not None and not capacity_check(remaining):
            ledger["last_wait"] = {"reason": "peer_startup_headroom", "at": time.time()}
            atomic_write_text(stamp, json.dumps(ledger))
            return None
        registered = json.loads(json.dumps(ledger))
        if startup_bytes:
            retained.append(
                {
                    "id": uuid4().hex,
                    "host": host_identity(),
                    "gpu_uuid": token,
                    "controller": controller.to_dict(),
                    "metadata": dict(metadata or {}),
                    "startup_bytes": startup_bytes,
                    "created_at": time.time(),
                    "expires_at": time.time() + startup_seconds,
                    "phase": "startup",
                    "observation": "pending-worker-identity",
                }
            )
        ledger["started_monotonic"] = time.monotonic()
        # Persist BEFORE creating a worker. A write error afterwards must never conceal an
        # already running Future from its native controller (or leave an untracked CUDA Run).
        atomic_write_text(stamp, json.dumps(ledger))
        try:
            result = launch()
        except BaseException:
            atomic_write_text(stamp, json.dumps(registered))
            raise
        if result is None:
            atomic_write_text(stamp, json.dumps(registered))
        else:
            waiter = ledger["waiters"][owner_key]
            waiter["started_at"] = time.time()
            waiter["waiting_since"] = None
            waiter["launches"] = int(waiter.get("launches", 0)) + 1
            cost = (metadata or {}).get("expected_seconds")
            waiter["served_seconds"] = float(waiter.get("served_seconds", 0)) + (
                float(cost)
                if isinstance(cost, (int, float)) and math.isfinite(cost) and cost > 0
                else 1.0
            )
            try:
                atomic_write_text(stamp, json.dumps(ledger))
            except OSError:
                # The before-spawn commitment is already durable. Losing an optional fairness
                # update cannot hide the returned Future or start another copy of this Run.
                if on_wait is not None:
                    on_wait("fairness_update_failed_commitment_preserved")
        return result
    finally:
        lock.release()


def _register_waiter(
    ledger: dict,
    owner: ProcessIdentity,
    metadata: Mapping[str, object],
    startup_bytes: int,
    *,
    ttl: float,
) -> str:
    """Bounded operational opportunities, not reservations of Trials or scientific actions."""
    now = time.time()
    waiters = ledger.get("waiters", {})
    if not isinstance(waiters, dict) or len(waiters) > 256:
        raise ValueError("Shared admission waiter inventory invalid/oversized.")
    waiters = {
        key: row
        for key, row in waiters.items()
        if 0 <= now - float(row["last_seen"]) <= ttl
        and ProcessIdentity.from_mapping(row["controller"]).storage_status()
        not in {"dead", "reused"}
    }
    key = hashlib.sha256(
        json.dumps([owner.pid, owner.create_time, metadata.get("study")]).encode()
    ).hexdigest()
    row = waiters.setdefault(
        key,
        {"controller": owner.to_dict(), "launches": 0, "served_seconds": 0, "waiting_since": now},
    )
    if row.get("waiting_since") is None:
        row["waiting_since"] = now
    priority = metadata.get("priority", 1.0)
    if (
        isinstance(priority, bool)
        or not isinstance(priority, (int, float))
        or not math.isfinite(priority)
        or priority <= 0
    ):
        raise ValueError("Shared resource priority must be finite and positive.")
    row.update(last_seen=now, required_bytes=startup_bytes, priority=priority)
    waiters[key] = row
    ledger["waiters"] = waiters
    return key


def _fairness_peer(
    waiters: Mapping[str, dict], own: str, headroom: int, *, now: float, minimum_wait: float
) -> bool:
    """Age/priority/service-weighted opportunity, never forced alternating or unsafe backfill."""

    def score(row: Mapping) -> float:
        since = row.get("waiting_since")
        if not isinstance(since, (int, float)):
            return 0.0
        launches = int(row.get("launches", 0))
        # Accumulated estimated service is only an operational tie-break/debt, not a comparison
        # of scientific objectives or an exact measurement of heterogeneous Run performance.
        return max(0.0, now - since) * float(row["priority"]) / (1 + launches)

    current = score(waiters[own])
    return any(
        key != own
        and row.get("waiting_since") is not None
        and now - float(row["waiting_since"]) >= minimum_wait
        and int(row["required_bytes"]) <= headroom
        and score(row) > current
        for key, row in waiters.items()
    )


def _startup_remaining(
    records: list[dict],
    controller: ProcessIdentity,
    study: object,
    observed: Mapping[int, int],
) -> tuple[int, list[dict]]:
    """Only peers' not-yet-observed startup bytes, never free memory minus resident bytes twice."""
    extra = 0
    retained = []
    absent_owners: dict[tuple[int, float, str, str], bool] = {}
    for record in records:
        owner = ProcessIdentity.from_mapping(record["controller"])
        metadata = record["metadata"]
        worker: ProcessIdentity | None = None
        raw_path = metadata.get("worker_identity_path")
        if raw_path:
            path = Path(raw_path)
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Unsafe shared GPU worker identity.")
            try:
                if path.stat().st_size > 65536:
                    raise ValueError("Oversized shared GPU worker identity.")
                raw = json.loads(path.read_text())
                if raw.get("controller_pid") != owner.pid:
                    raise ValueError("Shared GPU worker controller mismatch.")
                if (
                    raw.get("controller_create_time") is not None
                    and raw["controller_create_time"] != owner.create_time
                ):
                    raise ValueError("Shared GPU worker controller lifetime mismatch.")
                worker = ProcessIdentity(
                    int(raw["pid"]),
                    int(raw.get("process_group", raw["pid"])),
                    float(raw["create_time"]),
                    "",
                    owner.job_id,
                )
            except FileNotFoundError:
                pass  # Spawn has not yet published its exact worker identity.
        status = worker.storage_status() if worker else owner.storage_status()
        if worker is None and status in {"dead", "reused"}:
            key = (owner.pid, owner.create_time, owner.job_id, str(metadata.get("job_state_path")))
            if key not in absent_owners:
                absent_owners[key] = _job_absence_proven(metadata, owner)
            if absent_owners[key]:
                continue
        if worker is not None and status in {"dead", "reused"}:
            continue
        if worker is not None and status == "alive" and time.time() >= record["expires_at"]:
            continue  # Finite startup window, not a permanent reservation of this Run's floor.
        retained.append(record)
        if (owner.pid, owner.create_time, metadata.get("study")) == (
            controller.pid,
            controller.create_time,
            study,
        ):
            continue  # This controller's native future envelopes already account for its Runs.
        resident = observed.get(worker.pid, 0) if worker is not None else 0
        extra += max(0, int(record["startup_bytes"]) - resident)
    return extra, retained


def _job_absence_proven(metadata: Mapping[str, object], owner: ProcessIdentity) -> bool:
    """Reconcile an orphan through native ownership, never age, missing PID or htop alone."""
    raw_path = metadata.get("job_state_path")
    if not isinstance(raw_path, str):
        return False
    path = Path(raw_path)
    if path.name != "state.json" or path.parent.name != owner.job_id or path.resolve() != path:
        return False
    try:
        from lambdaforge.controlplane.ProcessSupervisor import ProcessSupervisor

        try:
            path.parent.stat()
        except FileNotFoundError:
            # Deleting a finished Job also removes its worker identity. Do not turn its
            # conservative cold-start envelope into a permanent whole-GPU reservation.
            # Missing files/age alone are NOT proof: verify the parent ownership and use
            # the same native per-Job process-marker/lifetime scope as normal observation.
            parent = path.parent.parent.stat()
            if os.name == "posix" and parent.st_uid != os.getuid():
                return False
            return ProcessSupervisor.detached_owner_absent(owner)
        for name in ("state.json", "request.json"):
            if (path.parent / name).stat().st_size > 2 * 1024 * 1024:
                return False
        observation = ProcessSupervisor.observe(path.parent)
        return observation.get("provider_observation") in {"terminal", "lost_confirmed"}
    except (OSError, ValueError, TypeError, KeyError):
        return False
