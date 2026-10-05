"""Durable execution-only member allocation, owned by the ordinary provider Job.

The provider owns GPU grants and release; native ARI owns admission and isolated Runs.
This inbox contains only concrete leased shards, never an optimizer or a seed generator.
The baseline authority executes one finite wave at a time, retaining the provider allocation
between waves. It deliberately advertises no spare capacity during a running wave.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.Fleet import FleetMember
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.StudyCoordinator import StudyShard
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.shard import execute_concrete_shard, verify_provider_gpu_grant

ENVELOPE_LIMIT = 8 * 1024**2


def read_envelope(path: Path) -> dict[str, Any]:
    """Read a bounded regular owner file; no symlink or permissive version fallback."""
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("Member metadata must be a regular owned file.")
    if path.stat().st_size > ENVELOPE_LIMIT:
        raise ValueError("Member metadata exceeds the bounded control-envelope limit.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Member metadata must be a JSON object.")
    return value


def allocation_manifest(root: Path) -> dict[str, Any]:
    value = read_envelope(root / "allocation.json")
    if value.get("allocation_version") != 1 or not re.fullmatch(
        r"job-fleet-allocation-[a-f0-9]{64}", str(value.get("job_id", ""))
    ):
        raise ValueError("Unsupported or unidentified member allocation.")
    if value.get("root") != str(root.absolute()):
        raise ValueError("Member allocation belongs to another owned root.")
    return value


def verified_equivalence(root: Path, allocation: Mapping[str, Any]) -> ExecutionEquivalence:
    expected = ExecutionEquivalence(**allocation["equivalence"])
    if not allocation.get("discover_hardware"):
        return expected
    attestation = read_envelope(root / "attestation.json")
    actual = ExecutionEquivalence(**attestation["equivalence"])
    if (
        attestation.get("job_id") != allocation["job_id"]
        or any(
            getattr(actual, name) != getattr(expected, name)
            for name in ("code", "environment", "inputs", "numerics")
        )
        or actual.hardware == "unattested"
    ):
        raise ValueError("Member attestation differs from its exact preparation identities.")
    return actual


def enqueue(root: Path, manifest: Mapping[str, Any]) -> str:
    """Idempotently publish an immutable wave on its owner before returning its ACK."""
    allocation = allocation_manifest(root)
    equivalence = verified_equivalence(root, allocation)
    shard = StudyShard.from_mapping(manifest["shard"])
    if (
        manifest.get("allocation_version") != 1
        or manifest.get("job_id") != allocation["job_id"]
        or shard.cluster != allocation["member"]["cluster"]
        or shard.study_identity != allocation["study_identity"]
        or shard.member.to_dict() != allocation["member"]
        or manifest.get("equivalence") != equivalence.to_dict()
        or set(manifest["invocations"]) != {lease.run.key for lease in shard.leases}
        or not 1 <= len(shard.leases) <= 64
        or any(lease.attempt != 1 for lease in shard.leases)
    ):
        raise ValueError("Member inbox requires exact fresh allocation/Study/lease identities.")
    if not re.fullmatch(r"[a-f0-9]{64}", shard.shard_id):
        raise ValueError("Member inbox requires an exact shard digest.")
    if len(json.dumps(manifest, allow_nan=False).encode("utf-8")) > ENVELOPE_LIMIT:
        raise ValueError("Member wave exceeds the bounded control-envelope limit.")
    from lambdaforge.work.config import WorkConfig
    from lambdaforge.work.shard import prepare_concrete_shard

    config = WorkConfig.from_yaml(allocation["source"])
    class_name = config.levels[0].runs[0].work_class
    values = {
        key: {**value, "source": allocation["source"]}
        for key, value in manifest["invocations"].items()
    }
    if any(value["definition"]["work_class"] != class_name for value in values.values()):
        raise ValueError("Member invocations differ from the prepared Work class.")
    prepare_concrete_shard(
        shard,
        values,
        root=root / "waves" / shard.shard_id / "worker",
        resources=ResourceRequest.from_mapping(allocation["resources"]),
        verified_equivalence=equivalence,
        parallelism=len(shard.leases),
        input_bindings=allocation["input_bindings"],
    )
    with CrossProcessFileLock(
        root / ".inbox.lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
    ):
        if (root / "shutdown.json").exists():
            raise ValueError("Member allocation is draining; no new wave may be accepted.")
        path = root / "inbox" / (shard.shard_id + ".json")
        incoming = {lease.run.key for lease in shard.leases}
        for previous in (root / "inbox").glob("*.json"):
            if previous == path:
                continue
            other = read_envelope(previous)
            if incoming & set(other["invocations"]):
                raise ValueError("A logical Run cannot acquire a second member wave.")
        if path.exists():
            if read_envelope(path) != dict(manifest):
                raise ValueError("A member wave cannot change after acceptance.")
        else:
            atomic_write_json(path, dict(manifest))
    return allocation["job_id"]


def observe(root: Path, *, shard_id: str | None = None) -> dict[str, Any]:
    allocation = allocation_manifest(root)
    offer_path = root / "offer.json"
    offer = read_envelope(offer_path) if offer_path.exists() else None
    payload: dict[str, Any] = {"job_id": allocation["job_id"], "offer": offer}
    if shard_id is not None:
        if not re.fullmatch(r"[a-f0-9]{64}", shard_id):
            raise ValueError("Observation needs an exact shard digest.")
        from lambdaforge.work.shard import observe_worker

        wave_path = root / "inbox" / (shard_id + ".json")
        wave = read_envelope(wave_path) if wave_path.exists() else None
        payload["accepted"] = wave is not None
        payload["worker_root"] = str(root / "waves" / shard_id / "worker")
        if wave is not None:
            # Existing bounded result projection, not a second evidence store.
            import contextlib
            import io

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                observe_worker(Path(payload["worker_root"]), list(wave["invocations"]))
            payload["observation"] = json.loads(stream.getvalue())
    return payload


def scientific_stream(root: Path, shard_id: str, cursors: Mapping[str, Any]) -> dict[str, Any]:
    """Read native scalar records only from the exact accepted allocation wave."""
    from lambdaforge.work.fleet_stream import CHANNELS, read_chunk, regular_path
    from lambdaforge.work.study import StudyTelemetry, study_run_key

    owner = allocation_manifest(root)
    if not re.fullmatch(r"[a-f0-9]{64}", shard_id):
        raise ValueError("Scalar read requires an exact accepted shard digest.")
    wave = read_envelope(root / "inbox" / (shard_id + ".json"))
    shard = StudyShard.from_mapping(wave["shard"])
    if not 1 <= len(cursors) <= 64 or set(cursors) - {lease.run.key for lease in shard.leases}:
        raise ValueError("Scalar cursors must name exact leased Runs (at most 64).")
    worker = root / "waves" / shard_id / "worker"
    state_path = worker / "worker.json"
    state = read_envelope(state_path) if state_path.exists() else {}
    if state and state.get("manifest", {}).get("shard") != wave["shard"]:
        raise ValueError("Native scalar owner differs from its accepted wave.")
    invocations = state.get("manifest", {}).get("invocations", ())
    native = {(value["trial_index"], value.get("seed")): value for value in invocations}
    telemetry = StudyTelemetry(worker / "study")
    streams = {}
    for lease in shard.leases:
        if lease.run.key not in cursors:
            continue
        value = native.get((int(lease.run.candidate), lease.run.seed))
        run = telemetry._run_state(study_run_key(value)) if value is not None else {}
        if run and (
            run.get("attempt_id") != f"attempt-{lease.attempt:04d}"
            or run.get("trial") != int(lease.run.candidate)
            or run.get("seed") != lease.run.seed
            or run.get("phase") != lease.run.phase
        ):
            raise ValueError("Scalar telemetry belongs to another scientific Attempt.")
        terminal = lease.run.key in state.get("results", {})
        channels = {}
        for channel, field in zip(CHANNELS, ("metrics_path", "training_metrics_path"), strict=True):
            if set(cursors[lease.run.key]) != set(CHANNELS):
                raise ValueError("Unknown scalar cursor channels.")
            if run and run.get(field):
                path = Path(run[field])
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or not path.is_relative_to(worker / "execution")
                ):
                    raise ValueError("Scalar metadata contains a foreign owner path.")
            elif terminal:
                # A failed import can finish before publishing native telemetry. The result
                # still owns its Attempt paths; never turn absent metadata into lost evidence.
                path = Path(state["results"][lease.run.key]["result"]["run_dir"]) / (
                    channel + ".jsonl"
                )
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or not path.is_relative_to(worker / "execution")
                ):
                    raise ValueError("Terminal scalar path belongs to another owner.")
            else:
                path = (
                    worker
                    / "unstarted"
                    / lease.run.key.removeprefix("sha256:")
                    / (channel + ".jsonl")
                )
            regular_path(path)
            channels[channel] = read_chunk(path, cursors[lease.run.key][channel], terminal=terminal)
        streams[lease.run.key] = {
            "stream_version": 1,
            "identity": {
                "study_identity": shard.study_identity,
                "run_key": lease.run.key,
                "attempt": lease.attempt,
                "lease_id": lease.lease_id,
                "cluster": shard.cluster,
                "shard_id": shard_id,
            },
            "channels": channels,
        }
    return {"job_id": owner["job_id"], "streams": streams}


def request_pruning(root: Path, shard_id: str, command: Mapping[str, Any]) -> dict[str, Any]:
    """Publish an exact central stop request, without declaring a Run pruned or killing it."""
    from lambdaforge.work.atomic import atomic_write_bytes
    from lambdaforge.work.fleet_stream import regular_path
    from lambdaforge.work.runner import _performance_pruning_protected
    from lambdaforge.work.study import StudyTelemetry, study_run_key

    owner = allocation_manifest(root)
    if not re.fullmatch(r"[a-f0-9]{64}", shard_id):
        raise ValueError("Pruning requires an accepted shard digest.")
    wave = read_envelope(root / "inbox" / (shard_id + ".json"))
    shard = StudyShard.from_mapping(wave["shard"])
    lease = next((value for value in shard.leases if value.run.key == command.get("run_key")), None)
    if lease is None or command.get("identity") != {
        "study_identity": shard.study_identity,
        "run_key": lease.run.key,
        "attempt": lease.attempt,
        "lease_id": lease.lease_id,
        "cluster": shard.cluster,
        "shard_id": shard_id,
    }:
        raise ValueError("Pruning command does not own the exact live lease.")
    value = wave["invocations"][lease.run.key]
    if _performance_pruning_protected(value):
        raise ValueError("Required, startup, confirmation and continuation evidence is protected.")
    reason, evidence = command.get("reason"), command.get("evidence")
    if (
        not isinstance(reason, str)
        or not reason
        or len(reason) > 4096
        or not isinstance(evidence, Mapping)
    ):
        raise ValueError("Pruning requires bounded central scientific evidence.")
    worker = root / "waves" / shard_id / "worker"
    state = read_envelope(worker / "worker.json")
    if state.get("manifest", {}).get("shard") != wave["shard"]:
        raise ValueError("Pruning owner differs from its accepted wave.")
    requests = worker / "pruning" / (lease.run.key.removeprefix("sha256:") + ".json")
    regular_path(requests)
    with CrossProcessFileLock(
        worker / ".pruning.lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
    ):
        if requests.exists() and read_envelope(requests) != dict(command):
            raise ValueError("A pruning request cannot change after publication.")
        if lease.run.key in state.get("results", {}):
            return {"job_id": owner["job_id"], "status": "already-terminal"}
        run = StudyTelemetry(worker / "study")._run_state(study_run_key(value))
        if not run or run.get("state") != "running":
            return {"job_id": owner["job_id"], "status": "not-running"}
        if (
            run.get("attempt_id") != f"attempt-{lease.attempt:04d}"
            or run.get("trial") != int(lease.run.candidate)
            or run.get("seed") != lease.run.seed
            or run.get("phase") != lease.run.phase
        ):
            raise ValueError("Pruning telemetry belongs to another scientific Attempt.")
        seed_token = value.get("seed") if value.get("seed") is not None else "none"
        token = f"trial-{value['trial_index']:05d}-seed-{seed_token}"
        stop = worker / "execution" / "hpo-control" / (token + ".stop")
        regular_path(stop)
        regular_path(stop.with_name(stop.name + ".evidence.json"))
        atomic_write_json(requests, dict(command))
        atomic_write_json(stop.with_name(stop.name + ".evidence.json"), dict(evidence))
        atomic_write_bytes(stop, reason.encode("utf-8"))
    return {"job_id": owner["job_id"], "status": "requested"}


def run_member(root: Path) -> None:
    """Hold one real provider allocation across waves; never restart an unknown owner."""
    allocation = allocation_manifest(root)
    if os.environ.get("LAMBDAFORGE_JOB_ID") != allocation["job_id"]:
        raise ValueError("Member must run inside its exact provider Job.")
    resources = ResourceRequest.from_mapping(allocation["resources"])
    member = FleetMember.from_mapping(allocation["member"])
    equivalence = ExecutionEquivalence(**allocation["equivalence"])
    if allocation.get("discover_hardware"):
        from dataclasses import replace

        from lambdaforge.work.shard import attest_provider_hardware

        tokens, hardware = attest_provider_hardware(allocation["job_id"], member, resources)
        equivalence = replace(equivalence, hardware=hardware)
    else:
        tokens = verify_provider_gpu_grant(allocation["job_id"], member, resources, equivalence)
    capacity = min(resources.cpu_cores, len(tokens) if tokens else resources.cpu_cores)
    capacity = min(capacity, member.max_runs or capacity)
    with CrossProcessFileLock(
        root / ".owner.lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
    ):
        state_path = root / "member.json"
        if state_path.exists():
            raise RuntimeError("A prior member owner must be reconciled, not silently restarted.")
        if allocation.get("discover_hardware"):
            atomic_write_json(
                root / "attestation.json",
                {"job_id": allocation["job_id"], "equivalence": equivalence.to_dict()},
            )
        atomic_write_json(state_path, {"state": "running", "job_id": allocation["job_id"]})
        busy = threading.Event()
        stopping = threading.Event()
        finished: set[str] = set()
        acknowledged: set[str] = set()
        mutex = threading.Lock()

        def heartbeat() -> None:
            while not stopping.is_set():
                now = time.time()
                slots, granted, free = capacity, len(tokens), None
                reason = "admissible"
                try:
                    if resources.gpu_count:
                        from lambdaforge.work.runner import (
                            _current_gpu_grant,
                            _gpu_memory_inventory,
                        )

                        current = (
                            _current_gpu_grant(tokens)
                            if os.environ.get("LAMBDAFORGE_GPU_VISIBILITY_COMMAND")
                            else frozenset(tokens)
                        )
                        # Intersection only: never accept tokens newly reported by a site.
                        if current is None:
                            slots, granted, reason = 0, 0, "grant-unavailable"
                        else:
                            memory = _gpu_memory_inventory(len(tokens))
                            usable = [
                                value[0]
                                for token, value in zip(tokens, memory, strict=True)
                                if token in current and value[0] >= resources.gpu_memory_bytes
                            ]
                            granted = len(usable)
                            slots = min(capacity, granted)
                            free = min(usable) if usable else 0
                            if not usable:
                                reason = "memory-headroom-or-grant"
                    if busy.is_set():
                        slots, reason = 0, "baseline-wave-active"
                    if (root / "shutdown.json").exists():
                        slots, reason = 0, "member-draining"
                except (RuntimeError, ValueError, OSError):
                    slots, granted, free, reason = 0, 0, None, "admission-probe-unavailable"
                with mutex:
                    leases = sorted(acknowledged)
                atomic_write_json(
                    root / "offer.json",
                    {
                        "offer_version": 1,
                        "job_id": allocation["job_id"],
                        "equivalence": equivalence.to_dict(),
                        "observed_at": now,
                        "valid_until": now + 15,
                        "slots": slots,
                        "admissible_gpus": granted,
                        "free_memory_bytes": free,
                        "acknowledged_leases": leases,
                        "reason": reason,
                    },
                )
                stopping.wait(3)

        thread = threading.Thread(target=heartbeat, name="member-heartbeat", daemon=True)
        thread.start()
        try:
            while True:
                waves = sorted((root / "inbox").glob("*.json"))
                for path in waves:
                    if path.stem in finished:
                        continue
                    wave = read_envelope(path)
                    shard = StudyShard.from_mapping(wave["shard"])
                    busy.set()
                    with mutex:
                        acknowledged.update(lease.lease_id for lease in shard.leases)
                    values = {
                        key: {**value, "source": allocation["source"]}
                        for key, value in wave["invocations"].items()
                    }
                    # Allocation ownership is already verified. Shard ownership remains exact;
                    # this explicit binding is used instead of overwriting the inherited Job ID.
                    execute_concrete_shard(
                        shard,
                        values,
                        root=root / "waves" / shard.shard_id / "worker",
                        resources=resources,
                        verified_equivalence=equivalence,
                        parallelism=min(capacity, len(shard.leases)),
                        input_bindings=allocation["input_bindings"],
                        provider_job_id=allocation["job_id"],
                    )
                    finished.add(path.stem)
                    with mutex:
                        acknowledged.clear()
                    busy.clear()
                if (root / "shutdown.json").exists():
                    break
                time.sleep(0.2)
        finally:
            stopping.set()
            thread.join(timeout=5)
            atomic_write_json(root / "offer.json", {"offer_version": 1, "slots": 0})
        atomic_write_json(state_path, {"state": "completed", "job_id": allocation["job_id"]})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--observe", action="store_true")
    action.add_argument("--enqueue", type=Path)
    action.add_argument("--shutdown", action="store_true")
    action.add_argument("--stream")
    action.add_argument("--prune")
    parser.add_argument("--shard")
    args = parser.parse_args(argv)
    root = args.root.absolute()
    if args.stream is not None or args.prune is not None:
        if args.shard is None:
            raise ValueError("Scientific stream/command requires its exact shard.")
        raw = args.stream if args.stream is not None else args.prune
        if len(raw.encode("utf-8")) > 64 * 1024:
            raise ValueError("Scientific request exceeds its bounded envelope.")
        value = json.loads(raw)
        payload = (
            scientific_stream(root, args.shard, value)
            if args.stream is not None
            else request_pruning(root, args.shard, value)
        )
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode("utf-8")) > ENVELOPE_LIMIT:
            raise ValueError("Scientific stream exceeds its bounded envelope.")
        print(encoded)
    elif args.observe:
        payload = observe(root, shard_id=args.shard)
        raw = json.dumps(payload, allow_nan=False)
        if len(raw.encode("utf-8")) > ENVELOPE_LIMIT:
            raise ValueError("Member observation exceeds its bounded envelope limit.")
        print(raw)
    elif args.enqueue:
        try:
            job_id = enqueue(root, read_envelope(args.enqueue))
        except (ValueError, TypeError) as error:
            owner = allocation_manifest(root)
            print(json.dumps({"job_id": owner["job_id"], "rejected": str(error)}))
        else:
            print(json.dumps({"job_id": job_id}))
    elif args.shutdown:
        owner = allocation_manifest(root)
        with CrossProcessFileLock(
            root / ".inbox.lock", shared=False, timeout_seconds=5, poll_interval_seconds=0.05
        ):
            atomic_write_json(root / "shutdown.json", {"job_id": owner["job_id"]})
    else:
        run_member(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
