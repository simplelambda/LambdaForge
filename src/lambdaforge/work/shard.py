"""Concrete CPU shard execution through the existing isolated Work dispatcher.

This is an internal execution boundary, not a second Study runner or a public launch route.
It accepts already prepared invocations and executor-verified equivalence. Production bundle,
environment and provider preparation must precede this boundary. GPU shards deliberately fail
closed until granted-executor integration and global/native Attempt recovery are connected.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.StudyCoordinator import StudyShard
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.hpo.AdaptiveSearch import AdaptiveSearchPolicy
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json
from lambdaforge.work.config import import_work_class
from lambdaforge.work.models import WorkResult
from lambdaforge.work.runner import WorkRunner, _execute_adaptive_dispatch


def prepare_concrete_shard(
    shard: StudyShard,
    invocations: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
    resources: ResourceRequest,
    verified_equivalence: ExecutionEquivalence,
    parallelism: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int | None], str]]:
    """Validate a finite exact CPU queue without creating state or starting a child.

    ``invocations`` is keyed by global Run key and uses the native dispatcher specification.
    The caller is responsible for verified environment/bundle/input preparation. All invocations
    are validated before any child starts. No HPO/planner callback, seed stream or global state
    is available to the worker. A interrupted owner is not automatically restarted: its live
    children must first be reconciled by the provider authority.
    """
    if resources.gpu_count:
        raise ValueError("GPU shards require granted provider/ARI integration; not enabled yet.")
    if (
        isinstance(parallelism, bool)
        or not isinstance(parallelism, int)
        or not 1 <= parallelism <= resources.cpu_cores
    ):
        raise ValueError("CPU shard parallelism must fit its prepared host allocation.")
    if shard.member.max_runs is not None and parallelism > shard.member.max_runs:
        raise ValueError("Shard parallelism exceeds the Fleet member Run cap.")
    if set(invocations) != {lease.run.key for lease in shard.leases}:
        raise ValueError("Shard invocations must match every exact leased Run, without additions.")
    root = root.absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Owned shard state cannot live below a symlink.")
    prepared = []
    identities: dict[tuple[int, int | None], str] = {}
    execution_ids = set()
    for lease in shard.leases:
        if lease.run.requires_gpu or lease.run.equivalence != verified_equivalence:
            raise ValueError("Shard invocation does not match its verified CPU execution stratum.")
        if lease.attempt != 1:
            raise ValueError(
                "Recovered shards require global/native Attempt binding; not enabled yet."
            )
        value = json.loads(json.dumps(dict(invocations[lease.run.key]), allow_nan=False))
        allowed = {
            "definition",
            "parameters",
            "trial_parameters",
            "seed",
            "seed_metadata",
            "trial_index",
            "execution_id",
            "execution_dir",
            "source",
            "restart",
            "hpo_phase",
            "hpo_fidelity",
            "evidence_required",
            "evidence_requirement",
            "candidate_pool_index",
            "hpo_startup_anchor",
            "hpo_probe_purpose",
            "hpo_target_questions",
            "study_recovery",
            "sweep_block_ordinal",
            "sweep_lookahead",
            "predicted_duration_seconds",
        }
        if set(value) - allowed:
            raise ValueError("Concrete shard invocations cannot inject operational runtime fields.")
        if (
            value.get("parameters") != dict(lease.run.parameters)
            or value.get("seed") != lease.run.seed
            or value.get("hpo_phase", "search") != lease.run.phase
            or value.get("hpo_fidelity", {}) != dict(lease.run.fidelity)
        ):
            raise ValueError("Native invocation differs from its exact scientific lease.")
        if (
            value.get("restart") is not False
            or value.get("study_recovery")
            or any(
                name in value
                for name in (
                    "gpu_slot",
                    "gpu_index",
                    "gpu_semaphore",
                    "recovery_checkpoint_root",
                    "hpo_controller_restart",
                    "hpo_scientific_continuation",
                )
            )
        ):
            raise ValueError("A fresh shard cannot inject devices, restart or recovery authority.")
        if lease.run.fidelity.get("current", 0) != 0:
            raise ValueError("Fidelity continuation requires a prepared owned checkpoint binding.")
        source = Path(value["source"])
        if not source.is_absolute() or source.is_symlink() or not source.is_file():
            raise ValueError("Shard source must be a prepared regular absolute YAML path.")
        WorkRunner._verify_shared_bundle_inputs(source)
        definition = value["definition"]
        if ResourceRequest.from_mapping(definition["resources"]).gpu_count:
            raise ValueError("A CPU shard cannot contain a GPU invocation.")
        target = import_work_class(definition["work_class"])
        inspect.signature(target.run).bind(None, **value["parameters"])
        trial = value.get("trial_index")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
            raise ValueError("Shard invocation needs an exact native Trial index.")
        identity = (trial, lease.run.seed)
        if identity in identities:
            raise ValueError("One shard cannot run competing phases of a native Run concurrently.")
        identities[identity] = lease.run.key
        if not isinstance(value.get("execution_id"), str) or not value["execution_id"]:
            raise ValueError("Shard requires its original native Execution identity.")
        execution_ids.add(value["execution_id"])
        value["execution_dir"] = str(root / "execution")
        value["definition"] = {**definition, "has_variants": True, "study_expected": False}
        prepared.append(value)
    if len(execution_ids) != 1:
        raise ValueError("A shard cannot combine native Execution identities.")
    objective = prepared[0]["definition"].get("objective") or {}
    if any((value["definition"].get("objective") or {}) != objective for value in prepared):
        raise ValueError("Shard invocations cannot reinterpret the shared scientific objective.")
    return prepared, identities


def execute_concrete_shard(
    shard: StudyShard,
    invocations: Mapping[str, Mapping[str, Any]],
    *,
    root: Path,
    resources: ResourceRequest,
    verified_equivalence: ExecutionEquivalence,
    parallelism: int,
) -> tuple[dict[str, Any], ...]:
    """Execute validated exact CPU leases through native isolated Runs, with no planner.

    Fresh results persist incrementally. A restarted completed worker replays immutable outcomes;
    an interrupted owner requires provider reconciliation and must never be silently restarted.
    """
    root = root.absolute()
    prepared, identities = prepare_concrete_shard(
        shard,
        invocations,
        root=root,
        resources=resources,
        verified_equivalence=verified_equivalence,
        parallelism=parallelism,
    )
    objective = prepared[0]["definition"].get("objective") or {}
    manifest = {"worker_version": 1, "shard": shard.to_dict(), "invocations": prepared}
    lock = CrossProcessFileLock(
        root / ".owner.lock", shared=False, timeout_seconds=5.0, poll_interval_seconds=0.05
    )
    with lock:
        state_path = root / "worker.json"
        if state_path.is_symlink():
            raise ValueError("Owned shard worker state is symlinked.")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("manifest") != manifest:
                raise ValueError("Owned shard manifest is immutable.")
            if state.get("state") != "completed":
                raise RuntimeError("Interrupted shard ownership must be reconciled, not rerun.")
            return tuple(state["results"][lease.run.key] for lease in shard.leases)
        if any(path.name != ".owner.lock" for path in root.iterdir()):
            raise ValueError("Shard state is missing from a nonempty owned root; refusing reset.")
        state = {"manifest": manifest, "state": "running", "results": {}}
        atomic_write_json(state_path, state)
        leases = {lease.run.key: lease for lease in shard.leases}

        def persist_result(
            result: WorkResult,
            queued: Sequence[Mapping[str, Any]],
            pending: Sequence[Mapping[str, Any]],
        ) -> Sequence[dict[str, Any]]:
            # Returning exactly queued preserves the native dispatcher's remaining queue;
            # it is not scientific refill and must never manufacture a new invocation.
            del pending
            if result.trial is None:
                raise ValueError("Native shard result omitted its Trial identity.")
            key = identities[(int(result.trial["index"]), result.seed)]
            lease = leases[key]
            if result.attempt_number != 1:
                raise ValueError("Unbound local retry cannot silently become global evidence.")
            envelope = {
                "study_identity": shard.study_identity,
                "run_key": key,
                "attempt": lease.attempt,
                "lease_id": lease.lease_id,
                "cluster": shard.cluster,
                "parameters": dict(lease.run.parameters),
                "equivalence": verified_equivalence.to_dict(),
                "state": "pruned" if result.pruned else "completed" if result.ok else result.status,
                "result": result.to_dict(),
            }
            state["results"][key] = envelope
            atomic_write_json(state_path, state)
            return tuple(dict(value) for value in queued)

        _execute_adaptive_dispatch(
            prepared,
            resources=resources,
            policy=AdaptiveSearchPolicy(
                max_parallel=parallelism, early_stopping=False, failure_retries=0
            ),
            objective_metric=str(objective.get("metric", "")),
            objective_mode=str(objective.get("mode", "max")),
            objective=objective,
            historical_results=(),
            parallelism=parallelism,
            telemetry=None,
            on_result=persist_result,
        )
        if set(state["results"]) != set(leases):
            raise RuntimeError("Shard ended without an outcome for every concrete leased Run.")
        state["state"] = "completed"
        atomic_write_json(state_path, state)
        return tuple(state["results"][lease.run.key] for lease in shard.leases)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute an owned prepared manifest under the existing provider supervisor."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args(argv)
    path = arguments.manifest.absolute()
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("Shard manifest must be an owned regular file.")
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Prepared shard manifest exceeds the bounded control-envelope limit.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_version") != 1:
        raise ValueError("Unsupported prepared shard manifest.")
    if value.get("root") != str(path.parent / "worker"):
        raise ValueError("Prepared worker root must belong to the exact provider manifest.")
    execute_concrete_shard(
        StudyShard.from_mapping(value["shard"]),
        value["invocations"],
        root=Path(value["root"]),
        resources=ResourceRequest.from_mapping(value["resources"]),
        verified_equivalence=ExecutionEquivalence(**value["equivalence"]),
        parallelism=value["parallelism"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
