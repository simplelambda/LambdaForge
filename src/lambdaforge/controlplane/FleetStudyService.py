"""Public fresh adaptive/fixed Fleet execution using one native WorkRunner and provider owners.

No adaptive optimizer lives here. Unsupported scientific protocols fail before any allocation.
The durable coordinator currently runs locally; each member is prepared through ControlPlane.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ControlPlane import ControlPlane
from lambdaforge.controlplane.Fleet import ClusterHealth, Fleet
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor
from lambdaforge.controlplane.StudyCoordinator import StudyCoordinator
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.coordinated_dispatch import CoordinatedDispatcher


def fleet_preflight(config: WorkConfig, catalog: ClusterCatalog, fleet: Fleet) -> dict[str, Any]:
    """Read-only structural preflight; readiness is never inferred from physical observations."""
    if fleet.coordinator != "local":
        raise ValueError("Fleet coordinator must currently be local; remote drivers are pending.")
    if len(config.levels) != 1 or len(config.levels[0].runs) != 1:
        raise ValueError("Fleet execution requires one Study, not a composed Work.")
    definition = config.levels[0].runs[0]
    if not definition.study_expected:
        raise ValueError("Public Fleet execution requires a parameter Study or repeated seeds.")
    if definition.search_policy is not None and definition.search_policy.fidelity is not None:
        raise ValueError("Fleet fidelity requires owned checkpoint continuation; not enabled yet.")
    if config.source is None:
        raise ValueError("Fleet execution requires an authored YAML source.")

    def contains_dataset(value: Any) -> bool:
        if isinstance(value, Mapping):
            return set(value) == {"dataset"} or any(
                contains_dataset(child) for child in value.values()
            )
        return isinstance(value, list | tuple) and any(contains_dataset(child) for child in value)

    if contains_dataset(definition.parameters):
        raise ValueError("Distributed dataset placement attestation is not integrated yet.")
    report = WorkConfig.validate_file(config.source)
    if not report.valid:
        raise ValueError("Invalid Fleet Work: " + "; ".join(report.errors))
    eligible = []
    excluded = []
    for member in fleet.members:
        profile = catalog.get(member.cluster)
        reasons = []
        if member.state is not ClusterHealth.ONLINE:
            reasons.append(member.state.value)
        if profile.environment != "managed" or member.cluster == "local":
            reasons.append("immutable-managed-member-required")
        if profile.auth.mode == "password" and profile.auth.credential is None:
            reasons.append("detached-password-reference-required")
        if reasons:
            if member.required:
                raise ValueError(f"Required member {member.cluster}: {', '.join(reasons)}")
            excluded.append({"cluster": member.cluster, "reasons": reasons})
        else:
            eligible.append(member.to_dict())
    if not eligible:
        raise ValueError("Fleet has no eligible prepared member.")
    return {
        "fleet_preflight_version": 1,
        "fleet": fleet.name,
        "coordinator": fleet.coordinator,
        "study": config.preflight(),
        "members": eligible,
        "excluded": excluded,
        "environment_readiness": "verified during preparation",
        "execution_equivalence": "attested inside each real allocation before any Run is leased",
        "gpu_grant": "not acquired by dry-run",
        "dispatch": False,
    }


def execute_fleet(
    source: Path,
    fleet: Fleet,
    plane: ControlPlane,
    *,
    parent_job_id: str,
    rerun: bool = False,
) -> dict[str, Any]:
    """Prepare member owners, verify one stratum, then run the original planner once."""
    config = WorkConfig.from_yaml(source)
    preview = fleet_preflight(config, plane.catalog, fleet)
    definition = config.levels[0].runs[0]
    root = plane.jobs.store.root / "fleets" / parent_job_id
    execution_root = root / "execution"
    plan = WorkRunner(execution_root=execution_root).plan(config, rerun=rerun)
    # Rerun chooses an identity once, then every native planner call must use that same plan.
    if rerun:
        raise ValueError("Fleet --rerun is pending durable native plan binding.")
    coordinator = StudyCoordinator(root / "coordinator")
    eligible = {item["cluster"] for item in preview["members"]}
    selected = replace(fleet, members=tuple(m for m in fleet.members if m.cluster in eligible))
    coordinator.initialize(
        plan.scientific_fingerprint,
        plan.execution_id,
        selected,
        max_runs=definition.execution_policy.max_runs,
        max_time_seconds=definition.execution_policy.max_time_seconds,
    )
    executors: dict[str, PreparedShardExecutor] = {}
    dispatcher: CoordinatedDispatcher

    def invocation(key: str) -> Mapping[str, Any]:
        return dispatcher.invocation(key)

    try:
        # Unattested fields are preparation requests only. They NEVER enter GlobalRun/placement.
        # Register each owner before preparation so partial failure also drains earlier members.
        for member in selected.members:
            resources = config.resources
            if resources.gpu_count and member.max_gpus is not None:
                resources = replace(resources, gpu_count=min(resources.gpu_count, member.max_gpus))
            identity = ScientificIdentity.from_payload(
                {
                    "parent_job": parent_job_id,
                    "execution": plan.execution_id,
                    "member": member.cluster,
                }
            ).digest.removeprefix("sha256:")
            executor = PreparedShardExecutor(
                plane.catalog.get(member.cluster),
                member,
                root=root / "members" / member.cluster,
                resources=resources,
                equivalence=ExecutionEquivalence(*(["unattested"] * 5)),
                invocation=invocation,
                control_plane=plane,
                allocation_id=identity,
                discover_equivalence=True,
            )
            executors[member.cluster] = executor
            executor.start_allocation(
                source, plan.scientific_fingerprint, parent_job_id=parent_job_id
            )
        while True:
            offers = [executor.offer(()) for executor in executors.values()]
            for cluster, executor in executors.items():
                state = plane.jobs.get(executor.allocation_job_id, include_study=False).state
                if state.terminal:
                    raise RuntimeError(
                        f"Member {cluster} failed before execution readiness; "
                        f"inspect lf jobs logs {executor.allocation_job_id}."
                    )
            if all(offer.local_admission_verified for offer in offers):
                break
            if coordinator.snapshot()["remaining_seconds"] == 0:
                raise RuntimeError("Fleet preparation exhausted the original Study time budget.")
            time.sleep(1)
        if len({executor.equivalence for executor in executors.values()}) != 1:
            raise ValueError(
                "Fleet members have different verified execution strata; no Run launched."
            )
        dispatcher = CoordinatedDispatcher(coordinator, executors, poll_seconds=1)
        result = WorkRunner(dispatcher=dispatcher, execution_root=execution_root).run(config)
        return result.to_dict()
    finally:
        # Draining never kills scientific children and is idempotent. On an unreachable member
        # preserve the exact owned Job for reconciliation/operator cancellation, not a replacement.
        for executor in executors.values():
            try:
                executor.drain_allocation()
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                print(
                    f"[fleet] Member {executor.member.cluster} could not be drained "
                    f"({type(error).__name__}); its exact owner remains recorded for inspection.",
                    file=sys.stderr,
                    flush=True,
                )


def main(argv: Sequence[str] | None = None) -> int:
    values = tuple(argv if argv is not None else sys.argv[1:])
    if len(values) != 1:
        raise ValueError("Fleet coordinator requires its exact durable submission request.")
    from lambdaforge.controlplane.ClusterProfile import ClusterProfile
    from lambdaforge.ProjectContext import ProjectContext

    request = json.loads(Path(values[0]).read_text(encoding="utf-8"))
    job_id = request["job_id"]
    if os.environ.get("LAMBDAFORGE_JOB_ID") != job_id:
        raise ValueError("Fleet coordinator must run under its exact provider Job.")
    fleet = Fleet.from_mapping(
        request["fleet"]["name"], {key: request["fleet"][key] for key in ("members", "coordinator")}
    )
    project_data = request.get("project")
    project = (
        ProjectContext(Path(project_data["root"]), project_data["project_id"])
        if project_data
        else None
    )
    catalog = ClusterCatalog(
        {
            name: ClusterProfile.from_mapping(name, value)
            for name, value in request["member_profiles"].items()
        }
    )
    jobs = JobService(catalog, JobStore(request["job_store"], project=project))
    outcome = execute_fleet(
        Path(request["config"]), fleet, ControlPlane(catalog, jobs=jobs), parent_job_id=job_id
    )
    print(f"{outcome['name']}: {outcome['status']} ({outcome['execution_id']})")
    return 0 if outcome["status"] == "succeeded" else 4


if __name__ == "__main__":
    raise SystemExit(main())
