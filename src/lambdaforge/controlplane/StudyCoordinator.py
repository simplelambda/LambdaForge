"""Durable operational Run ownership beneath a single scientific Study planner.

This module neither proposes candidates nor invokes Work.run. Executors accept concrete shards
through an idempotent protocol and retain their existing scheduler and local admission authority.
An unavailable executor never authorizes re-execution of its leases.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from lambdaforge.controlplane.Fleet import ClusterHealth, Fleet, FleetMember, positive_limit
from lambdaforge.controlplane.FleetPlacement import (
    ClusterOffer,
    GlobalPlacementBroker,
    GlobalRun,
)
from lambdaforge.reproducibility.ScientificIdentity import ScientificIdentity
from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock
from lambdaforge.work.atomic import atomic_write_json

_ACTIVE = {"leased", "queued", "running", "unknown_remote"}
_TERMINAL = {"completed", "pruned", "failed", "cancelled"}
_PRIORITIES = {"required": 0, "decision": 1, "confirmation": 2, "primary": 3, "predictive": 4}


class ContradictoryEvidenceError(RuntimeError):
    """A result contradicted accepted identity/evidence and has been quarantined."""


class ShardRejectedError(RuntimeError):
    """Executor positively rejected BEFORE accepting any Run; not a network ambiguity."""


@dataclass(frozen=True, slots=True)
class RunLease:
    """Exact logical Run/Attempt authorization granted to one owned shard."""

    run: GlobalRun
    attempt: int
    lease_id: str

    def __post_init__(self) -> None:
        if positive_limit(self.attempt, "Run Attempt") is None:
            raise ValueError("Run Attempt cannot be null.")
        if not isinstance(self.lease_id, str) or not self.lease_id.strip():
            raise ValueError("Run lease requires a nonempty exact identity.")

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run.to_dict(), "attempt": self.attempt, "lease_id": self.lease_id}


@dataclass(frozen=True, slots=True)
class StudyShard:
    """A finite assigned queue with no optimizer, seed allocator or global-state permission."""

    shard_id: str
    study_identity: str
    cluster: str
    leases: tuple[RunLease, ...]
    member: FleetMember

    def __post_init__(self) -> None:
        if not self.shard_id or not self.study_identity or not self.leases:
            raise ValueError("A shard requires owned identities and a nonempty concrete queue.")
        if self.cluster != self.member.cluster:
            raise ValueError("Shard target does not match its Fleet member.")
        if any(lease.run.study_identity != self.study_identity for lease in self.leases):
            raise ValueError("Shard contains evidence belonging to another Study.")
        if len({lease.run.key for lease in self.leases}) != len(self.leases) or len(
            {lease.lease_id for lease in self.leases}
        ) != len(self.leases):
            raise ValueError("Shard Run and lease identities must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_version": 1,
            "shard_id": self.shard_id,
            "study_identity": self.study_identity,
            "cluster": self.cluster,
            "member": self.member.to_dict(),
            "leases": [lease.to_dict() for lease in self.leases],
        }


@dataclass(frozen=True, slots=True)
class RemoteObservation:
    """Factual executor state. LOST needs positive local-authority proof, not a timeout."""

    run_key: str
    attempt: int
    lease_id: str
    state: str
    result: Mapping[str, Any] | None = None
    loss_proof: Mapping[str, Any] | None = None


class ShardExecutor(Protocol):
    """Provider boundary; implementations must use the configured scheduler/ARI/GPU policy.

    submit is idempotent by shard_id. observe must query an owned durable executor registry;
    missing transport/provider evidence is UNKNOWN, never LOST. A shard can finish while the
    coordinator is offline. No worker has credentials to control another execution cluster.
    """

    def submit(self, shard: StudyShard) -> str: ...

    def observe(self, shard: StudyShard, job_id: str | None) -> Sequence[RemoteObservation]: ...


class StudyCoordinator:
    """Atomic leases and idempotent ingestion; scientific proposals arrive from the real planner.

    A transaction is short and never spans provider calls. Immutable results live separately;
    the atomic state is authoritative for accepted digests, leases, attempts, placement and
    member controls. Repeated delivery repairs a crash between result publication and state
    publication. A driver should hold leadership() across its planning/reconciliation loop.
    """

    def __init__(self, root: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root).absolute()
        self.clock = clock
        self._validate_root()
        self.state_path = self.root / "coordinator.json"

    def _validate_root(self) -> None:
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError("Coordinator state cannot live below a symlink.")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Coordinator root must be a directory.")

    def _lock(self, name: str = ".state.lock") -> CrossProcessFileLock:
        self._validate_root()
        return CrossProcessFileLock(
            self.root / name,
            shared=False,
            timeout_seconds=5.0,
            poll_interval_seconds=0.05,
        )

    @contextmanager
    def leadership(self) -> Iterator[None]:
        """Only one driver may plan/dispatch this Study; a process exit releases ownership."""
        with self._lock(".coordinator-owner.lock"):
            yield

    @contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        with self._lock():
            value = self._read()
            yield value
            value["revision"] += 1
            atomic_write_json(self.state_path, value)

    def _read(self) -> dict[str, Any]:
        if self.state_path.is_symlink():
            raise ValueError("Coordinator state is symlinked.")
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or type(value.get("coordinator_version")) is not int
            or value["coordinator_version"] != 1
        ):
            raise ValueError("Missing, corrupt or unsupported coordinator state; refusing restart.")
        required = {
            "study_identity",
            "execution_id",
            "runs",
            "shards",
            "fleet",
            "members",
            "revision",
            "created_at",
            "attempts_reserved",
            "max_runs",
            "max_time_seconds",
            "fleet_history",
            "quarantine",
            "idle_reasons",
        }
        if not required <= value.keys() or any(
            not isinstance(value[name], dict)
            for name in ("runs", "shards", "members", "fleet", "idle_reasons")
        ):
            raise ValueError("Coordinator state is incomplete; refusing restart.")
        if any(
            type(value[name]) is not int or value[name] < 0
            for name in ("revision", "attempts_reserved")
        ) or any(not isinstance(value[name], list) for name in ("fleet_history", "quarantine")):
            raise ValueError("Coordinator state has invalid accounting; refusing restart.")
        if value["fleet"].get("fleet_version") != 1:
            raise ValueError("Unsupported persisted Fleet version; refusing restart.")
        return value

    def initialize(
        self,
        study_identity: str,
        execution_id: str,
        fleet: Fleet,
        *,
        max_runs: int | None = None,
        max_time_seconds: float | None = None,
    ) -> None:
        """Create a new owned execution exactly once, or verify its immutable startup contract."""
        if not study_identity or not execution_id:
            raise ValueError("Coordinator requires exact Study and Execution identities.")
        positive_limit(max_runs, "Study max_runs")
        if max_time_seconds is not None and (
            not isinstance(max_time_seconds, (int, float))
            or isinstance(max_time_seconds, bool)
            or not 0 < max_time_seconds < float("inf")
        ):
            raise ValueError("Study max_time_seconds must be finite and positive.")
        with self._lock():
            if self.state_path.exists():
                value = self._read()
                if (
                    value["study_identity"],
                    value["execution_id"],
                    value["fleet"]["name"],
                    value["fleet"]["coordinator"],
                    value["max_runs"],
                    value["max_time_seconds"],
                ) != (
                    study_identity,
                    execution_id,
                    fleet.name,
                    fleet.coordinator,
                    max_runs,
                    max_time_seconds,
                ):
                    raise ValueError("Restart cannot change Study/Execution identity or budgets.")
                return
            atomic_write_json(
                self.state_path,
                {
                    "coordinator_version": 1,
                    "revision": 0,
                    "study_identity": study_identity,
                    "execution_id": execution_id,
                    "created_at": self.clock(),
                    "fleet": fleet.to_dict(),
                    "members": {
                        member.cluster: {
                            "state": member.state.value,
                            "health": "degraded",
                            "failures": 0,
                            "retry_at": 0.0,
                        }
                        for member in fleet.members
                    },
                    "runs": {},
                    "shards": {},
                    "attempts_reserved": 0,
                    "max_runs": max_runs,
                    "max_time_seconds": max_time_seconds,
                    "fleet_history": [],
                    "quarantine": [],
                    "idle_reasons": {},
                },
            )

    def enqueue(self, runs: Sequence[GlobalRun]) -> None:
        """Idempotently accept planner decisions; a same key/different definition is an error."""
        with self._transaction() as value:
            for run in runs:
                if run.study_identity != value["study_identity"]:
                    raise ValueError("A Run belongs to another Study.")
                existing = value["runs"].get(run.key)
                if existing is not None:
                    if existing["definition"] != run.to_dict():
                        raise ValueError(
                            "A logical Run identity cannot acquire a different definition."
                        )
                    continue
                value["runs"][run.key] = {
                    "definition": run.to_dict(),
                    "state": "planned",
                    "attempts": [],
                    "accepted": None,
                }

    def cancel_planned(self, run_key: str, *, reason: str) -> bool:
        """Withdraw an unstarted predictive proposal only; never stop a resident worker here."""
        with self._transaction() as value:
            record = value["runs"][run_key]
            if (
                record["state"] != "planned"
                or record["definition"]["priority_class"] != "predictive"
            ):
                return False
            record.update(state="cancelled", withdrawal={"reason": reason, "at": self.clock()})
            return True

    def set_member_state(self, cluster: str, state: ClusterHealth) -> None:
        if state not in {ClusterHealth.ONLINE, ClusterHealth.DRAINING, ClusterHealth.DISABLED}:
            raise ValueError("Operator state must be online, draining or disabled.")
        with self._transaction() as value:
            member = value["members"][cluster]
            member["state"] = state.value
            value["fleet_history"].append(
                {"cluster": cluster, "state": state.value, "at": self.clock()}
            )

    def update_fleet(self, fleet: Fleet) -> None:
        """Compatible operational additions/caps only; active removed members stay draining."""
        with self._transaction() as value:
            if (
                fleet.name != value["fleet"]["name"]
                or fleet.coordinator != value["fleet"]["coordinator"]
            ):
                raise ValueError("A live coordinator cannot change its fleet/coordinator role.")
            old = {item["cluster"]: item for item in value["fleet"]["members"]}
            new = {member.cluster: member.to_dict() for member in fleet.members}
            for cluster, descriptor in old.items():
                if cluster not in new:
                    new[cluster] = {**descriptor, "state": "draining"}
                    value["members"][cluster]["state"] = "draining"
            for cluster, descriptor in new.items():
                member = value["members"].setdefault(
                    cluster, {"health": "degraded", "failures": 0, "retry_at": 0.0}
                )
                member["state"] = descriptor["state"]
            value["fleet"] = {**fleet.to_dict(), "members": list(new.values())}
            value["fleet_history"].append({"at": self.clock(), "fleet": value["fleet"]})

    @staticmethod
    def _members(value: Mapping[str, Any]) -> dict[str, FleetMember]:
        return {
            item["cluster"]: replace(
                FleetMember.from_mapping(item),
                state=ClusterHealth(value["members"][item["cluster"]]["state"]),
            )
            for item in value["fleet"]["members"]
        }

    @staticmethod
    def _shard_active(value: Mapping[str, Any], shard_id: str) -> bool:
        return any(
            value["runs"][key]["state"] in _ACTIVE
            and value["runs"][key]["attempts"][-1]["shard_id"] == shard_id
            for key in value["shards"][shard_id]["run_keys"]
        )

    def plan_shards(
        self, offers: Sequence[ClusterOffer], *, adaptive: bool = False
    ) -> tuple[StudyShard, ...]:
        """Reserve unique attempts BEFORE crossing a provider boundary.

        Fixed evidence amortizes the whole current offer. Adaptive shards are one dispatch
        wave only: no extra speculative queue is created by placement. Unknown leases occupy
        caps until positively reconciled. Scientific Run/time budgets also include lookahead.
        """
        shards = []
        broker = GlobalPlacementBroker()
        with self._transaction() as value:
            now = self.clock()
            if (
                value["max_time_seconds"] is not None
                and now - value["created_at"] >= value["max_time_seconds"]
            ):
                value["idle_reasons"] = {"study": ["time-budget-exhausted"]}
                return ()
            members = self._members(value)
            if len({offer.cluster for offer in offers}) != len(offers):
                raise ValueError("One fresh offer per cluster is required.")
            observed_offers = {offer.cluster: offer for offer in offers}
            for cluster, member in members.items():
                offer = observed_offers.get(cluster)
                if member.required and (offer is None or offer.health == ClusterHealth.UNREACHABLE):
                    raise RuntimeError(f"Required fleet member {cluster!r} is unreachable.")
            active = Counter(
                record["attempts"][-1]["cluster"]
                for record in value["runs"].values()
                if record["state"] in _ACTIVE
            )
            jobs = Counter(
                shard["cluster"]
                for shard_id, shard in value["shards"].items()
                if self._shard_active(value, shard_id)
            )
            # Offers already exclude locally-known residents. Subtract only leases that the
            # local executor has NOT acknowledged, closing the plan/submit observation gap.
            remaining = {
                offer.cluster: max(
                    0,
                    offer.slots
                    - sum(
                        record["state"] in _ACTIVE
                        and record["attempts"][-1]["cluster"] == offer.cluster
                        and record["attempts"][-1]["lease_id"] not in offer.acknowledged_leases
                        for record in value["runs"].values()
                    ),
                )
                for offer in offers
            }
            assigned: dict[str, list[RunLease]] = {}
            planned = sorted(
                (
                    (key, record)
                    for key, record in value["runs"].items()
                    if record["state"] == "planned"
                ),
                key=lambda pair: (_PRIORITIES[pair[1]["definition"]["priority_class"]], pair[0]),
            )
            value["idle_reasons"] = {}
            for key, record in planned:
                if (
                    value["max_runs"] is not None
                    and value["attempts_reserved"] >= value["max_runs"]
                ):
                    value["idle_reasons"][key] = {"study": ["run-budget-exhausted"]}
                    break
                run = GlobalRun.from_mapping(record["definition"])
                fresh = [
                    replace(offer, slots=remaining[offer.cluster])
                    for offer in offers
                    if value["members"].get(offer.cluster, {}).get("retry_at", 0) <= now
                ]
                # Multiple assignments to this same shard consume just one scheduler Job.
                placement_jobs = {
                    cluster: count - int(cluster in assigned) for cluster, count in jobs.items()
                }
                placement, exclusions = broker.place(
                    run, fresh, members, now=now, active=active, jobs=placement_jobs
                )
                if placement is None:
                    value["idle_reasons"][key] = {
                        name: list(reasons) for name, reasons in exclusions.items()
                    }
                    continue
                cluster = placement.cluster
                lease = RunLease(run, len(record["attempts"]) + 1, uuid4().hex)
                record["state"] = "leased"
                record["attempts"].append(
                    {
                        "attempt": lease.attempt,
                        "lease_id": lease.lease_id,
                        "cluster": cluster,
                        "created_at": now,
                        "acknowledged_at": None,
                        "state": "leased",
                        "job_id": None,
                        "placement": dict(placement.reason),
                    }
                )
                value["attempts_reserved"] += 1
                remaining[cluster] -= 1
                active[cluster] += 1
                if cluster not in assigned:
                    jobs[cluster] += 1
                assigned.setdefault(cluster, []).append(lease)
            for cluster, leases in assigned.items():
                shard_id = ScientificIdentity.from_payload(
                    {
                        "execution": value["execution_id"],
                        "cluster": cluster,
                        "leases": [lease.lease_id for lease in leases],
                    }
                ).digest.removeprefix("sha256:")
                shard = StudyShard(
                    shard_id, value["study_identity"], cluster, tuple(leases), members[cluster]
                )
                shards.append(shard)
                value["shards"][shard_id] = {
                    "cluster": cluster,
                    "job_id": None,
                    "run_keys": [lease.run.key for lease in leases],
                    "definition": shard.to_dict(),
                    "mode": "adaptive-wave" if adaptive else "fixed-wave",
                }
                for lease in leases:
                    value["runs"][lease.run.key]["attempts"][-1]["shard_id"] = shard_id
        return tuple(shards)

    def submit_shard(self, shard: StudyShard, executor: ShardExecutor) -> str | None:
        """An ambiguous acknowledgement remains UNKNOWN; never automatically submit twice."""
        # Verification and the submission-intent fence are a SINGLE transaction: competing
        # drivers cannot both cross the provider boundary, even before remote idempotency.
        with self._transaction() as value:
            persisted = value["shards"][shard.shard_id]
            if persisted["definition"] != shard.to_dict():
                raise ValueError("Shard does not match its durable lease record.")
            if persisted["job_id"] is not None:
                return str(persisted["job_id"])
            if any(value["runs"][lease.run.key]["state"] != "leased" for lease in shard.leases):
                raise RuntimeError("Ambiguous shard submission must be reconciled, not repeated.")
            if value["members"][shard.cluster]["state"] != "online":
                # A proven unsubmitted lease may be withdrawn without touching a remote PID.
                # Preserve the Attempt audit but release its unspent dispatch reservation.
                for lease in shard.leases:
                    record = value["runs"][lease.run.key]
                    record["state"] = "planned"
                    record["attempts"][-1].update(
                        state="withdrawn_before_submit", reason="operator-drain-or-disable"
                    )
                value["attempts_reserved"] -= len(shard.leases)
                return None
            for lease in shard.leases:
                record = value["runs"][lease.run.key]
                record["state"] = "unknown_remote"
                record["attempts"][-1]["state"] = "unknown_remote"
        try:
            job_id = executor.submit(shard)
            if not job_id:
                raise RuntimeError("Executor returned no durable Job acknowledgement.")
        except ShardRejectedError as error:
            with self._transaction() as value:
                for lease in shard.leases:
                    record = value["runs"][lease.run.key]
                    record["state"] = "failed"
                    record["attempts"][-1].update(
                        state="rejected",
                        failure={"type": type(error).__name__, "message": str(error)},
                    )
                value["attempts_reserved"] -= len(shard.leases)
                value["members"][shard.cluster]["health"] = "degraded"
            return None
        except Exception as error:
            health = (
                ClusterHealth.UNREACHABLE
                if isinstance(error, (ConnectionError, TimeoutError, OSError))
                else ClusterHealth.DEGRADED
            )
            self._mark_unavailable(shard.cluster, f"{type(error).__name__}: {error}", health)
            return None
        with self._transaction() as value:
            value["shards"][shard.shard_id]["job_id"] = job_id
            for lease in shard.leases:
                record = value["runs"][lease.run.key]
                # An owned observation/result can arrive while submit awaits its ACK. Record
                # the ACK against its EXACT Attempt, never the latest replacement Attempt;
                # neither completed evidence nor a running observation may regress to queued.
                attempt = next(
                    item for item in record["attempts"] if item["lease_id"] == lease.lease_id
                )
                attempt["job_id"] = job_id
                if attempt["acknowledged_at"] is None:
                    attempt["acknowledged_at"] = self.clock()
                if attempt is record["attempts"][-1] and attempt["state"] == "unknown_remote":
                    attempt["state"] = "queued"
                    record["state"] = "queued"
        return job_id

    def mark_unreachable(self, cluster: str, reason: str) -> None:
        """Backoff the member; retain every active lease without creating a replacement."""
        self._mark_unavailable(cluster, reason, ClusterHealth.UNREACHABLE)

    def _mark_unavailable(self, cluster: str, reason: str, health: ClusterHealth) -> None:
        with self._transaction() as value:
            member = value["members"][cluster]
            member["health"] = health.value
            member["failures"] += 1
            member["retry_at"] = self.clock() + min(300, 2 ** min(member["failures"], 8))
            member["reason"] = reason
            for record in value["runs"].values():
                if record["state"] in _ACTIVE and record["attempts"][-1]["cluster"] == cluster:
                    record["state"] = "unknown_remote"
                    record["attempts"][-1]["state"] = "unknown_remote"

    @staticmethod
    def _shard(value: Mapping[str, Any]) -> StudyShard:
        if value.get("shard_version") != 1:
            raise ValueError("Unsupported persisted shard version.")
        return StudyShard(
            value["shard_id"],
            value["study_identity"],
            value["cluster"],
            tuple(
                RunLease(GlobalRun.from_mapping(item["run"]), item["attempt"], item["lease_id"])
                for item in value["leases"]
            ),
            FleetMember.from_mapping(value["member"]),
        )

    def reconcile(self, cluster: str, executor: ShardExecutor) -> dict[str, int]:
        """Ask the existing local authority; only a positively LOST Attempt becomes retryable."""
        with self._lock():
            value = self._read()
            selected = [
                (self._shard(shard["definition"]), shard["job_id"])
                for shard_id, shard in value["shards"].items()
                if shard["cluster"] == cluster and self._shard_active(value, shard_id)
            ]
        counts: Counter[str] = Counter()
        try:
            for shard, job_id in selected:
                observations = executor.observe(shard, job_id)
                expected = {
                    (lease.run.key, lease.attempt, lease.lease_id) for lease in shard.leases
                }
                for observation in observations:
                    if (
                        observation.run_key,
                        observation.attempt,
                        observation.lease_id,
                    ) not in expected:
                        raise ValueError(
                            "Executor observation does not belong to the requested shard."
                        )
                    if observation.result is not None:
                        expected_result_owner = {
                            "run_key": observation.run_key,
                            "attempt": observation.attempt,
                            "lease_id": observation.lease_id,
                            "cluster": shard.cluster,
                        }
                        if any(
                            observation.result.get(name) != owner
                            for name, owner in expected_result_owner.items()
                        ):
                            raise ValueError(
                                "Result envelope belongs to a different observed lease."
                            )
                        self.ingest(observation.result)
                        counts["ingested"] += 1
                    else:
                        self._observe(observation)
                        counts[observation.state] += 1
                reported = {(item.run_key, item.attempt, item.lease_id) for item in observations}
                for key, attempt, lease_id in expected - reported:
                    self._observe(RemoteObservation(key, attempt, lease_id, "unknown_remote"))
                    counts["unknown_remote"] += 1
        except (ConnectionError, TimeoutError, OSError) as error:
            self.mark_unreachable(cluster, f"{type(error).__name__}: {error}")
            counts["unreachable"] += 1
            return dict(counts)
        with self._transaction() as value:
            value["members"][cluster].update(health="online", failures=0, retry_at=0.0)
        return dict(counts)

    def _observe(self, observation: RemoteObservation) -> None:
        if observation.state not in {"queued", "running", "unknown_remote", "lost"}:
            raise ValueError(
                "Terminal observations require structured evidence, not a status alone."
            )
        if observation.state == "lost" and (
            not observation.loss_proof
            or observation.loss_proof.get("authority")
            not in {"scheduler", "owned-process-store", "owned-shard-store"}
            or observation.loss_proof.get("lease_id") != observation.lease_id
            or observation.loss_proof.get("no_live_owner") is not True
        ):
            raise ValueError("A missing heartbeat/connection is not proof of a lost Attempt.")
        with self._transaction() as value:
            record = value["runs"][observation.run_key]
            attempt = record["attempts"][-1]
            if (
                attempt["attempt"] != observation.attempt
                or attempt["lease_id"] != observation.lease_id
            ):
                return  # A delayed old observation cannot regress a replacement Attempt.
            if record["state"] in _TERMINAL or attempt["state"] == "lost":
                return
            if attempt["state"] == "running" and observation.state == "queued":
                return  # Delayed queue snapshots cannot undo an observed start.
            attempt["state"] = observation.state
            if attempt["acknowledged_at"] is None:
                attempt["acknowledged_at"] = self.clock()
            if observation.state == "lost":
                attempt["loss_proof"] = dict(observation.loss_proof or {})
                record["state"] = "lost"
            else:
                record["state"] = observation.state

    def retry_lost(self, run_key: str, *, maximum_retries: int = 1) -> bool:
        """Only confirmed operational loss, explicitly bounded; never retry scientific failures."""
        if (
            isinstance(maximum_retries, bool)
            or not isinstance(maximum_retries, int)
            or not 0 <= maximum_retries <= 3
        ):
            raise ValueError("maximum_retries must be 0–3.")
        with self._transaction() as value:
            record = value["runs"][run_key]
            if record["state"] != "lost" or len(record["attempts"]) > maximum_retries:
                return False
            record["state"] = "planned"
            return True

    def ingest(self, evidence: Mapping[str, Any]) -> bool:
        """Accept once by immutable Study/Run/Attempt/lease/stratum; quarantine contradictions."""
        # Strict JSON also rejects NaN, arbitrary Python objects and untyped payloads.
        normalized = json.loads(json.dumps(dict(evidence), allow_nan=False, sort_keys=True))
        digest = ScientificIdentity.from_payload(normalized).digest
        quarantined = False
        accepted = False
        with self._transaction() as value:
            key = normalized.get("run_key")
            record = value["runs"].get(key)
            duplicate = record is not None and record.get("accepted") == digest
            if duplicate:
                return False
            reason = None
            if record is None or not record["attempts"]:
                reason = "unassigned-run"
            else:
                attempt = record["attempts"][-1]
                definition = record["definition"]
                expected = {
                    "study_identity": value["study_identity"],
                    "run_key": key,
                    "attempt": attempt["attempt"],
                    "lease_id": attempt["lease_id"],
                    "cluster": attempt["cluster"],
                    "parameters": definition["parameters"],
                    "equivalence": definition["equivalence"],
                }
                if any(
                    normalized.get(name) != expected_value
                    for name, expected_value in expected.items()
                ):
                    reason = "identity-or-execution-equivalence-mismatch"
                elif record.get("accepted") is not None:
                    reason = "contradictory-terminal-evidence"
                elif attempt["state"] == "lost" or record["state"] not in _ACTIVE:
                    reason = "late-result-for-released-attempt"
                elif normalized.get("state") not in _TERMINAL or not isinstance(
                    normalized.get("result"), dict
                ):
                    reason = "invalid-terminal-envelope"
            filename = digest.removeprefix("sha256:") + ".json"
            directory = self.root / ("quarantine" if reason else "results")
            if directory.is_symlink():
                raise ValueError("Coordinator evidence directory is symlinked.")
            destination = directory / filename
            if destination.is_symlink():
                raise ValueError("Coordinator evidence is symlinked.")
            atomic_write_json(destination, normalized)
            if reason:
                value["quarantine"].append({"digest": digest, "reason": reason, "run_key": key})
                quarantined = True
            else:
                record["accepted"] = digest
                record["state"] = normalized["state"]
                record["attempts"][-1].update(
                    state=normalized["state"], finished_at=self.clock(), result_digest=digest
                )
                accepted = True
        if quarantined:
            raise ContradictoryEvidenceError(f"Result quarantined: {digest}; {reason}.")
        return accepted

    def result(self, run_key: str) -> dict[str, Any] | None:
        with self._lock():
            digest = self._read()["runs"][run_key]["accepted"]
            if digest is None:
                return None
            path = self.root / "results" / (digest.removeprefix("sha256:") + ".json")
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("Accepted evidence is symlinked.")
            value = json.loads(path.read_text(encoding="utf-8"))
            if ScientificIdentity.from_payload(value).digest != digest:
                raise ValueError("Accepted evidence digest mismatch.")
            return dict(value)

    def snapshot(self) -> dict[str, Any]:
        """Bounded semantic status; do not transfer every per-Run result to Overview."""
        with self._lock():
            value = self._read()
            return {
                "coordinator_version": value["coordinator_version"],
                "revision": value["revision"],
                "study_identity": value["study_identity"],
                "execution_id": value["execution_id"],
                "fleet": value["fleet"]["name"],
                "members": value["members"],
                "counts": dict(Counter(record["state"] for record in value["runs"].values())),
                "priority_counts": dict(
                    Counter(
                        record["definition"]["priority_class"]
                        for record in value["runs"].values()
                        if record["state"] in _ACTIVE
                    )
                ),
                "attempts_reserved": value["attempts_reserved"],
                "quarantined": len(value["quarantine"]),
                "idle_reasons": dict(list(value["idle_reasons"].items())[:25]),
            }
