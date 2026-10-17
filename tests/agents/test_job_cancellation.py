"""Native cancellation uses the shared human approval boundary, never live science."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from lambdaforge.agents.approvals import ApprovalService
from lambdaforge.agents.errors import AgentApprovalError
from lambdaforge.agents.services import AgentServices
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="cancellation"\nversion="0.1"\n')
    client = AgentServices(tmp_path, permissions=("READ", "PLAN", "OPERATE"))
    catalog = ClusterCatalog({"local": ClusterProfile("local")}, project=client.scope.project)
    store = JobStore(tmp_path / "jobs", project=client.scope.project)
    jobs = JobService(catalog, store)
    record = JobRecord(
        job_id="job-exact-one",
        cluster="local",
        scheduler="local",
        scheduler_id=None,
        state=JobState.PREPARING,
        command=("not-launched",),
        work_dir=str(tmp_path / "work"),
        resources={},
        created_at_utc="2026-10-12T10:00:00+00:00",
        updated_at_utc="2026-10-12T10:00:00+00:00",
        metadata={"scientific_identity": "sha256:exact-science"},
    )
    store.write(record)
    store.write(record.with_updates(job_id="job-sibling"))
    client._jobs = jobs
    approvals = ApprovalService(client.scope, jobs=jobs)
    return client, jobs, approvals


def approved(approvals, proposal):
    approvals.approve_local(
        proposal["request_id"],
        plan_hash=proposal["plan_hash"],
        channel="console",
        confirmation=proposal["request_id"],
    )


def test_preview_pure_and_cancel_native_exact_job_preserves_siblings_and_evidence(workspace):
    client, jobs, approvals = workspace
    before = {p: p.read_bytes() for p in client.scope.root.rglob("*") if p.is_file()}
    plan = client.prepare_job_cancellation("job-exact-one")
    assert plan.status == "ok"
    assert {p: p.read_bytes() for p in client.scope.root.rglob("*") if p.is_file()} == before
    assert not (client.scope.root / approvals.directory).exists()
    proposal = client.request_job_cancellation("job-exact-one").data
    assert (
        client.request_job_cancellation("job-exact-one").data["request_id"]
        == proposal["request_id"]
    )
    blocked = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert blocked.error.code == "APPROVAL_REQUIRED"
    approved(approvals, proposal)
    reply = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert reply.status == "ok"
    assert reply.data["result"]["state"] == "cancelled"
    assert reply.data["result"]["scientific_evidence_deleted"] is False
    assert jobs.get("job-sibling", refresh=False).state is JobState.PREPARING
    assert (
        jobs.get("job-exact-one", refresh=False).metadata["scientific_identity"]
        == "sha256:exact-science"
    )
    assert (
        client.job_cancellation_status(proposal["request_id"]).data["result"]
        == reply.data["result"]
    )
    assert (
        client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"]).error.code
        == "OPERATION_CONFLICT"
    )


@pytest.mark.parametrize("permissions", [("READ", "PLAN"), ("READ", "PLAN", "WRITE_PROJECT")])
def test_permissions_and_toolsets_never_become_cancellation_authority(workspace, permissions):
    client, jobs, _ = workspace
    restricted = AgentServices(client.scope.root, permissions=permissions)
    restricted._jobs = jobs
    assert restricted.request_job_cancellation("job-exact-one").error.code == "PERMISSION_DENIED"
    assert (
        restricted.request_agent_action("jobs.cancel", {"job_id": "job-exact-one"}).status
        == "error"
    )
    knowledge_only = AgentServices(
        client.scope.root, permissions=("OPERATE",), toolsets=("knowledge",)
    )
    assert (
        knowledge_only.request_job_cancellation("job-exact-one").error.code == "PERMISSION_DENIED"
    )
    assert jobs.get("job-exact-one", refresh=False).state is JobState.PREPARING


@pytest.mark.parametrize("change", ["owner", "terminal", "unknown", "fleet", "command", "storage"])
def test_changed_or_uncertain_job_refuses_stale_approval(workspace, change):
    client, jobs, approvals = workspace
    proposal = client.request_job_cancellation("job-exact-one").data
    approved(approvals, proposal)
    record = jobs.get("job-exact-one", refresh=False)
    updates = {
        "owner": {"scheduler_id": "changed-exact-owner"},
        "terminal": {"state": JobState.SUCCEEDED},
        "unknown": {"state": JobState.UNKNOWN},
        "fleet": {"metadata": {**record.metadata, "fleet_role": "coordinator"}},
        "command": {"command": ("different-command",)},
        "storage": {
            "metadata": {**record.metadata, "local_storage": {"run_root": "/different/owner"}}
        },
    }
    jobs.store.write(record.with_updates(**updates[change]))
    reply = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert reply.error.code == "PLAN_STALE"
    assert approvals.inspect(proposal["request_id"])["state"] == "approved"


def test_fresh_provider_uncertainty_blocks_before_native_cancel(workspace, monkeypatch):
    client, jobs, approvals = workspace
    proposal = client.request_job_cancellation("job-exact-one").data
    approved(approvals, proposal)
    original = jobs.get

    def uncertain(job_id, *, refresh=True, include_study=True):
        record = original(job_id, refresh=False)
        return record.with_updates(state=JobState.UNKNOWN) if refresh else record

    monkeypatch.setattr(jobs, "get", uncertain)
    reply = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert reply.error.code == "PROVIDER_UNCERTAIN"
    assert approvals.inspect(proposal["request_id"])["state"] == "failed"
    assert original("job-exact-one", refresh=False).state is JobState.PREPARING


def test_concurrent_apply_claims_once_and_expiry_is_not_authority(workspace):
    client, jobs, approvals = workspace
    proposal = client.request_job_cancellation("job-exact-one").data
    approved(approvals, proposal)

    def apply():
        return client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: apply(), range(2)))
    assert sum(r.status == "ok" for r in replies) == 1
    assert [r.error.code for r in replies if r.status == "error"] == ["OPERATION_CONFLICT"]
    audit = (client.scope.root / approvals.directory / "audit.jsonl").read_text()
    assert sum(json.loads(line)["event"] == "claimed" for line in audit.splitlines()) == 1
    assert jobs.get("job-exact-one", refresh=False).state is JobState.CANCELLED


def test_cross_project_service_and_path_selectors_rejected(workspace, tmp_path):
    client, jobs, approvals = workspace
    for invalid in ("name", "job-../../outside", "job-absent"):
        assert client.prepare_job_cancellation(invalid).status == "error"
    other = tmp_path / "other"
    other.mkdir()
    (other / "pyproject.toml").write_text('[project]\nname="other"\nversion="0.1"\n')
    foreign = AgentServices(other, permissions=("OPERATE",))
    foreign._jobs = jobs
    assert foreign.request_job_cancellation("job-exact-one").error.code == "PERMISSION_DENIED"
    with pytest.raises(AgentApprovalError):
        # Local review cannot approve a different plan hash.
        proposal = approvals.request("jobs.cancel", {"job_id": "job-exact-one"})
        approvals.approve_local(
            proposal["request_id"],
            plan_hash="wrong",
            channel="console",
            confirmation=proposal["request_id"],
        )


def test_native_scheduler_receives_exact_owner_once_without_study_reads(workspace):
    client, jobs, approvals = workspace
    record = jobs.get("job-exact-one", refresh=False)
    jobs.store.write(
        record.with_updates(
            scheduler_id="native-owned-worker",
            state=JobState.RUNNING,
            work_dir=str(jobs.store.root / record.job_id / "work"),
        )
    )

    class Scheduler:
        calls = []

        def state(self, scheduler_id):
            assert scheduler_id == "native-owned-worker"
            return JobState.RUNNING

        def details(self, scheduler_id):
            return {"state": "running"}

        def cancel(self, scheduler_id):
            self.calls.append(scheduler_id)

    scheduler = Scheduler()

    class Factory:
        def transport(self, profile):
            # No subprocess, real provider or filesystem-based telemetry is required.
            return object()

        def scheduler(self, profile, transport):
            return scheduler

    jobs.factory = Factory()
    proposal = client.request_job_cancellation("job-exact-one").data
    assert not scheduler.calls
    approved(approvals, proposal)
    reply = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert reply.status == "ok", (reply.error, jobs.store.get("job-exact-one").metadata)
    assert scheduler.calls == ["native-owned-worker"]


def test_expired_job_plan_and_corrupt_evidence_fail_explicitly(workspace, monkeypatch):
    client, jobs, approvals = workspace
    proposal = client.request_job_cancellation("job-exact-one").data
    approved(approvals, proposal)
    import lambdaforge.agents.approvals as module

    monkeypatch.setattr(module.time, "time", lambda: proposal["expires_at"] + 1)
    reply = client.apply_job_cancellation(proposal["request_id"], proposal["plan_hash"])
    assert reply.error.code == "APPROVAL_EXPIRED"
    assert jobs.get("job-exact-one", refresh=False).state is JobState.PREPARING
    monkeypatch.undo()

    def corrupt(*args, **kwargs):
        raise json.JSONDecodeError("bad native evidence", "{}", 0)

    monkeypatch.setattr(jobs, "get", corrupt)
    assert client.prepare_job_cancellation("job-exact-one").error.code == "EVIDENCE_CORRUPT"


def test_storage_and_provider_binding_are_json_and_reject_catalog_changes(workspace):
    from dataclasses import replace

    from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy

    client, jobs, approvals = workspace
    record = jobs.get("job-exact-one", refresh=False)
    jobs.store.write(
        record.with_updates(
            metadata={
                **record.metadata,
                "local_storage": ClusterStoragePolicy.from_mapping(
                    {"run_root": str(jobs.store.root)}, workspace=str(client.scope.root)
                ).to_dict(),
            }
        )
    )
    proposal = client.request_job_cancellation("job-exact-one")
    assert proposal.status == "ok", proposal.error
    approved(approvals, proposal.data)
    profile = replace(jobs.catalog.definition("local"), workspace=str(client.scope.root / "new"))
    jobs.catalog = ClusterCatalog({"local": profile}, project=client.scope.project)
    reply = client.apply_job_cancellation(proposal.data["request_id"], proposal.data["plan_hash"])
    assert reply.error.code == "PLAN_STALE"
