"""Native operational Jobs survive UI/controller exit; fixtures never contact real clusters."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterAuthentication import ClusterAuthentication
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
from lambdaforge.data.DatasetOperations import DatasetOperations
from lambdaforge.data.DatasetReplicationService import DatasetReplicationService
from tests.datasets.test_compressed_replication import transfer  # noqa: F401


@pytest.fixture
def durable(request):
    service, record, root = request.getfixturevalue("transfer")
    for name in service.clusters.names():
        service.clusters._profiles[name] = replace(
            service.clusters.get(name), environment="existing"
        )
    jobs = JobService(service.clusters, JobStore(root / "history"), factory=service.factory)
    return DatasetReplicationService(service, jobs), record, root


def finish(value, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = value.status(job_id)
        if row["state"] in {"succeeded", "failed", "cancelled", "timeout"}:
            assert row["state"] == "succeeded", value.jobs.logs(job_id, tail=100)
            return row
        time.sleep(0.1)
    pytest.fail(f"Transfer did not terminate: {value.jobs.logs(job_id, tail=100)}")


def test_native_job_reconnects_and_does_not_become_scientific_work(durable):
    value, record, _root = durable
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    assert plan.job_id and plan.requires_controller_online
    fresh = DatasetReplicationService(
        value.datasets,
        JobService(
            value.datasets.clusters,
            value.jobs.store,
            factory=value.datasets.factory,
        ),
    )
    row = finish(fresh, plan.job_id)
    assert row["progress"]["phase"] == "completed"
    assert row["content_id"] == record.dataset_id
    placement = next(
        p for p in value.datasets.registry.get(record.key).placements if p.cluster == "gpu16"
    )
    assert DatasetOperations.verify(Path(placement.root), record.dataset_id)["valid"]
    assert not aggregate_research_work(value.jobs.list(refresh=False))


@pytest.mark.parametrize("relative", [True, False])
def test_worker_request_and_local_snapshot_paths_are_cwd_independent(
    durable, monkeypatch, relative
):
    from lambdaforge.ProjectContext import ProjectContext

    value, record, root = durable
    (root / "pyproject.toml").write_text('[project]\nname="replication-fixture"\nversion="1.0"\n')
    nested = root / "commands"
    nested.mkdir()
    monkeypatch.chdir(nested)
    project = ProjectContext(root, "replication-paths")
    value.datasets.clusters.project = project
    value.jobs.store.project = project

    def local_path(path):
        return str(path if relative else root / path)

    for name in ("local", "gpu16"):
        profile = value.datasets.clusters.get(name)
        base = Path(".lambdaforge") / name
        value.datasets.clusters._profiles[name] = replace(
            profile,
            workspace=local_path(base),
            storage=replace(
                profile.storage,
                state_root=local_path(base / "state"),
                cache_root=local_path(base / "cache"),
                run_root=local_path(base / "jobs"),
                dataset_root=local_path(base / "datasets"),
                lease_root=local_path(Path("leases")),
            ),
        )
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    job = value.jobs.get(plan.job_id, refresh=False)
    request_path = Path(job.command[-1])
    expected = root / ".lambdaforge/local/jobs" / job.job_id / "work"
    assert request_path == expected / "replication.json"
    assert job.work_dir == str(expected)
    snapshot = json.loads(request_path.read_text())
    assert Path(snapshot["registry"]).is_absolute()
    for name in ("local", "gpu16"):
        profile = snapshot["profiles"][name]
        assert profile["workspace"] == str(root / ".lambdaforge" / name)
        for field in ("state_root", "cache_root", "run_root", "dataset_root", "lease_root"):
            assert Path(profile["storage"][field]).is_absolute()
            assert Path(profile["storage"][field]).is_relative_to(root)
    supervisor = json.loads((expected.parent / "request.json").read_text())
    assert supervisor["work_dir"] == str(expected)
    assert supervisor["command"][-1] == str(request_path)
    # Real native supervisor executes DatasetReplicationWorker with cwd=work_dir.
    row = finish(value, plan.job_id)
    assert row["state"] == "succeeded"
    assert row["placement"]["root"].startswith(str(root / ".lambdaforge/gpu16/datasets"))
    assert not list(nested.rglob("replication.json"))


def test_normalization_preserves_remote_paths_and_pins_local_interpreter(durable):
    from lambdaforge.ProjectContext import ProjectContext

    value, _record, root = durable
    value.datasets.clusters.project = ProjectContext(root, "remote-paths")
    remote = replace(
        value.datasets.clusters.get("gpu16"),
        transport="ssh",
        host="host.invalid",
        workspace="/remote/nonexistent/workspace",
        python="envs/runtime/bin/python",
        storage=replace(
            value.datasets.clusters.get("gpu16").storage,
            state_root="/remote/state",
            cache_root="/remote/cache",
            run_root="/remote/jobs",
            dataset_root="/remote/datasets",
        ),
    )
    value.datasets.clusters._profiles["gpu16"] = remote
    original = value.datasets.clusters.get("gpu16").to_dict()
    value.datasets.clusters._profiles["local"] = replace(
        value.datasets.clusters.get("local"), python="bin/project-python"
    )
    profiles = value._execution_profiles()
    assert profiles["gpu16"].to_dict() == original
    assert profiles["local"].python == str(root / "bin/project-python")
    assert not (root / "remote").exists()


def test_early_worker_failure_has_redacted_diagnostic_in_status_and_list(
    durable, monkeypatch, capsys
):
    import importlib
    from types import SimpleNamespace

    value, record, _root = durable
    original = value.datasets.factory.scheduler

    def scheduler(profile, transport):
        native = original(profile, transport)
        submit = native.submit

        def fail_before_result(_command, resources, *, job_id=None, **kwargs):
            return submit(
                (
                    sys.executable,
                    "-m",
                    "lambdaforge.data.DatasetReplicationWorker",
                    str(
                        Path(kwargs["work_dir"])
                        / "replication request missing-password=FAKE_SECRET"
                    ),
                ),
                resources,
                job_id=job_id,
                **kwargs,
            )

        native.submit = fail_before_result
        return native

    monkeypatch.setattr(value.datasets.factory, "scheduler", scheduler)
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        row = value.status(plan.job_id)
        if row["state"] == "failed":
            break
        time.sleep(0.1)
    assert row["state"] == "failed"
    assert "FileNotFoundError" in row["error"] and "replication request missing" in row["error"]
    assert "FAKE_SECRET" not in json.dumps(row)
    assert plan.job_id in row["logs_command"]
    assert value.jobs.get(plan.job_id, refresh=False).stderr == ""
    # Older terminal caches have no diagnostic; the first refreshed listing upgrades them.
    value.jobs.store.merge_metadata(
        plan.job_id, {"dataset_replication_outcome": {"progress": {}, "result": {}}}
    )
    assert value.list(record.key)[0]["error"] == row["error"]
    job_root = Path(value.jobs.get(plan.job_id, refresh=False).work_dir).parent
    log = job_root / "stderr.log"
    log.write_bytes(b"old operational output\n" * 1500 + log.read_bytes())
    assert value.status(plan.job_id)["error"] == row["error"]
    assert not (job_root / "result.json").exists() and not (job_root / "progress.json").exists()
    assert all(p.cluster != "gpu16" for p in value.datasets.registry.get(record.key).placements)
    assert "FAKE_SECRET" not in json.dumps(value.jobs.store.get(plan.job_id).metadata)
    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    service = SimpleNamespace(
        replications=value.list, replication_status=value.status, discovery_warnings=()
    )
    monkeypatch.setattr(module, "DatasetService", lambda **_kwargs: service)
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _path: value.datasets.clusters)
    for operation in ("transfers", "transfer-status"):
        args = SimpleNamespace(
            dataset_command=operation,
            dataset=record.key,
            job_id=plan.job_id,
            clusters=None,
            json=True,
        )
        assert module.DatasetCommands.run(args) == 0
        response = json.loads(capsys.readouterr().out)
        assert (response[0] if isinstance(response, list) else response)["error"] == row["error"]
    module.DatasetCommands._human("transfers", value.list(record.key))
    output = capsys.readouterr().out
    assert "FileNotFoundError" in output and row["logs_command"] in output
    assert "FAKE_SECRET" not in output
    # A missing/rotated log still produces a useful diagnostic, not error="".
    (job_root / "stderr.log").unlink()
    unavailable = value.status(plan.job_id)
    assert "no stderr diagnostic available" in unavailable["error"]
    assert unavailable["logs_command"] in unavailable["error"]


def test_duplicate_concurrent_requests_reuse_exact_operational_job(durable, monkeypatch):
    value, record, root = durable
    original = value.datasets.factory.scheduler
    gate = root / "release"

    def scheduler(profile, transport):
        result = original(profile, transport)
        submit = result.submit

        def gated(command, resources, *, job_id=None, **kwargs):
            wrapper = (
                "import os,sys,time;from pathlib import Path;"
                "exec('while not Path(sys.argv[1]).exists(): time.sleep(.05)');"
                "os.execv(sys.argv[2],sys.argv[2:])"
            )
            return submit(
                (sys.executable, "-c", wrapper, str(gate), *command),
                resources,
                job_id=job_id,
                **kwargs,
            )

        result.submit = gated
        return result

    monkeypatch.setattr(value.datasets.factory, "scheduler", scheduler)
    try:
        with ThreadPoolExecutor(2) as pool:
            plans = list(
                pool.map(
                    lambda _: value.enqueue(record.key, source="local", destination="gpu16"),
                    range(2),
                )
            )
        assert plans[0].job_id == plans[1].job_id
        assert len(value.jobs.list(refresh=False)) == 1
    finally:
        gate.touch()
    finish(value, plans[0].job_id)


def test_parent_process_exits_before_transfer_starts_but_job_finishes(durable):
    value, record, root = durable
    gate = root / "parent-gone"
    config = root / "controller.json"
    config.write_text(
        json.dumps(
            {
                "registry": str(value.datasets.registry.path),
                "history": str(value.jobs.store.root),
                "profiles": {
                    n: value.datasets.clusters.get(n).to_dict()
                    for n in value.datasets.clusters.names()
                },
                "dataset": record.key,
                "gate": str(gate),
            }
        )
    )
    script = """import json,sys
from pathlib import Path
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.data.DatasetRegistry import DatasetRegistry
from lambdaforge.data.DatasetService import DatasetService
from lambdaforge.data.DatasetReplicationService import DatasetReplicationService
p=json.loads(Path(sys.argv[1]).read_text())
c=ClusterCatalog({n:ClusterProfile.from_mapping(n,v) for n,v in p['profiles'].items()})
f=ControlPlaneFactory(); original=f.scheduler
def scheduler(profile,transport):
 s=original(profile,transport); submit=s.submit
 def gated(command,resources,*,job_id=None,**kwargs):
  wrapper=("import os,sys,time;from pathlib import Path;"
           "exec('while not Path(sys.argv[1]).exists(): time.sleep(.05)');"
           "os.execv(sys.argv[2],sys.argv[2:])")
  return submit((sys.executable,'-c',wrapper,p['gate'],*command),
                resources,job_id=job_id,**kwargs)
 s.submit=gated
 return s
f.scheduler=scheduler
d=DatasetService(DatasetRegistry(p['registry']),c,f)
v=DatasetReplicationService(d,JobService(c,JobStore(p['history']),f))
print(v.enqueue(p['dataset'],source='local',destination='gpu16').job_id,flush=True)
"""
    parent = subprocess.run(
        [sys.executable, "-c", script, str(config)],
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )
    job_id = parent.stdout.strip()
    try:
        assert value.status(job_id)["state"] in {"running", "staging"}
        assert all(p.cluster != "gpu16" for p in value.datasets.registry.get(record.key).placements)
    finally:
        gate.touch()
    finish(value, job_id)


@pytest.mark.parametrize("project_scoped", [False, True])
def test_direct_transfer_is_supervised_on_source_without_controller_stream(
    durable, monkeypatch, project_scoped
):
    value, record, _root = durable
    service = value.datasets
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    for name in ("gpu12", "gpu16"):
        service.clusters._profiles[name] = replace(
            service.clusters.get(name), transport="ssh", host=f"{name}.invalid"
        )
    if project_scoped:
        from lambdaforge.data.DatasetRegistry import DatasetRegistry
        from lambdaforge.ProjectContext import ProjectContext

        project = ProjectContext(_root, "transfer-project")
        service.clusters.project = project
        value.jobs.store.project = project
        origin = service.clusters.get("gpu12")
        DatasetRegistry(Path(origin.storage.state_root) / "datasets.json").register(
            service.registry.get(record.key)
        )
    transport = LocalTransport()
    monkeypatch.setattr(service.factory, "transport", lambda _p: transport)
    monkeypatch.setattr(service, "_site_ssh", lambda _p, command: command)
    monkeypatch.setattr(
        transport, "stream", lambda _cmd: pytest.fail("Controller must not relay direct bytes")
    )
    plan = value.enqueue(record.key, source="gpu12", destination="gpu16", route="direct")
    assert not plan.requires_controller_online
    job = value.jobs.get(plan.job_id, refresh=False)
    assert job.work_dir == str(
        Path(service.clusters.get("gpu12").storage.job_root) / plan.job_id / "work"
    )
    if project_scoped:
        assert job.work_dir.count("projects/transfer-project") == 1
    row = finish(value, plan.job_id)
    assert row["host"] == "gpu12" and row["route"] == "direct-ssh"
    assert row["progress"]["phase"] == "completed"
    assert service.resolve_placement(record.key, "gpu16").physically_available


def test_detached_relay_refuses_interactive_credentials_without_creating_job(durable):
    value, record, _root = durable
    value.datasets.clusters._profiles["gpu16"] = replace(
        value.datasets.clusters.get("gpu16"),
        transport="ssh",
        host="invalid.test",
        auth=ClusterAuthentication("password"),
    )
    # Preview/read-only discovery is stubbed; no password prompt or real connection.
    from lambdaforge.data.DatasetMaterializationPlan import DatasetMaterializationPlan

    value.datasets.replicate = lambda *_args, **_kwargs: DatasetMaterializationPlan(
        record.key,
        "gpu16",
        "REPLICATE",
        "local",
        transfer_route="controller-stream",
        content_id=record.dataset_id,
    )
    with pytest.raises(ValueError, match="cannot prompt"):
        value.enqueue(record.key, source="local", destination="gpu16")
    assert not value.jobs.list(refresh=False)


def test_status_rejects_unsafe_paths_and_unrelated_jobs(durable, monkeypatch):
    value, record, root = durable
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    finish(value, plan.job_id)
    original = value.jobs.get(plan.job_id)
    monkeypatch.setattr(
        value.jobs,
        "get",
        lambda *_a, **_k: original.with_updates(work_dir=str(root / "elsewhere/work")),
    )
    with pytest.raises(ValueError, match="Unsafe transfer"):
        value.status(plan.job_id)
    monkeypatch.setattr(value.jobs, "get", lambda *_a, **_k: original.with_updates(job_type="work"))
    with pytest.raises(ValueError, match="not a Dataset"):
        value.status(plan.job_id)


def test_transfer_commands_are_registered():
    from lambdaforge.cli.parser import build_parser

    parser = build_parser()
    assert (
        parser.parse_args(["datasets", "transfers", "example@7", "--json"]).dataset == "example@7"
    )
    assert (
        parser.parse_args(["datasets", "transfer-status", "job-example", "--json"]).job_id
        == "job-example"
    )


def test_status_reads_only_metadata_and_rejects_changed_identity(durable, monkeypatch):
    value, record, _root = durable
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    finish(value, plan.job_id)
    monkeypatch.setattr(
        DatasetOperations, "verify", lambda *_a, **_k: pytest.fail("No asset scans in status")
    )
    assert value.status(plan.job_id)["state"] == "succeeded"
    root = Path(value.jobs.get(plan.job_id).work_dir).parent
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    result["content_id"] = "sha256:" + "0" * 64
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="pinned Job identity"):
        value.status(plan.job_id)


def test_history_preserves_native_state_when_progress_host_is_unreachable(durable, monkeypatch):
    value, record, _root = durable
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    finish(value, plan.job_id)

    value.jobs.store.merge_metadata(plan.job_id, {"dataset_replication_outcome": None})

    def offline(_record):
        raise OSError("Host unreachable")

    monkeypatch.setattr(value, "_metadata", offline)
    row = value.list(record.key)[0]
    assert row["state"] == "succeeded"
    assert "Host unreachable" in row["observation_error"]


def test_corrupt_source_fails_owned_job_without_publishing_destination(durable):
    value, record, _root = durable
    source = Path(record.placements[0].root)
    next(path for path in (source / "assets").rglob("*") if path.is_file()).write_bytes(b"corrupt")
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        row = value.status(plan.job_id)
        if row["state"] == "failed":
            break
        time.sleep(0.1)
    assert row["state"] == "failed"
    assert "verification failed" in value.jobs.logs(plan.job_id, tail=100).lower()
    assert not any(p.cluster == "gpu16" for p in value.datasets.registry.get(record.key).placements)
    assert not list(
        Path(value.datasets.clusters.get("gpu16").storage.dataset_root).rglob(
            "dataset-artifact.json"
        )
    )


def test_terminal_history_uses_cached_metadata_without_host_queries(durable, monkeypatch):
    value, record, _root = durable
    plan = value.enqueue(record.key, source="local", destination="gpu16")
    finish(value, plan.job_id)
    monkeypatch.setattr(
        value, "_metadata", lambda *_: pytest.fail("No repeated terminal host queries")
    )
    row = value.list(record.key)[0]
    assert row["state"] == "succeeded" and row["progress"]["phase"] == "completed"


def test_transfer_protects_both_endpoints_even_when_job_host_is_local(durable, monkeypatch):
    import importlib

    from lambdaforge.controlplane.jobs import JobRecord, JobState

    value, record, root = durable
    job = JobRecord(
        "job-copy",
        "local",
        "local",
        None,
        JobState.RUNNING,
        (sys.executable,),
        str(root / "job-copy/work"),
        {},
        "now",
        "now",
        job_type="dataset-replication",
        metadata={
            "dataset_ids": [record.dataset_id],
            "dataset_replication": {"source": "gpu12", "destination": "gpu16"},
        },
    )
    value.jobs.store.write(job)
    module = importlib.import_module("lambdaforge.data.DatasetService")
    monkeypatch.setattr(module, "JobService", lambda *_a, **_k: value.jobs)
    assert value.datasets._active_consumers(record, "gpu12") == ("job-copy",)
    assert value.datasets._active_consumers(record, "gpu16") == ("job-copy",)
    assert value.datasets._active_consumers(record, "unrelated") == ()
    value.jobs.store.write(job.with_updates(state=JobState.SUCCEEDED))
    assert value.datasets._active_consumers(record, "gpu12") == ()
