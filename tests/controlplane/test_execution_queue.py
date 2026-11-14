"""Persistent FIFO regressions: temporary metadata and tiny local commands, never clusters."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionQueue import ExecutionQueue
from lambdaforge.controlplane.ExecutionQueueService import ExecutionQueueService
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.SubmissionService import SubmissionService
from lambdaforge.execution.ResourceRequest import ResourceRequest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Direct execution queues require POSIX")


def profile(root):
    return ClusterProfile(
        "local",
        workspace=str(root),
        python=sys.executable,
        storage={"safety": {"min_free_percent": 0}},
    )


def test_fifo_ten_entries_pending_unknown_terminal_and_cancel(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    before = set(tmp_path.rglob("*"))
    assert queue.snapshot()["entries"] == []
    assert set(tmp_path.rglob("*")) == before
    for i in range(10):
        queue.register(f"job-{i}", ())
    queue.register("job-5", ())
    assert len(queue.snapshot()["entries"]) == 10

    def observe(_):
        return {"state": "running", "released": False}

    assert queue.waiting_on("job-0", observe) == []
    assert queue.waiting_on("job-1", observe) == ["job-0"]
    assert queue.waiting_on("job-1", lambda _: {"state": "unknown"}) == ["job-0"]
    # Terminal status alone is not permission to overlap cleanup/owned descendants.
    assert queue.waiting_on("job-1", lambda _: {"state": "failed"}) == ["job-0"]
    assert queue.waiting_on("job-1", lambda _: {"state": "failed", "released": True}) == []
    queue.mark("job-1", "aborted")
    assert queue.waiting_on("job-2", observe) == []
    with pytest.raises(InterruptedError):
        queue.waiting_on("job-1", observe)
    # Persisted release receipts survive deletion of old terminal Job workspaces.
    again = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    assert again.waiting_on("job-2", lambda _: pytest.fail("Read settled predecessor")) == []


def test_concurrent_registration_is_atomic_without_duplicate_tickets(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda i: queue.register(f"job-{i % 10}", ()), range(40)))
    assert len(queue.snapshot()["entries"]) == 10
    assert len({v["job_id"] for v in queue.snapshot()["entries"]}) == 10


def test_external_preparation_failure_is_releasable_but_dispatched_is_not(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    queue.register("job-waiting", ["job-ordinary"])
    assert queue.waiting_on("job-waiting", lambda _: {"state": "unknown"}) == ["job-ordinary"]
    queue.abort_preparation("job-ordinary")
    assert queue.waiting_on("job-waiting", lambda _: pytest.fail("Aborted predecessor")) == []
    queue.mark("job-waiting", "dispatched")
    with pytest.raises(ValueError, match="native provider cancellation"):
        queue.abort_preparation("job-waiting")


@pytest.mark.parametrize("invalid", ["../job-a", "job-a/../../escape", "not-job", "job-x\n"])
def test_invalid_ids_never_create_queue(tmp_path, invalid):
    with pytest.raises(ValueError, match="identity"):
        ExecutionQueue(tmp_path / "state", tmp_path / "jobs").register(invalid, ())
    assert list(tmp_path.iterdir()) == []


def test_symlink_and_corrupt_metadata_fail_closed(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    queue.register("job-a", ())
    (queue.root / "queue.json").write_text('{"version":99,"entries":[]}')
    with pytest.raises(ValueError, match="Corrupt"):
        queue.waiting_on("job-a", lambda _: {})
    (tmp_path / "alias").symlink_to(tmp_path / "state", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        ExecutionQueue(tmp_path / "alias", tmp_path / "jobs").register("job-b", ())


def test_lost_manifest_does_not_silently_restart_fifo(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    queue.register("job-a", ())
    (queue.root / "queue.json").unlink()
    with pytest.raises(ValueError, match="Missing execution queue manifest"):
        queue.register("job-b", ())
    with pytest.raises(ValueError, match="Missing execution queue manifest"):
        queue.snapshot()


def test_releasing_untracked_job_does_not_create_a_queue(tmp_path):
    ExecutionQueue(tmp_path / "state", tmp_path / "jobs").release("job-a")
    assert list(tmp_path.iterdir()) == []


def test_target_service_rejects_unsupported_scheduler_without_writes(tmp_path):
    slurm = ClusterProfile("batch", scheduler="slurm", workspace=str(tmp_path))
    with pytest.raises(ValueError, match="not SLURM/Fleet"):
        ExecutionQueueService(ClusterCatalog({"batch": slurm})).snapshot("batch")
    assert list(tmp_path.iterdir()) == []


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for native fixture supervisor")


def test_native_detached_supervisors_wait_fifo_without_resource_reservations(tmp_path):
    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    service = ExecutionQueueService(catalog)
    scheduler = ProcessScheduler(LocalTransport(), selected)
    resource = ResourceRequest(cpu_cores=1)
    done = tmp_path / "release-a"
    started = tmp_path / "started-a"
    sequence = tmp_path / "sequence.jsonl"
    children = []

    def launch(job, command, admission=None):
        children.append(job)
        return scheduler.submit(
            command, resource, work_dir=tmp_path, job_id=job, execution_queue=admission
        )

    def state(job):
        path = Path(selected.storage.job_root) / job / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    try:
        launch(
            "job-a",
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys;"
                "pathlib.Path(sys.argv[1]).touch();"
                "\nwhile not pathlib.Path(sys.argv[2]).exists():time.sleep(.05)",
                str(started),
                str(done),
            ],
        )
        wait_for(started.exists)
        b = service.register("local", "job-b", ())
        c = service.register("local", "job-c", ())
        # C finishes preparation first; it must still wait for B's target-side ticket.
        command = (
            "import pathlib,sys,time,json;"
            "p=pathlib.Path(sys.argv[1]);name=sys.argv[2];"
            "\nwith p.open('a') as f:f.write(json.dumps([name,'start'])+'\\n')\n"
            "time.sleep(.2)\n"
            "with p.open('a') as f:f.write(json.dumps([name,'end'])+'\\n')\n"
            "sys.exit(int(sys.argv[3]))"
        )
        launch("job-c", [sys.executable, "-c", command, str(sequence), "c", "0"], c)
        launch("job-b", [sys.executable, "-c", command, str(sequence), "b", "2"], b)
        wait_for(lambda: state("job-b").get("state") == "queued")
        wait_for(lambda: state("job-c").get("state") == "queued")
        with pytest.raises(RuntimeError, match="dispatch is already reserved"):
            scheduler.submit(
                [sys.executable, "-c", "raise AssertionError('Duplicate')"],
                resource,
                work_dir=tmp_path,
                job_id="job-b",
                execution_queue=b,
            )
        assert state("job-b")["state"] == "queued"
        assert not sequence.exists()
        for job in ("job-b", "job-c"):
            row = state(job)
            assert not row.get("allocated_cpus") and not row.get("allocated_gpus")
            assert not row.get("storage_admission") and not row.get("process_identity")
        done.touch()
        wait_for(lambda: state("job-c").get("state") == "succeeded", timeout=35)
        assert state("job-b")["state"] == "failed"
        assert [json.loads(v) for v in sequence.read_text().splitlines()] == [
            ["b", "start"],
            ["b", "end"],
            ["c", "start"],
            ["c", "end"],
        ]
        wait_for(
            lambda: all(v["phase"] == "released" for v in service.snapshot("local")["entries"])
        )
    finally:
        done.touch()
        for job in children:
            row = state(job)
            if row.get("state") not in {"succeeded", "failed", "cancelled", "timeout"}:
                scheduler.cancel(job)


def test_queued_study_keeps_native_seed_parallelism_before_successor_starts(tmp_path, monkeypatch):
    """Two real CPU Runs overlap inside one queued Job; its successor still waits."""
    import yaml

    pytest.importorskip("torch", reason="The importable native Work fixtures use PyTorch")
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(repository / "src"), str(repository))))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selected = profile(tmp_path)
    service = ExecutionQueueService(ClusterCatalog({"local": selected}))
    scheduler = ProcessScheduler(LocalTransport(), selected)
    resources = ResourceRequest(cpu_cores=2)
    markers = tmp_path / "markers"
    markers.mkdir()
    config = tmp_path / "parallel.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "run": "tests.work_cases.QueuedParallelWork",
                "with": {"markers": str(markers)},
                "seeds": [1, 2],
                "resources": {"cpu": 2},
                "execution": {"max_parallel": 2},
            }
        )
    )
    first = service.register("local", "job-parallel-study", ())
    second = service.register("local", "job-next-study", ())

    def state(job):
        path = Path(selected.storage.job_root) / job / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    try:
        scheduler.submit(
            [sys.executable, "-m", "lambdaforge", "run", str(config)],
            resources,
            work_dir=tmp_path,
            job_id="job-parallel-study",
            execution_queue=first,
        )
        scheduler.submit(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; p=pathlib.Path(sys.argv[1]);"
                "assert all((p/f'seed-{s}.finished').is_file() for s in (1,2));"
                "(p/'successor.started').touch()",
                str(markers),
            ],
            ResourceRequest(cpu_cores=1),
            work_dir=tmp_path,
            job_id="job-next-study",
            execution_queue=second,
        )
        wait_for(
            lambda: state("job-next-study").get("state") in {"succeeded", "failed"}, timeout=90
        )
        stderr = Path(selected.storage.job_root) / "job-parallel-study" / "stderr.log"
        assert state("job-parallel-study")["state"] == "succeeded", stderr.read_text()
        assert state("job-next-study")["state"] == "succeeded"
        assert state("job-parallel-study")["requested_resources"] == resources.to_dict()
        starts = [float((markers / f"seed-{seed}.started").read_text()) for seed in (1, 2)]
        ends = [float((markers / f"seed-{seed}.finished").read_text()) for seed in (1, 2)]
        assert max(starts) <= min(ends)
        assert (markers / "successor.started").is_file()
    finally:
        for job in ("job-parallel-study", "job-next-study"):
            if state(job).get("state") not in {"succeeded", "failed", "cancelled", "timeout"}:
                scheduler.cancel(job)


def test_native_queue_cancellation_does_not_cancel_predecessor(tmp_path):
    selected = profile(tmp_path)
    queue = ExecutionQueue(selected.storage.state_root, selected.storage.job_root)
    queue.register("job-b", ("job-not-prepared",))
    scheduler = ProcessScheduler(LocalTransport(), selected)
    sentinel = tmp_path / "must-not-run"
    scheduler.submit(
        [
            sys.executable,
            "-c",
            "import pathlib,sys;pathlib.Path(sys.argv[1]).touch()",
            str(sentinel),
        ],
        ResourceRequest(cpu_cores=1),
        work_dir=tmp_path,
        job_id="job-b",
        execution_queue={"version": 1, "job_id": "job-b"},
    )
    try:
        wait_for(lambda: scheduler.state("job-b") is JobState.QUEUED)
        scheduler.cancel("job-b")
        wait_for(lambda: scheduler.state("job-b") is JobState.CANCELLED)
        assert not sentinel.exists()
        assert queue.snapshot()["entries"][0]["phase"] == "external"
    finally:
        scheduler.cancel("job-b")


def test_enqueue_tracks_ticket_and_cancel_before_handoff(tmp_path, monkeypatch):
    from types import SimpleNamespace

    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    jobs = JobService(catalog, JobStore(tmp_path / "catalog"))
    config = tmp_path / "work.yaml"
    config.write_text("run: tests.work_cases.Producer\n")
    # Only the detached controller launcher is replaced; registration uses real local transport.
    import importlib

    module = importlib.import_module("lambdaforge.controlplane.SubmissionService")

    class Launcher:
        DEVNULL = -3

        @staticmethod
        def Popen(*args, **kwargs):
            return SimpleNamespace(pid=12345)

    monkeypatch.setattr(module, "subprocess", Launcher)
    handle = SubmissionService(catalog, jobs).enqueue(config, cluster="local", execution_queue=True)
    row = jobs.get(handle.job_id, refresh=False)
    assert row.metadata["execution_queue"]["job_id"] == handle.job_id
    request = json.loads(
        (jobs.store.root / "submissions" / handle.job_id / "request.json").read_text()
    )
    assert request["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert row.metadata["pinned_scientific_identity"] == row.metadata["scientific_identity"]
    assert jobs.cancel(handle.job_id).state is JobState.CANCELLED
    assert ExecutionQueueService(catalog).snapshot("local")["entries"][-1]["phase"] == "aborted"


def test_queue_retains_predecessor_workspaces_until_release(tmp_path):
    from lambdaforge.controlplane.StorageOperations import StorageOperations

    selected = profile(tmp_path)
    roots = StorageOperations._roots(selected.storage.to_dict())
    previous = roots["job_workspaces"] / "job-a"
    previous.mkdir(parents=True)
    (previous / "request.json").write_text(json.dumps({"command": [sys.executable]}))
    (previous / "state.json").write_text(json.dumps({"state": "succeeded"}))
    queue = ExecutionQueue(selected.storage.state_root, selected.storage.job_root)
    queue.register("job-b", ["job-a"])
    inventory = StorageOperations.environment_references(roots)
    assert {"job-a", "job-b"} <= inventory["protected_jobs"]
    with pytest.raises(ValueError, match="waiting execution queue"):
        StorageOperations.delete_job(selected.storage.to_dict(), "job-a", apply=False)
    queue.release("job-a")
    queue.release("job-b")
    assert "job-a" not in StorageOperations.environment_references(roots)["protected_jobs"]
    assert StorageOperations.delete_job(selected.storage.to_dict(), "job-a", apply=False)


def test_duplicate_dispatch_is_rejected_without_releasing_original_ticket(tmp_path):
    queue = ExecutionQueue(tmp_path / "state", tmp_path / "jobs")
    queue.register("job-a", ())
    queue.mark("job-a", "dispatched")
    with pytest.raises(ValueError, match="already has a native supervisor"):
        queue.mark("job-a", "dispatched")
    assert queue.snapshot()["entries"][0]["phase"] == "dispatched"


def test_old_runtime_cannot_ignore_queue_and_start_science(tmp_path):
    from lambdaforge.controlplane.CommandResult import CommandResult

    class OldRuntime(LocalTransport):
        def run(self, command, **kwargs):
            if any("callable(getattr(ProcessSupervisor," in value for value in command):
                return CommandResult(2, stderr="Queue method is not installed")
            return super().run(command, **kwargs)

    selected = profile(tmp_path)
    with pytest.raises(RuntimeError, match="cannot honor sequential execution"):
        ProcessScheduler(OldRuntime(), selected).submit(
            [sys.executable, "-c", "raise AssertionError('Must never launch')"],
            ResourceRequest(cpu_cores=1),
            work_dir=tmp_path,
            job_id="job-old-runtime",
            execution_queue={"version": 1, "job_id": "job-old-runtime"},
        )
    assert not (Path(selected.storage.job_root) / "job-old-runtime").exists()


def test_cli_queue_flag_routes_to_native_submission_and_snapshot(tmp_path, monkeypatch, capsys):
    from lambdaforge.cli.CommandLineInterface import CommandLineInterface
    from lambdaforge.controlplane.jobs import JobHandle

    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    monkeypatch.setattr(ClusterCatalog, "load", classmethod(lambda *a, **k: catalog))
    calls = []

    def enqueue(self, config, **kwargs):
        calls.append(kwargs)
        return JobHandle("job-cli-fixture", "local", JobState.PREPARING)

    monkeypatch.setattr(SubmissionService, "enqueue", enqueue)
    config = tmp_path / "work.yaml"
    config.write_text("run: tests.work_cases.Producer\n")
    assert CommandLineInterface.main(["run", str(config), "--queue", "--json"]) == 0
    assert calls[0]["execution_queue"] is True
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-cli-fixture"
    assert CommandLineInterface.main(["queue", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == []
    assert not (Path(selected.storage.state_root) / "execution-queue").exists()


def test_shared_storage_keeps_cluster_queues_independent(tmp_path):
    from dataclasses import replace

    twelve = replace(profile(tmp_path), name="cluster12")
    sixteen = replace(profile(tmp_path), name="cluster16")
    service = ExecutionQueueService(ClusterCatalog({"cluster12": twelve, "cluster16": sixteen}))
    service.register("cluster12", "job-twelve", ())
    old = Path(twelve.storage.job_root) / "job-twelve"
    old.mkdir(parents=True)
    (old / "state.json").write_text(
        json.dumps({"job_id": "job-twelve", "cluster": "cluster12", "state": "running"})
    )
    ticket = service.register("cluster16", "job-sixteen", ())
    assert ticket["ticket"]["blockers"] == []
    queue = ExecutionQueue(sixteen.storage.state_root, sixteen.storage.job_root, "cluster16")
    assert queue.waiting_on("job-sixteen", lambda _: pytest.fail("Cross-cluster predecessor")) == []
    assert [v["job_id"] for v in service.snapshot("cluster12")["entries"]] == ["job-twelve"]
    assert [v["job_id"] for v in service.snapshot("cluster16")["entries"]] == ["job-sixteen"]
    assert len(ExecutionQueue.for_storage(twelve.storage.state_root, twelve.storage.job_root)) == 2


def test_controller_request_write_failure_aborts_ticket(tmp_path, monkeypatch):
    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    jobs = JobService(catalog, JobStore(tmp_path / "catalog"))
    config = tmp_path / "work.yaml"
    config.write_text("run: tests.work_cases.Producer\n")

    def fail(*args, **kwargs):
        raise OSError("Synthetic controller disk failure")

    monkeypatch.setattr(SubmissionService, "_launch_preparation", staticmethod(fail))
    with pytest.raises(OSError, match="Synthetic controller disk failure"):
        SubmissionService(catalog, jobs).enqueue(config, cluster="local", execution_queue=True)
    assert jobs.store.records()[0].state is JobState.FAILED
    assert ExecutionQueueService(catalog).snapshot("local")["entries"][0]["phase"] == "aborted"


def test_provider_failure_aborts_unstarted_ticket(tmp_path, monkeypatch):
    from lambdaforge.diagnostics import LambdaForgeError

    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    jobs = JobService(catalog, JobStore(tmp_path / "catalog"))
    admission = ExecutionQueueService(catalog).register("local", "job-provider-failure", ())
    resource = ResourceRequest(cpu_cores=1)
    jobs.reserve(
        cluster="local",
        resources=resource,
        config_path=tmp_path / "work.yaml",
        job_id="job-provider-failure",
        metadata={"execution_queue": admission},
    )

    class Refused:
        def submit(
            self, command, resources, *, work_dir, dry_run=False, job_id=None, execution_queue=None
        ):
            raise RuntimeError("Synthetic provider refusal before launch")

    monkeypatch.setattr(jobs.factory, "scheduler", lambda *a, **k: Refused())
    with pytest.raises(LambdaForgeError):
        jobs.submit(
            [sys.executable, "-c", "raise AssertionError('Do not launch')"],
            cluster="local",
            resources=resource,
            work_dir=tmp_path,
            job_id="job-provider-failure",
        )
    assert ExecutionQueueService(catalog).snapshot("local")["entries"][0]["phase"] == "aborted"


def test_queue_metadata_failure_does_not_rewrite_completed_science(tmp_path):
    selected = profile(tmp_path)
    catalog = ClusterCatalog({"local": selected})
    admission = ExecutionQueueService(catalog).register("local", "job-corrupt-release", ())
    manifest = (
        ExecutionQueue(selected.storage.state_root, selected.storage.job_root).root / "queue.json"
    )
    scheduler = ProcessScheduler(LocalTransport(), selected)
    scheduler.submit(
        [
            sys.executable,
            "-c",
            "import pathlib,sys;pathlib.Path(sys.argv[1]).unlink()",
            str(manifest),
        ],
        ResourceRequest(cpu_cores=1),
        work_dir=tmp_path,
        job_id="job-corrupt-release",
        execution_queue=admission,
    )
    path = Path(selected.storage.job_root) / "job-corrupt-release" / "state.json"
    state = wait_for(
        lambda: (
            (value if value.get("queue_release_warning") else None)
            if (value := json.loads(path.read_text()))
            else None
        )
    )
    assert state["state"] == "succeeded" and state["scientific_cleanup_complete"]


def test_unprepared_queue_ticket_is_not_a_missing_scientific_recovery(tmp_path, monkeypatch):
    from lambdaforge.controlplane.StorageOperations import StorageOperations

    selected = profile(tmp_path)
    queue = ExecutionQueue(selected.storage.state_root, selected.storage.job_root)
    queue.register("job-not-prepared", ())
    monkeypatch.setattr(
        StorageOperations,
        "_process_references",
        staticmethod(lambda roots: {"environments": set(), "jobs": set(), "inspectable": True}),
    )
    references = StorageOperations.environment_references(
        StorageOperations._roots(selected.storage.to_dict())
    )
    assert "job-not-prepared" in references["protected_jobs"]
    assert "*" not in references["environments"]
    assert not any(
        row["reason"] == "recovery-owner-request-missing" for row in references["issues"]
    )


def test_queue_view_reports_corrupt_metadata_without_loading_scientific_files(tmp_path):
    selected = profile(tmp_path)
    service = ExecutionQueueService(ClusterCatalog({"local": selected}))
    service.register("local", "job-observed", ())
    job = Path(selected.storage.job_root) / "job-observed"
    job.mkdir(parents=True)
    (job / "state.json").write_text("corrupt")
    # Reading this file would block; snapshots must only read operational metadata.
    os.mkfifo(job / "study.json")
    row = service.snapshot("local")["entries"][0]
    assert row["last_provider_state"] == "unavailable" and row["observation_error"]
