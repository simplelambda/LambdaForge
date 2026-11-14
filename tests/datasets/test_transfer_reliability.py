"""Route consent, peer-host execution and honest operational observation without real SSH."""

from __future__ import annotations

import io
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.data.DatasetTransferRouting import classify_route_failure
from tests.datasets.test_compressed_replication import transfer  # noqa: F401
from tests.datasets.test_durable_replication import durable, finish  # noqa: F401


@pytest.fixture
def peers(request, monkeypatch):
    value, record, root = request.getfixturevalue("durable")
    service = value.datasets
    service.replicate(record.key, source="local", destination="gpu12", apply=True)
    for name in ("gpu12", "gpu16"):
        service.clusters._profiles[name] = replace(
            service.clusters.get(name), transport="ssh", host=f"{name}.invalid"
        )
    transport = LocalTransport()
    monkeypatch.setattr(service.factory, "transport", lambda _profile: transport)
    monkeypatch.setattr(service, "_site_ssh", lambda _profile, command: command)
    return value, record, root, transport


@pytest.mark.parametrize(
    "message,category",
    [
        ("Permission denied (publickey)", "authentication"),
        ("Host key verification failed", "host-key"),
        ("Connection timed out", "timeout"),
        ("Connection refused", "network"),
        ("PermissionError: directory not writable", "permissions"),
        ("ModuleNotFoundError: No module named lambdaforge", "tools"),
        ("EOFError", "indeterminate"),
    ],
)
def test_route_failure_categories_do_not_guess_unknown_causes(message, category):
    assert classify_route_failure(message) == category


@pytest.mark.parametrize("pull", [False, True])
def test_direct_directions_use_appropriate_host_no_controller_stream(peers, monkeypatch, pull):
    value, record, _root, transport = peers
    service = value.datasets

    def site(profile, command):
        if pull and profile.name == "gpu16":
            return (
                sys.executable,
                "-c",
                "import sys;sys.stderr.write('Permission denied (publickey)');sys.exit(255)",
            )
        return command

    monkeypatch.setattr(service, "_site_ssh", site)
    monkeypatch.setattr(
        transport, "stream", lambda *_: pytest.fail("Direct bytes must not pass controller")
    )
    plan = value.preview(record.key, source="gpu12", destination="gpu16", route="direct")
    expected = "direct-ssh-pull" if pull else "direct-ssh"
    assert plan.transfer_route == expected and not plan.requires_controller_online
    assert plan.execution_host == ("gpu16" if pull else "gpu12")
    if pull:
        assert plan.route_diagnostics[0]["category"] == "authentication"
        assert plan.route_diagnostics[1]["category"] == "ready"
    submitted = value.enqueue(
        record.key, source="gpu12", destination="gpu16", route="direct", approved_route=expected
    )
    row = finish(value, submitted.job_id)
    assert row["host"] == plan.execution_host and row["route"] == expected
    assert row["progress"]["phase"] == "completed"
    assert row["last_byte_progress_at_utc"]
    assert service.resolve_placement(record.key, "gpu16").physically_available


def deny_sites(service, monkeypatch):
    monkeypatch.setattr(
        service,
        "_site_ssh",
        lambda _p, _c: (
            sys.executable,
            "-c",
            "import sys;sys.stderr.write('Host key verification failed');sys.exit(255)",
        ),
    )


def test_auto_relay_is_only_a_proposal_and_domain_requires_consent(peers, monkeypatch):
    value, record, _root, _transport = peers
    deny_sites(value.datasets, monkeypatch)
    plan = value.preview(record.key, source="gpu12", destination="gpu16")
    assert plan.relay_consent_required and plan.requires_controller_online
    assert plan.data_path == ("gpu12", "controller", "gpu16")
    assert len(plan.route_diagnostics) == 2
    assert {d["category"] for d in plan.route_diagnostics} == {"host-key"}
    assert plan.estimated_bytes == record.placements[0].size_bytes
    with pytest.raises(ValueError, match="not approved"):
        value.enqueue(record.key, source="gpu12", destination="gpu16")
    assert not value.jobs.list(refresh=False)
    with pytest.raises(Exception, match="both directions"):
        value.enqueue(record.key, source="gpu12", destination="gpu16", route="direct")
    from lambdaforge.controlplane.jobs import JobState
    from lambdaforge.controlplane.SchedulerSubmission import SchedulerSubmission

    def submitted(command, resources, *, work_dir, job_id, **_kwargs):
        return SchedulerSubmission(job_id, JobState.QUEUED, work_dir=work_dir)

    monkeypatch.setattr(
        value.datasets.factory, "scheduler", lambda *_: SimpleNamespace(submit=submitted)
    )
    applied = value.enqueue(
        record.key,
        source="gpu12",
        destination="gpu16",
        route="relay",
        approved_route="controller-stream",
        approved_diagnostics=plan.route_diagnostics,
    )
    job = value.jobs.get(applied.job_id, refresh=False)
    assert job.metadata["dataset_replication"]["route_diagnostics"] == list(plan.route_diagnostics)


@pytest.mark.parametrize(
    "interactive,answer,accepted", [(False, "", False), (True, "n", False), (True, "yes", True)]
)
def test_cli_relay_consent_before_apply(peers, monkeypatch, capsys, interactive, answer, accepted):
    import importlib

    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    value, record, _root, _transport = peers
    deny_sites(value.datasets, monkeypatch)
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _path: value.datasets.clusters)
    monkeypatch.setattr(module, "DatasetService", lambda **_k: value.datasets)
    calls = []

    def enqueue(selector, **options):
        calls.append(options)
        return value.preview(
            selector,
            **{
                k: v
                for k, v in options.items()
                if k not in {"approved_route", "approved_diagnostics"}
            },
        )

    monkeypatch.setattr(value.datasets, "enqueue_replication", enqueue)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: interactive))
    monkeypatch.setattr("builtins.input", lambda _text: answer)
    args = SimpleNamespace(
        dataset_command="replicate",
        clusters=None,
        dataset=record.key,
        source="gpu12",
        destination="gpu16",
        route="auto",
        apply=True,
        json=False,
    )
    if accepted:
        assert module.DatasetCommands.run(args) == 0
        assert calls[0]["route"] == "relay" and calls[0]["approved_route"] == "controller-stream"
    else:
        with pytest.raises(ValueError, match="not approved|explicit acceptance"):
            module.DatasetCommands.run(args)
        assert not calls
    output = capsys.readouterr().out
    assert "controller" in output and "original/uncompressed" in output and "Host key" in output


def test_route_change_after_preview_never_falls_back_to_relay(peers, monkeypatch):
    value, record, _root, transport = peers
    plan = value.preview(record.key, source="gpu12", destination="gpu16")
    deny_sites(value.datasets, monkeypatch)
    monkeypatch.setattr(
        transport, "stream", lambda *_: pytest.fail("No byte transfer after route change")
    )
    with pytest.raises(ValueError, match="route changed"):
        value.enqueue(
            record.key, source="gpu12", destination="gpu16", approved_route=plan.transfer_route
        )
    assert not value.jobs.list(refresh=False)


def test_probe_checks_endpoints_without_creating_publication(peers):
    value, record, root, _transport = peers
    before = set(root.rglob("*"))
    plan = value.preview(record.key, source="gpu12", destination="gpu16")
    assert plan.transfer_route == "direct-ssh"
    assert set(root.rglob("*")) == before


def test_unobservable_or_stale_process_is_not_displayed_as_confirmed_running(request, monkeypatch):
    value, record, root = request.getfixturevalue("durable")
    from lambdaforge.controlplane.jobs import JobRecord, JobState

    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    job = JobRecord(
        "job-stale-copy",
        "local",
        "local",
        None,
        JobState.RUNNING,
        (sys.executable,),
        str(root / "work"),
        {},
        old,
        old,
        job_type="dataset-replication",
        metadata={
            "dataset_replication": {
                "dataset": record.key,
                "source": "gpu12",
                "destination": "gpu16",
            },
            "operational_observation": {
                "last_confirmed_state": "running",
                "last_success_at_utc": old,
                "last_confirmed_at_utc": old,
            },
        },
    )
    value.jobs.store.write(job)
    row = value.status(job.job_id, refresh=False)
    assert row["state"] == "running" and not row["state_confirmed"]
    assert "unconfirmed/stale" in row["display_state"] and row["observation_age_seconds"] >= 3600
    monkeypatch.setattr(
        value, "_metadata", lambda *_: (_ for _ in ()).throw(OSError("Host unreachable"))
    )
    row = value.list(record.key)[0]
    assert not row["state_confirmed"] and row["state"] == "running"
    assert "Host unreachable" in row["observation_error"]


def test_live_verification_and_byte_timestamps_are_separate(request):
    value, record, root = request.getfixturevalue("durable")
    from lambdaforge.controlplane.jobs import JobRecord, JobState

    now = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    job = JobRecord(
        "job-verifying",
        "local",
        "local",
        None,
        JobState.RUNNING,
        (sys.executable,),
        str(root / "work"),
        {},
        now,
        now,
        job_type="dataset-replication",
        metadata={
            "dataset_replication": {"dataset": record.key},
            "operational_observation": {
                "last_confirmed_state": "running",
                "last_success_at_utc": now,
                "last_confirmed_at_utc": now,
                "heartbeat_at_utc": now,
            },
            "dataset_replication_outcome": {
                "progress": {
                    "phase": "verify-destination",
                    "compressed_bytes": 100,
                    "last_byte_progress_at_utc": old,
                }
            },
        },
    )
    value.jobs.store.write(job)
    row = value.status(job.job_id, refresh=False)
    assert row["state_confirmed"] and row["activity"] == "verifying"
    assert row["byte_progress_age_seconds"] >= 120 and row["observation_age_seconds"] < 5
    assert "percent" not in row


def test_receiver_eof_keeps_context_and_never_publishes(peers, monkeypatch):
    from contextlib import contextmanager

    from lambdaforge.controlplane.BinaryCommand import BinaryCommand

    value, record, _root, transport = peers
    original = transport.stream

    class BrokenInput(io.BytesIO):
        def write(self, _data):
            raise EOFError("primary receiver EOF")

    @contextmanager
    def streams(command):
        if "receive" in command:
            sink = BinaryCommand(
                BrokenInput(),
                io.BytesIO(),
                io.BytesIO(b"receiver: permission denied"),
                SimpleNamespace(poll=lambda: 23),
            )
            yield sink
        else:
            with original(command) as sender:
                yield sender

    monkeypatch.setattr(transport, "stream", streams)
    with pytest.raises(RuntimeError, match="receiver.*gpu16.*EOFError") as caught:
        value.datasets.replicate(
            record.key, source="gpu12", destination="gpu16", route="relay", apply=True
        )
    assert isinstance(caught.value.__cause__, EOFError)
    assert "permission denied" in str(caught.value) and "23" in str(caught.value)
    assert all(p.cluster != "gpu16" for p in value.datasets.registry.get(record.key).placements)


def test_pull_sender_exit_keeps_peer_diagnostic_without_publication(peers):
    from lambdaforge.data.DatasetRegistry import DatasetRegistry
    from lambdaforge.data.DatasetTransfer import pull_copy

    value, record, _root, _transport = peers
    target = value.datasets.clusters.get("gpu16")
    options = {
        "root": target.storage.dataset_root,
        "record": record.to_dict(),
        "cluster": target.name,
        "registry": str(Path(target.storage.state_root) / "datasets.json"),
        "storage": target.storage.to_dict(),
    }
    with pytest.raises(RuntimeError, match="sender.*source permissions denied") as caught:
        pull_copy(
            options,
            [
                sys.executable,
                "-c",
                "import sys;sys.stderr.write('source permissions denied');sys.exit(17)",
            ],
        )
    assert isinstance(caught.value.__cause__, ValueError)
    assert not DatasetRegistry(options["registry"]).records()
    assert not list(Path(options["root"]).glob("**/.replication-*.tmp"))


@pytest.mark.parametrize("pull", [False, True])
def test_direct_progress_observes_receiver_verification_and_publication(peers, pull):
    from lambdaforge.data import DatasetTransfer as helper

    value, record, _root, _transport = peers
    target = value.datasets.clusters.get("gpu16")
    root = value.datasets.resolve_placement(record.key, "gpu12").placement.root
    options = {
        "root": target.storage.dataset_root,
        "record": record.to_dict(),
        "cluster": target.name,
        "registry": str(Path(target.storage.state_root) / "datasets.json"),
        "storage": target.storage.to_dict(),
        "report_phases": True,
    }
    import json

    code = Path(helper.__file__).read_text()
    phases = []

    def progress(phase, _message, **_values):
        phases.append(phase)

    if pull:
        result = helper.pull_copy(
            options, [sys.executable, "-c", code, "pack", root, record.dataset_id], progress
        )
    else:
        result = helper.push_copy(
            Path(root),
            record.dataset_id,
            [sys.executable, "-c", code, "receive", json.dumps(options)],
            progress,
        )
    assert result["verified"]
    assert "transfer" in phases
    assert phases[-2:] == ["verify-destination", "publication"]


def test_terminal_native_process_observation_propagates_to_history_and_jobs(request):
    from lambdaforge.controlplane.jobs import JobState

    value, record, _root = request.getfixturevalue("durable")
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
    native = value.jobs.get(plan.job_id, refresh=False)
    # Fixture-only cached running observation; actual owned supervisor already says failed.
    value.jobs.store.write(native.with_updates(state=JobState.RUNNING))
    rows = value.list(record.key)
    assert rows[0]["state"] == rows[0]["display_state"] == "failed"
    assert rows[0]["state_confirmed"] and rows[0]["error"]
    assert value.jobs.list(refresh=False)[0].state is JobState.FAILED
    assert value.jobs.get(plan.job_id).state is JobState.FAILED


def test_effective_probe_rejects_missing_source_or_unusable_target(tmp_path):
    from lambdaforge.data.DatasetTransfer import probe

    with pytest.raises(PermissionError, match="not readable"):
        probe({"role": "source", "root": str(tmp_path / "missing")})
    file = tmp_path / "not-a-directory"
    file.write_text("occupied")
    with pytest.raises(PermissionError, match="not writable"):
        probe({"role": "destination", "root": str(file)})


def test_json_apply_auto_never_prompts_or_copies_without_relay_acceptance(peers, monkeypatch):
    import importlib

    module = importlib.import_module("lambdaforge.cli.DatasetCommands")
    value, record, _root, _transport = peers
    deny_sites(value.datasets, monkeypatch)
    monkeypatch.setattr(module.ClusterCatalog, "load", lambda _path: value.datasets.clusters)
    monkeypatch.setattr(module, "DatasetService", lambda **_kwargs: value.datasets)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("JSON mode must never prompt"))
    args = SimpleNamespace(
        dataset_command="replicate",
        clusters=None,
        dataset=record.key,
        source="gpu12",
        destination="gpu16",
        route="auto",
        apply=True,
        json=True,
    )
    with pytest.raises(ValueError, match="explicit acceptance"):
        module.DatasetCommands.run(args)
    assert not value.jobs.list(refresh=False)
