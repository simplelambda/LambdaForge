"""High-level Work deletion remains preview-first and exact-root bounded."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.StorageService import StorageService
from lambdaforge.controlplane.Transport import CommandResult, Transport
from lambdaforge.controlplane.WorkService import WorkService


class FakeRemoteTransport(Transport):
    """Execute no commands while exposing the exact remote cleanup request."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        self.commands.append(tuple(command))
        return CommandResult(
            0,
            json.dumps(
                {
                    "candidates": [{"name": "job-remote", "bytes": 4}],
                    "reclaimable_bytes": 4,
                    "applied": command[-1] == "true",
                    "preserved": ["datasets"],
                }
            ),
        )

    def put(self, source: str | Path, destination: str | Path) -> None:
        raise AssertionError(f"Cleanup must not transfer data: {source} -> {destination}")


class FakeFactory:
    """Return one in-memory remote transport."""

    def __init__(self) -> None:
        self.transport_instance = FakeRemoteTransport()

    def transport(self, profile: ClusterProfile) -> Transport:
        del profile
        return self.transport_instance


def test_work_delete_previews_then_removes_only_exact_owned_job(tmp_path: Path) -> None:
    profile = ClusterProfile(
        "local",
        workspace=str(tmp_path),
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "jobs"),
            "dataset_root": str(tmp_path / "datasets"),
        },
    )
    catalog = ClusterCatalog({"local": profile})
    store = JobStore(tmp_path / "job-records")
    jobs = JobService(catalog, store)
    now = datetime.now(timezone.utc).isoformat()
    record = JobRecord(
        "job-owned",
        "local",
        "local",
        "job-owned",
        JobState.SUCCEEDED,
        ("python", "work.py"),
        str(tmp_path / "jobs" / "job-owned" / "work"),
        {},
        now,
        now,
        metadata={
            "name": "named-work",
            "scientific_identity": "sha256:" + "1" * 64,
            "scientific_revision": "1" * 12,
        },
        job_type="work",
    )
    store.write(record)
    owned = tmp_path / "jobs" / "job-owned"
    owned.mkdir(parents=True)
    (owned / "result.json").write_text("{}", encoding="utf-8")
    dataset = tmp_path / "datasets" / "keep"
    dataset.mkdir(parents=True)
    (dataset / "data").write_text("keep", encoding="utf-8")
    service = WorkService(
        catalog,
        jobs=jobs,
        storage=StorageService(catalog),
    )

    preview = service.delete("named-work")
    assert not preview["applied"]
    assert owned.is_dir()
    assert store.get("job-owned").job_id == "job-owned"

    applied = service.delete("named-work", apply=True)
    assert applied["applied"]
    assert not owned.exists()
    assert (dataset / "data").read_text(encoding="utf-8") == "keep"
    assert not (store.root / "job-owned.json").exists()
    repeated = service.delete("named-work", apply=True)
    assert repeated["applied"]
    assert repeated["already_deleted"]
    assert repeated["workspaces"] == []


def test_remote_work_delete_uses_the_bounded_storage_operation(tmp_path: Path) -> None:
    profile = ClusterProfile(
        "remote",
        transport="ssh",
        host="remote.invalid",
        workspace="/scratch/research",
    )
    catalog = ClusterCatalog({"remote": profile})
    store = JobStore(tmp_path / "job-records")
    jobs = JobService(catalog, store)
    now = datetime.now(timezone.utc).isoformat()
    store.write(
        JobRecord(
            "job-remote",
            "remote",
            "local",
            "job-remote",
            JobState.FAILED,
            ("python", "work.py"),
            "/scratch/research/.lambdaforge/jobs/job-remote/work",
            {},
            now,
            now,
            metadata={
                "name": "remote-work",
                "scientific_identity": "sha256:" + "2" * 64,
            },
            job_type="work",
        )
    )
    factory = FakeFactory()
    service = WorkService(
        catalog,
        jobs=jobs,
        storage=StorageService(catalog, factory=factory),  # type: ignore[arg-type]
    )

    preview = service.delete("remote-work")
    assert preview["applied"] is False
    assert "delete-job" in factory.transport_instance.commands[-1]
    assert factory.transport_instance.commands[-1][-1] == "false"
    assert store.get("job-remote").job_id == "job-remote"

    applied = service.delete("remote-work", apply=True)
    assert applied["applied"] is True
    assert factory.transport_instance.commands[-1][-1] == "true"
    assert not (store.root / "job-remote.json").exists()
    repeated = service.delete("remote-work", apply=True)
    assert repeated["already_deleted"]
    assert len(factory.transport_instance.commands) == 2
