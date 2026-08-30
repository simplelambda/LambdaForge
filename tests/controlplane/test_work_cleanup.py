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
from lambdaforge.controlplane.ResearchWork import aggregate_research_work
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


def test_research_work_exposes_numbered_attempt_history_for_machine_clients() -> None:
    records = tuple(
        JobRecord(
            f"job-{number}",
            "local",
            "local",
            f"provider-{number}",
            state,
            ("python", "work.py"),
            "/tmp/work",
            {},
            f"2026-01-0{number}T00:00:00+00:00",
            f"2026-01-0{number}T00:01:00+00:00",
            retry_of="job-1" if number == 2 else None,
            metadata={"name": "study", "scientific_identity": "sha256:same"},
            job_type="work",
        )
        for number, state in ((1, JobState.FAILED), (2, JobState.RUNNING))
    )

    work = aggregate_research_work(records)[0].to_dict()

    assert work["attempts"] == 2
    assert [item["label"] for item in work["attempt_history"]] == [
        "Attempt 1",
        "Attempt 2",
    ]
    assert work["attempt_history"][1]["job_id"] == "job-2"


def test_research_work_ignores_legacy_single_run_telemetry_for_normal_work() -> None:
    now = datetime.now(timezone.utc).isoformat()
    normal = JobRecord(
        "job-normal",
        "local",
        "local",
        "provider-normal",
        JobState.RUNNING,
        ("python", "work.py"),
        "/tmp/work",
        {},
        now,
        now,
        metadata={
            "name": "preprocessing",
            "scientific_identity": "sha256:normal",
            "study_expected": False,
            "remote_state": {
                "study": {
                    "study_telemetry_version": 1,
                    "planned_runs": 1,
                    "candidates": [{"trial": 1, "runs": [{"seed": None}]}],
                }
            },
        },
        job_type="work",
    )
    legacy_study = normal.with_updates(
        job_id="job-study",
        scheduler_id="provider-study",
        metadata={
            "name": "training",
            "scientific_identity": "sha256:study",
            "remote_state": {
                "study": {
                    "study_telemetry_version": 1,
                    "planned_runs": 2,
                    "candidates": [
                        {"trial": 1, "runs": [{"seed": 4}, {"seed": 7}]}
                    ],
                }
            },
        },
    )

    works = {work.name: work.to_dict() for work in aggregate_research_work((normal, legacy_study))}

    assert works["preprocessing"]["study_expected"] is False
    assert works["preprocessing"]["study"] is None
    assert works["training"]["study_expected"] is True
    assert works["training"]["study"] is not None


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


def test_local_legacy_relative_job_root_recovers_real_terminal_state(tmp_path: Path) -> None:
    project = tmp_path / "consumer"
    experiments = project / "experiments"
    experiments.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='consumer'\nversion='1'\n")
    config = experiments / "dna_design.yaml"
    config.write_text("name: dna-design\nrun: consumer.Design\n", encoding="utf-8")
    profile = ClusterProfile("local")
    catalog = ClusterCatalog({"local": profile})
    store = JobStore(tmp_path / "history")
    now = datetime.now(timezone.utc).isoformat()
    job_id = "job-legacy-relative"
    relative_work = Path(".lambdaforge/remote/.lambdaforge/jobs") / job_id / "work"
    job_dir = experiments / relative_work.parent
    job_dir.mkdir(parents=True)
    (job_dir / "state.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "state": "failed",
                "message": "Requested 36 CPU cores but only 16 are available.",
            }
        ),
        encoding="utf-8",
    )
    store.write(
        JobRecord(
            job_id,
            "local",
            "local",
            job_id,
            JobState.UNKNOWN,
            ("python", "-m", "lambdaforge"),
            str(relative_work),
            {},
            now,
            now,
            config_path=str(config),
            metadata={"source_config_path": str(config), "unreachable_error": "old path"},
            job_type="work",
        )
    )

    recovered = JobService(catalog, store).get(job_id)

    assert recovered.state is JobState.FAILED
    assert recovered.metadata["remote_state"]["message"].startswith("Requested 36 CPU")
    assert "unreachable_error" not in recovered.metadata


def test_local_submission_storage_is_anchored_to_consumer_project(tmp_path: Path) -> None:
    project = tmp_path / "consumer"
    experiments = project / "experiments"
    experiments.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='consumer'\nversion='1'\n")
    config = experiments / "work.yaml"
    config.write_text("name: work\nrun: consumer.Work\n", encoding="utf-8")

    resolved = JobService._submission_profile(
        ClusterProfile("local"),
        config_path=str(config),
        work_dir=experiments,
    )

    assert Path(resolved.workspace) == project / ".lambdaforge/remote"
    assert Path(resolved.storage.run_root) == (
        project / ".lambdaforge/remote/.lambdaforge/jobs"
    )


def test_clear_history_removes_terminal_jobs_and_preserves_active(tmp_path: Path) -> None:
    profile = ClusterProfile(
        "local",
        workspace=str(tmp_path),
        storage={
            "state_root": str(tmp_path / "state"),
            "cache_root": str(tmp_path / "cache"),
            "run_root": str(tmp_path / "jobs"),
        },
    )
    catalog = ClusterCatalog({"local": profile})
    store = JobStore(tmp_path / "history")
    jobs = JobService(catalog, store)
    now = datetime.now(timezone.utc).isoformat()
    for job_id, state in (
        ("job-succeeded", JobState.SUCCEEDED),
        ("job-failed", JobState.FAILED),
        ("job-running", JobState.RUNNING),
    ):
        work = tmp_path / "jobs" / job_id / "work"
        work.mkdir(parents=True)
        store.write(
            JobRecord(
                job_id,
                "local",
                "local",
                None if state is JobState.RUNNING else job_id,
                state,
                ("python", "work.py"),
                str(work),
                {},
                now,
                now,
                metadata={"name": job_id},
            )
        )
        submission = store.root / "submissions" / job_id
        submission.mkdir(parents=True)
        (submission / "request.json").write_text("{}", encoding="utf-8")
    service = WorkService(catalog, jobs=jobs, storage=StorageService(catalog))

    preview = service.clear_history()
    assert set(preview["terminal_jobs"]) == {"job-succeeded", "job-failed"}
    assert preview["active_jobs_preserved"] == ["job-running"]
    assert (tmp_path / "jobs/job-succeeded").is_dir()

    applied = service.clear_history(apply=True)

    assert set(applied["deleted_jobs"]) == {"job-succeeded", "job-failed"}
    assert store.get("job-running",).state is JobState.RUNNING
    assert (tmp_path / "jobs/job-running").is_dir()
    assert not (tmp_path / "jobs/job-succeeded").exists()
    assert not (store.root / "submissions/job-succeeded").exists()
