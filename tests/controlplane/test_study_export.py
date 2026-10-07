from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StudyExportService import (
    _ARCHIVE_SCRIPT,
    StudyExportService,
    _extract_safe,
)
from lambdaforge.work import WorkConfig, WorkRunner


@pytest.mark.parametrize(
    ("state", "finalized"),
    (
        (JobState.SUCCEEDED, True),
        (JobState.RUNNING, False),
        (JobState.CANCELLED, True),
    ),
)
def test_study_export_downloads_bounded_job_evidence(
    tmp_path: Path, state: JobState, finalized: bool
) -> None:
    job_id = "job-20260923000000-export"
    job_root = tmp_path / "jobs" / job_id
    work_root = job_root / "work"
    work_root.mkdir(parents=True)
    config_path = work_root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "remote-study",
                "run": "tests.work_cases.SeedWork",
                "seeds": [2, 4],
                "objective": {"metric": "score", "mode": "max"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = WorkRunner().run(WorkConfig.from_yaml(config_path))
    if not finalized:
        (result.execution_dir / "result.json").unlink()
    elif state is not JobState.SUCCEEDED:
        terminal = json.loads((result.execution_dir / "result.json").read_text(encoding="utf-8"))
        terminal["status"] = state.value
        (result.execution_dir / "result.json").write_text(
            json.dumps(terminal), encoding="utf-8"
        )
    unrelated = work_root / ".lambdaforge" / "runs" / "other" / "execution-old"
    unrelated.mkdir(parents=True)
    (unrelated / "secret.txt").write_text("not this experiment", encoding="utf-8")
    (job_root / "stdout.log").write_text("scientific output\n", encoding="utf-8")
    study_root = job_root / "study"
    study_root.mkdir()
    (study_root / "controller-history.jsonl").write_text(
        '{"action":"FINISH","reason":"candidate budget consumed"}\n', encoding="utf-8"
    )
    record = JobRecord(
        job_id=job_id,
        cluster="local",
        scheduler="local",
        scheduler_id="1",
        state=state,
        command=("lf", "run", "config.yaml"),
        work_dir=str(work_root),
        resources={},
        created_at_utc="2026-09-23T00:00:00+00:00",
        updated_at_utc="2026-09-23T01:00:00+00:00",
    )

    class Jobs:
        def get(self, selected: str, *, refresh: bool = True) -> JobRecord:
            assert selected == job_id
            return record

        def study(self, selected: str) -> dict[str, str]:
            assert selected == job_id
            return {"execution_id": result.execution_id}

    class Works:
        def show(self, selector: str) -> SimpleNamespace:
            assert selector == "remote-study"
            return SimpleNamespace(work_id="work-export", job_ids=(job_id,))

    class Catalog:
        def get(self, cluster: str) -> SimpleNamespace:
            assert cluster == "local"
            return ClusterProfile("local", python=sys.executable)

    class Factory:
        def transport(self, profile: object) -> LocalTransport:
            return LocalTransport()

    service = StudyExportService(
        Catalog(),  # type: ignore[arg-type]
        jobs=Jobs(),  # type: ignore[arg-type]
        works=Works(),  # type: ignore[arg-type]
        factory=Factory(),  # type: ignore[arg-type]
    )
    progress: list[dict[str, object]] = []
    exported = service.export(
        "remote-study", tmp_path / "exports", progress=lambda value: progress.append(dict(value))
    )

    package = Path(exported["path"])
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert exported["cluster"] == "local"
    assert exported["captured_state"] == state.value
    assert exported["export_kind"] == (
        "final" if finalized and state is JobState.SUCCEEDED else "snapshot"
    )
    assert manifest["execution_id"] == result.execution_id
    assert manifest["status"] == state.value
    assert manifest["finalized"] is finalized
    assert (package / "control-plane" / "stdout.log").read_text() == "scientific output\n"
    assert (package / "study" / "controller-history.jsonl").is_file()
    assert (package / "execution" / "result.json").is_file() is finalized
    assert (package / "execution" / "execution.json").is_file()
    assert not (package / "execution" / "secret.txt").exists()
    assert not list(job_root.glob(".lambdaforge-export-*.zip"))
    assert [item["phase"] for item in progress] == [
        "resolving",
        "compressing",
        "downloading",
        "extracting",
        "finalizing",
        "complete",
    ]
    assert progress[-1]["terminal"] is True
    assert not list((tmp_path / "exports").glob(".lambdaforge-export-transfer-*"))


def test_study_export_captures_pre_execution_attempt(tmp_path: Path) -> None:
    job_id = "job-20260923000000-preparing"
    job_root = tmp_path / "jobs" / job_id
    work_root = job_root / "work"
    work_root.mkdir(parents=True)
    (work_root / "config.yaml").write_text(
        "name: preparing-study\nrun: tests.work_cases.SeedWork\n",
        encoding="utf-8",
    )
    (job_root / "lifecycle.jsonl").write_text(
        '{"state":"preparing"}\n', encoding="utf-8"
    )
    record = JobRecord(
        job_id=job_id,
        cluster="local",
        scheduler="local",
        scheduler_id="1",
        state=JobState.PREPARING,
        command=("lf", "run", "config.yaml"),
        work_dir=str(work_root),
        resources={},
        created_at_utc="2026-09-23T00:00:00+00:00",
        updated_at_utc="2026-09-23T00:01:00+00:00",
    )

    class Jobs:
        def get(self, selected: str, *, refresh: bool = True) -> JobRecord:
            assert selected == job_id
            return record

        def study(self, selected: str) -> dict[str, str]:
            raise KeyError(selected)

    class Works:
        def show(self, selector: str) -> SimpleNamespace:
            assert selector == "preparing-study"
            return SimpleNamespace(
                work_id="work-preparing", name="preparing-study", job_ids=(job_id,)
            )

    class Catalog:
        def get(self, cluster: str) -> SimpleNamespace:
            return ClusterProfile("local", python=sys.executable)

    class Factory:
        def transport(self, profile: object) -> LocalTransport:
            return LocalTransport()

    service = StudyExportService(
        Catalog(),  # type: ignore[arg-type]
        jobs=Jobs(),  # type: ignore[arg-type]
        works=Works(),  # type: ignore[arg-type]
        factory=Factory(),  # type: ignore[arg-type]
    )
    exported = service.export("preparing-study", tmp_path / "exports")

    package = Path(exported["path"])
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert exported["captured_state"] == "preparing"
    assert exported["export_kind"] == "snapshot"
    assert manifest["execution_id"] is None
    assert manifest["finalized"] is False
    assert manifest["status"] == "preparing"
    assert (package / "submitted-configuration" / "config.yaml").is_file()
    assert (package / "control-plane" / "lifecycle.jsonl").is_file()
    assert not list(job_root.glob(".lambdaforge-export-*.zip"))


def test_study_export_rejects_archive_traversal(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("unsafe", encoding="utf-8")
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.write(source, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_safe(archive, tmp_path / "destination")


@pytest.mark.parametrize("profile", ("default", "full"))
def test_remote_archive_profiles_account_for_compaction(
    tmp_path: Path, profile: str
) -> None:
    job = tmp_path / "job-20260926000000-profile"
    execution = job / "work" / ".lambdaforge" / "runs" / "study" / "execution-profile"
    run = execution / "runs" / "run-1"
    control = execution / "hpo-control"
    run.mkdir(parents=True)
    control.mkdir()
    (execution / "execution.json").write_text(
        json.dumps({"execution_id": "execution-profile"}), encoding="utf-8"
    )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps({"step": index, "value": index}) + "\n" for index in range(5000)),
        encoding="utf-8",
    )
    (control / "scheduler-trace.jsonl").write_text(
        '{"event":"WAIT","reason":"BLOCK_BARRIER"}\n' * 20,
        encoding="utf-8",
    )
    archive = tmp_path / f"{profile}.zip"

    subprocess.run(
        (
            sys.executable,
            "-c",
            _ARCHIVE_SCRIPT,
            str(job),
            str(archive),
            "execution-profile",
            job.name,
            profile,
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        profile_record = json.loads(package.read("export-profile.json"))
        metrics_name = "runs/study/execution-profile/runs/run-1/metrics.jsonl"
        trace_name = "runs/study/execution-profile/hpo-control/scheduler-trace.jsonl"
        metric_rows = package.read(metrics_name).splitlines()
    if profile == "default":
        assert len(metric_rows) == 4096
        assert trace_name not in names
        assert len(profile_record["omitted"]) == 2
        assert all(value["sha256"] for value in profile_record["omitted"])
        assert "study/resource-telemetry-summary.json" in names
    else:
        assert len(metric_rows) == 5000
        assert trace_name in names
        assert profile_record["omitted"] == []
    assert not (tmp_path / "escaped.txt").exists()
