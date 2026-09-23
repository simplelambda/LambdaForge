from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StudyExportService import StudyExportService, _extract_safe
from lambdaforge.work import WorkConfig, WorkRunner


def test_study_export_downloads_bounded_job_evidence(tmp_path: Path) -> None:
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
        state=JobState.SUCCEEDED,
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
            return SimpleNamespace(python=sys.executable)

    class Factory:
        def transport(self, profile: object) -> LocalTransport:
            return LocalTransport()

    service = StudyExportService(
        Catalog(),  # type: ignore[arg-type]
        jobs=Jobs(),  # type: ignore[arg-type]
        works=Works(),  # type: ignore[arg-type]
        factory=Factory(),  # type: ignore[arg-type]
    )
    exported = service.export("remote-study", tmp_path / "exports")

    package = Path(exported["path"])
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert exported["cluster"] == "local"
    assert manifest["execution_id"] == result.execution_id
    assert (package / "control-plane" / "stdout.log").read_text() == "scientific output\n"
    assert (package / "study" / "controller-history.jsonl").is_file()
    assert (package / "execution" / "result.json").is_file()
    assert not (package / "execution" / "secret.txt").exists()
    assert not list(job_root.glob(".lambdaforge-export-*.tar"))


def test_study_export_rejects_archive_traversal(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("unsafe", encoding="utf-8")
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as package:
        package.add(source, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract_safe(archive, tmp_path / "destination")
    assert not (tmp_path / "escaped.txt").exists()
