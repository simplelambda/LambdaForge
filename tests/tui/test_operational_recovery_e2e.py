"""Two native CPU Studies through a faulty provider and the Textual recovery picker.

The provider/submit acknowledgement is simulated; Work execution, persisted Attempts,
configuration validation, recovery locking and incident records are real local implementations.
No SSH/HPC, CUDA, SMTP or application dataset is involved.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Checkbox, Input  # noqa: E402

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog  # noqa: E402
from lambdaforge.controlplane.ClusterProfile import ClusterProfile  # noqa: E402
from lambdaforge.controlplane.jobs import JobRecord, JobState  # noqa: E402
from lambdaforge.controlplane.JobService import JobService  # noqa: E402
from lambdaforge.controlplane.JobStore import JobStore  # noqa: E402
from lambdaforge.controlplane.LocalTransport import LocalTransport  # noqa: E402
from lambdaforge.controlplane.OperationalIncidents import OperationalIncidents  # noqa: E402
from lambdaforge.controlplane.OverviewService import OverviewService  # noqa: E402
from lambdaforge.controlplane.RecoveryObservation import RecoveryObservation  # noqa: E402
from lambdaforge.tui.App import LambdaForgeApp, YamlFilePicker  # noqa: E402
from lambdaforge.tui.screens.Workspace import StudyRetryConfirmation, StudyWorkspace  # noqa: E402
from lambdaforge.work import WorkConfig, WorkRunner  # noqa: E402
from lambdaforge.work.models import atomic_json  # noqa: E402
from lambdaforge.work.runner import _execute_run  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


def test_two_study_outage_relocated_yaml_recovery_preserves_native_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = ClusterProfile(
        "fault-provider", transport="ssh", host="never-connect.invalid", workspace=str(tmp_path)
    )
    store = JobStore(tmp_path / "registry")
    phase = "alive"

    class Provider:
        def state(self, key):
            if phase == "outage":
                raise TimeoutError("simulated shared provider outage")
            return JobState.UNKNOWN if phase == "lost" else JobState.RUNNING

        def details(self, key):
            return {
                "provider_observation": "lost_confirmed" if phase == "lost" else "alive",
                "heartbeat_at_utc": datetime.now(timezone.utc).isoformat(),
                "progress": {"completed": 1, "total": 10},
            }

        def inventory(self):
            if phase == "outage":
                raise TimeoutError("simulated shared provider outage")
            return tuple(
                {
                    "request": {"job_id": record.job_id},
                    "state": {
                        "job_id": record.job_id,
                        "state": self.state(record.job_id).value,
                        **self.details(record.job_id),
                    },
                }
                for record in store.records()
                if record.scheduler_id is not None
            )

    factory = SimpleNamespace(
        transport=lambda profile: LocalTransport(), scheduler=lambda *args: Provider()
    )
    jobs = JobService(ClusterCatalog({profile.name: profile}), store, factory)
    evidence = []
    for name in ("a", "b"):
        key = f"job-owner-{name}"
        work_dir = tmp_path / key / "work"
        work_dir.mkdir(parents=True)
        source = tmp_path / f"study-{name}.yaml"
        source.write_text(
            f"name: study-{name}\nrun: tests.work_cases.FixedRecoveryWork\n"
            "seeds: [4, 7, 32, 54, 65, 94, 109, 124, 142, 167]\n"
            "objective: {metric: score, mode: max}\nresources: {cpu: 1}\n"
            "execution: {failure_retries: 0}\n"
        )
        result = WorkRunner(
            dispatcher=lambda values, **kwargs: tuple(_execute_run(value) for value in values),
            execution_root=work_dir / ".lambdaforge" / "runs" / f"study-{name}",
        ).run(WorkConfig.from_yaml(source))
        atomic_json(work_dir.parent / "result.json", result.to_dict())
        at = datetime.now(timezone.utc).isoformat()
        store.write(
            JobRecord(
                key,
                profile.name,
                "local",
                key,
                JobState.RUNNING,
                (),
                str(work_dir),
                {"cpu_cores": 1},
                at,
                at,
                config_path=str(source),
                job_type="work",
                metadata={
                    "name": f"study-{name}",
                    "study_expected": True,
                    "submission_mode": "asynchronous",
                    "run_arguments": [],
                    "source_config_path": str(source),
                    "scientific_identity": f"original-{name}",
                },
            )
        )
        original = {p: p.read_bytes() for p in result.execution_dir.rglob("*") if p.is_file()}
        moved = tmp_path / f"moved-{name}.yaml"
        moved.write_text("# Relocated without scientific edits\n" + source.read_text())
        source.unlink()
        evidence.append((key, moved, result.execution_dir, original))

    assert all(
        row.to_dict()["operational"]["health"] == "healthy" for row in OverviewService._jobs(jobs)
    )
    incidents = OperationalIncidents(store)
    phase = "outage"
    for _ in range(3):
        observed = OverviewService._jobs(jobs)
        assert all(row.to_dict()["operational"]["health"] != "healthy" for row in observed)
        assert all(
            row.to_dict()["operational"]["last_confirmed_state"] == "running" for row in observed
        )
    assert len(incidents.list()) == 2
    assert all(value["cluster"] == profile.name for value in incidents.list())
    assert all(value["notification"]["status"] == "pending" for value in incidents.list())
    phase = "lost"
    assert all(
        row.to_dict()["operational"]["health"] == "lost_confirmed"
        for row in OverviewService._jobs(jobs)
    )
    assert len(incidents.list()) == 4  # original connectivity incidents + confirmed loss

    actual_popen = subprocess.Popen

    def launch(command, *args, **kwargs):
        if "lambdaforge.controlplane.SubmissionWorker" in command:
            return SimpleNamespace(pid=1234)  # only provider preparation is simulated
        return actual_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)

    class Services(FakeServices):
        def retry_preview(self, job_id, *, config=None):
            return jobs.retry_preview(job_id, config=config)

        def retry_job(self, job_id, **options):
            return jobs.retry(job_id, **options).to_dict()

        def study_workspace(self, job_id, *, latest_job_id):
            current = jobs.get(latest_job_id, refresh=False)
            return {
                "candidates": [],
                "counts": {},
                "job_state": current.state.value,
                "operational": current.to_dict()["operational"],
                "recovery": RecoveryObservation.describe(current),
            }

    async def recover() -> None:
        for key, moved, execution, original in evidence:
            app = LambdaForgeApp(Services())
            async with app.run_test(size=(150, 55)) as pilot:
                screen = StudyWorkspace(
                    {
                        "name": key,
                        "primary_job_id": key,
                        "state": "unknown",
                        "operational": jobs.get(key, refresh=False).to_dict()["operational"],
                        "study": {"candidates": [], "counts": {}},
                    },
                    app.services,
                )
                app.push_screen(screen)
                await pilot.pause()
                await pilot.click("#study-retry")
                for _ in range(200):
                    if isinstance(app.screen, StudyRetryConfirmation):
                        break
                    await pilot.pause(0.05)
                assert isinstance(app.screen, StudyRetryConfirmation)
                assert not app.screen.preview["configuration"]["selected_exists"]
                dialog = app.screen
                dialog.on_button_pressed(
                    Button.Pressed(dialog.query_one("#retry-config-browse", Button))
                )
                await pilot.pause()
                assert isinstance(app.screen, YamlFilePicker)
                app.screen.query_one("#yaml-picker-location", Input).value = str(moved)
                await pilot.press("enter")
                # File selection uses the real picker validation/dismiss path.
                if isinstance(app.screen, YamlFilePicker):
                    app.screen._select(moved)
                await pilot.pause()
                assert isinstance(app.screen, StudyRetryConfirmation)
                dialog.on_button_pressed(
                    Button.Pressed(dialog.query_one("#retry-config-validate", Button))
                )
                for _ in range(200):
                    if dialog._valid_config and not dialog._validating:
                        break
                    await pilot.pause(0.05)
                assert dialog._valid_config
                assert (dialog.preview["reuse_runs"], dialog.preview["retry_runs"]) == (9, 1)
                dialog.query_one("#retry-compatible-code", Checkbox).value = True
                await pilot.pause()
                dialog.on_button_pressed(Button.Pressed(dialog.query_one("#retry-confirm", Button)))
                for _ in range(200):
                    if app.screen is screen and screen.operation_job_id != key:
                        break
                    await pilot.pause(0.05)
                assert screen.operation_job_id != key
                new_id = screen.operation_job_id
                assert jobs.get(new_id, refresh=False).to_dict()["recovery"]["phase"] == "accepted"
                assert any(value["status"] == "open" for value in incidents.list(job_id=key))
                jobs.update_preparation(new_id, "validation")
                assert jobs.get(new_id, refresh=False).to_dict()["recovery"]["phase"] == "preparing"
                # Loading the nine reused successes must not resolve the outage/loss.
                at = datetime.now(timezone.utc).isoformat()
                store.record_observation(
                    new_id,
                    started=at,
                    finished=at,
                    source="simulated-provider",
                    state=JobState.RUNNING,
                    details={
                        "provider_observation": "alive",
                        "heartbeat_at_utc": at,
                        "progress": {"completed": 9, "total": 10},
                    },
                )
                assert any(value["status"] == "open" for value in incidents.list(job_id=key))
                restored = WorkRunner(
                    dispatcher=lambda values, **kwargs: tuple(
                        _execute_run(value) for value in values
                    )
                ).run(
                    WorkConfig.from_yaml(moved), resume_execution=execution, accept_code_change=True
                )
                assert restored.status == "succeeded"
                assert restored.execution_id == execution.name
                assert next(run for run in restored.runs if run.seed == 54).attempt_number == 2
                assert all(
                    p.read_bytes() == data
                    for p, data in original.items()
                    if p.name == "execution.json"
                    or (p.name == "result.json" and json.loads(data).get("status") == "succeeded")
                )
                at = datetime.now(timezone.utc).isoformat()
                store.record_observation(
                    new_id,
                    started=at,
                    finished=at,
                    source="simulated-provider",
                    state=JobState.RUNNING,
                    details={
                        "provider_observation": "alive",
                        "heartbeat_at_utc": at,
                        "progress": {"completed": len(restored.runs), "total": 10},
                    },
                )
                assert all(value["status"] == "resolved" for value in incidents.list(job_id=key))
                assert store.get(key).state is JobState.UNKNOWN
                assert store.get(new_id).to_dict()["recovery"]["scientific_progress_observed"]
            # Reopening does not contact the vanished historical YAML or lose the audit.
            app = LambdaForgeApp(Services())
            async with app.run_test(size=(150, 55)) as pilot:
                app.push_screen(
                    StudyWorkspace(
                        {
                            "name": key,
                            "primary_job_id": new_id,
                            "state": "running",
                            "study_job_id": key,
                            "study": {"candidates": [], "counts": {}},
                        },
                        app.services,
                    )
                )
                await pilot.pause()
                assert app.screen.operation_job_id == new_id
                assert all(value["status"] == "resolved" for value in incidents.list(job_id=key))

    asyncio.run(recover())
