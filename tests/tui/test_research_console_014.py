from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

textual = pytest.importorskip("textual")
from textual.widgets import TabbedContent  # noqa: E402

from lambdaforge.cli.CommandLineInterface import CommandLineInterface  # noqa: E402
from lambdaforge.controlplane.ClusterProfile import ClusterProfile  # noqa: E402
from lambdaforge.tui.App import (  # noqa: E402
    ClusterEditor,
    LambdaForgeApp,
    WorkLaunchDialog,
    YamlFilePicker,
)
from lambdaforge.tui.models import CLI_PARITY, CONSOLE_ACTIONS, PALETTE_ACTIONS  # noqa: E402
from lambdaforge.tui.RecentWorkStore import RecentWorkStore  # noqa: E402
from lambdaforge.tui.screens.Workspace import (  # noqa: E402
    ClusterWorkspace,
    DatasetWorkspace,
    ExactConfirmation,
    HpoActionWorkspace,
    HpoParameterWorkspace,
    ResultWorkspace,
    SeedWorkspace,
    StudyWorkspace,
    TrialWorkspace,
    WorkWorkspace,
)
from lambdaforge.tui.services import ConsoleServices  # noqa: E402
from lambdaforge.tui.viewmodels import metric_display_name, objective_display_name  # noqa: E402
from lambdaforge.tui.widgets import MetricDashboard, ResourceDashboard  # noqa: E402


class FakeServices:
    def __init__(self, snapshot=None, *, fail: bool = False, recent=()):
        self.snapshot = snapshot or {"work": {"items": []}, "clusters": []}
        self.fail = fail
        self.calls: list[tuple[str, object]] = []
        self.recent = tuple(recent)

    def overview_snapshot(self):
        self.calls.append(("overview_snapshot", None))
        if self.fail:
            raise RuntimeError("provider temporarily unreachable")
        return self.snapshot

    def cluster_rows(self):
        return []

    def cluster_names(self):
        return ("local", "gpu12")

    def dataset_rows(self):
        return []

    def result_rows(self):
        return []

    def dataset_members(self, selector, *, limit=200):
        self.calls.append(("dataset_members", selector))
        return {
            "offset": 0,
            "returned": 2,
            "members": [
                {
                    "id": "protein-1",
                    "partitions": {"split": "train"},
                    "targets": {"label": 1},
                    "metadata": {"source": "test"},
                    "assets": {"structure": {"path": "1.cif"}},
                },
                {
                    "id": "protein-2",
                    "partitions": {"split": "validation"},
                    "targets": {"label": 0},
                    "metadata": {},
                    "assets": {},
                },
            ],
        }

    def dataset_summary(self, selector):
        self.calls.append(("dataset_summary", selector))
        return {
            "member_count": 2,
            "partitions": {"split": {"train": 1, "validation": 1}},
            "partition_targets": {
                "split": {
                    "train": {"label": {"1": 1}},
                    "validation": {"label": {"0": 1}},
                }
            },
        }

    def delete_dataset(self, selector, *, apply=False):
        self.calls.append(("delete_dataset", (selector, apply)))
        return {
            "dataset": selector,
            "safe": True,
            "applied": apply,
            "placements": [
                {
                    "cluster": "gpu12",
                    "placement_state": "registered_but_missing",
                    "action": "CLEAN_STALE_REGISTRATION",
                }
            ],
        }

    def analyze(self, selector, *, recompute=False):
        return {"source": {"status": "final"}, "selector": selector}

    def report(self, selector, output: Path):
        return output

    def validate_work(self, config):
        self.calls.append(("validate_work", config))
        return {"valid": True, "summary": "valid"}

    def explain_work(self, config):
        self.calls.append(("explain_work", config))
        return {"name": "training", "run": "project.Training"}

    def submit_work(self, config, cluster):
        self.calls.append(("submit_work", (config, cluster)))
        return {"job_id": "job-1"}

    def cancel_work(self, selector):
        self.calls.append(("cancel_work", selector))
        return {
            "work_id": selector,
            "cancelled_jobs": [{"job_id": "job-study", "state": "cancelled"}],
            "reconciled_cancelled_jobs": [],
            "status": "cancelled",
        }

    def delete_work(self, selector, *, apply=False):
        self.calls.append(("delete_work", (selector, apply)))
        return {
            "work_id": selector,
            "applied": apply,
            "will_remove": ["terminal Attempt workspaces", "project history record"],
        }

    def recent_work_configs(self, *, limit=12):
        self.calls.append(("recent_work_configs", limit))
        return self.recent[:limit]

    def remember_work_config(self, config, *, name=None):
        self.calls.append(("remember_work_config", (config, name)))

    def cluster_detail(self, name):
        self.calls.append(("cluster_detail", name))
        return {
            "profile": {
                "name": name,
                "host": "gpu.example",
                "scheduler": "slurm",
                "transport": "ssh",
                "workspace": "/remote/lf",
                "environment": "managed",
                "gpu_access": {"mode": "scheduler"},
            },
            "authentication_status": "keyring configured",
            "resources": {"online": True, "gpus": 2},
            "storage": {"free_bytes": 1000},
        }

    def doctor(self, name):
        self.calls.append(("doctor", name))
        return {"cluster": name, "status": "ok"}

    def bootstrap(self, name, *, project=None, dry_run=False, progress=None):
        self.calls.append(("bootstrap", (name, dry_run)))
        if progress is not None:
            progress("Resolving the managed runtime.")
        return {"cluster": name, "dry_run": dry_run}

    def set_cluster_credential(self, name, secret):
        self.calls.append(("credential", (name, secret)))
        return {"status": "stored"}

    def remove_cluster(self, name, *, apply=False):
        self.calls.append(("remove_cluster", (name, apply)))
        return {"cluster": name, "applied": apply, "will_remove": ["profile"]}

    def delete_result(self, selector, *, apply=False):
        self.calls.append(("delete_result", (selector, apply)))
        return {
            "execution": selector,
            "applied": apply,
            "will_remove": ["result envelope"],
        }

    def result_analysis(self, selector):
        return {
            "winner": {},
            "seed_analysis": {"status": "insufficient_evidence"},
            "surrogate": {"quality": "unavailable"},
            "parameter_importance": {},
            "findings": [],
        }

    def study(self, job_id):
        for item in self.snapshot.get("work", {}).get("items", []):
            if item.get("primary_job_id") == job_id:
                value = item["study"]
                if value is None:
                    raise KeyError(f"{job_id} has no Study telemetry")
                return value
        raise KeyError(job_id)

    def study_run(self, job_id, run_key, *, tail=2_000, curve_points=200):
        assert tail > 0 and curve_points >= 10
        curves = {
            "val_auprc": [{"step": step, "value": 0.5 + step / 1_000} for step in range(1, 101)],
            "val_auroc": [{"step": step, "value": 0.6 + step / 2_000} for step in range(1, 101)],
            "val_balanced_accuracy": [
                {"step": step, "value": 0.55 + step / 3_000} for step in range(1, 101)
            ],
            "val_mcc": [{"step": step, "value": 0.1 + step / 2_000} for step in range(1, 101)],
            "train_loss": [{"step": step, "value": 1.0 - step / 1_000} for step in range(1, 101)],
        }
        return {
            "key": run_key,
            "trial": 17,
            "seed": 54,
            "state": "running",
            "gpu_index": 1,
            "current_observed_objective": 0.6,
            "best_observed_objective": 0.663,
            "final_objective": None,
            "current_step": 100,
            "best_step": 63,
            "duration_seconds": 120,
            "objective_status": {"status": "partial"},
            "latest_metrics": {name: values[-1]["value"] for name, values in curves.items()},
            "curves": curves,
            "chart_filter": {},
            "resources": {"usage": {"gpu_mem_mb": 1024, "ram_mb": 2048}},
            "artifacts": [
                {
                    "name": "protein-view",
                    "role": "visualization",
                    "media_type": "text/html",
                    "size_bytes": 4096,
                    "sha256": "a" * 64,
                    "path": "/remote/run/artifacts/protein-view/1abc.html",
                    "managed_path": "/remote/run/artifacts/protein-view/1abc.html",
                    "published_path": None,
                    "retention": "managed-internal",
                    "metadata": {"protein": "1ABC"},
                },
                {
                    "name": "predictions",
                    "role": "predictions",
                    "media_type": "application/octet-stream",
                    "size_bytes": 8192,
                    "sha256": "b" * 64,
                    "path": "/project/data/predictions.npz",
                    "managed_path": None,
                    "published_path": "/project/data/predictions.npz",
                    "retention": "published-only",
                    "metadata": {},
                },
            ],
            "log": "epoch 100 complete",
            "paths": {},
        }


def test_console_starts_and_navigates() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(100, 32)) as pilot:
            assert app.query_one("#overview")
            app.show_screen("studies")
            await pilot.pause(0.05)
            assert app.query_one("#studies").display is True
            assert app.query_one("#overview").display is False
            app.show_screen("results")
            await pilot.pause(0.05)
            assert app.query_one("#results").display is True
            app.exit()

    asyncio.run(exercise())


def test_cluster_editor_round_trips_complete_profile_including_gpu_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ClusterProfile.from_mapping(
        "gpu-site",
        {
            "transport": "ssh",
            "scheduler": "local",
            "host": "gpu.example.org",
            "user": "researcher",
            "port": 2222,
            "auth": {
                "mode": "password",
                "credential": "keyring:cluster/gpu-site/researcher@gpu.example.org",
            },
            "known_hosts": "/secure/known_hosts",
            "connection": {
                "connect_timeout": 12,
                "auth_timeout": 18,
                "banner_timeout": 21,
                "keepalive": 25,
                "multiplex": False,
                "persist": 90,
                "command_timeout": 7200,
            },
            "workspace": "/srv/researcher",
            "storage": {
                "state_root": "/srv/state",
                "cache_root": "/srv/cache",
                "run_root": "/srv/runs",
                "dataset_root": "/datasets",
                "cache_max_size": "500GiB",
                "cache_max_age": "30d",
            },
            "python": {
                "strategy": "managed",
                "executable": "python3",
                "version": "3.11",
                "allow_managed_install": True,
            },
            "environment": "managed",
            "wheelhouse": "/srv/wheels",
            "pytorch": {"channel": "cu128", "require_cuda": True},
            "project_module": "project.works",
            "project_root": "/srv/project",
            "data_environment": "shared-data",
            "gpu_access": {
                "mode": "command",
                "command_prefix": ["gpu", "exec"],
                "claim_command": ["gpu", "claim", "--numgpus", "{gpu_count}"],
                "release_command": ["gpu", "release"],
            },
            "ssh_options": ["-J", "gateway.example.org"],
            "command_prefix": ["env", "SITE_PROFILE=research"],
        },
    ).to_dict()

    saved: dict[str, object] = {}

    def persist(path: Path, value: ClusterProfile) -> Path:
        saved.update(path=path, profile=value)
        return path

    monkeypatch.setattr("lambdaforge.tui.App.ClusterCatalog.add", persist)

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        app.services.catalog = SimpleNamespace(
            source=lambda _name: Path("/tmp/lambdaforge-clusters.yaml")
        )
        async with app.run_test(size=(130, 48)) as pilot:
            editor = ClusterEditor(app.services, profile)
            app.push_screen(editor)
            await pilot.pause(0.1)
            name, descriptor = editor._descriptor()
            restored = ClusterProfile.from_mapping(name, descriptor)

            assert restored.host == "gpu.example.org"
            assert restored.port == 2222
            assert restored.auth.credential == profile["auth"]["credential"]
            assert restored.connection.command_timeout == 7200
            assert restored.connection.multiplex is False
            assert restored.storage is not None
            assert restored.storage.cache_max_bytes == 500 * 1024**3
            assert restored.storage.cache_max_age_seconds == 30 * 86400
            assert restored.runtime_policy.strategy == "managed"
            assert restored.runtime_policy.version == "3.11"
            assert restored.pytorch.channel == "cu128"
            assert restored.pytorch.require_cuda is True
            assert restored.gpu_access.command_prefix == ("gpu", "exec")
            assert restored.gpu_access.claim_command[-1] == "{gpu_count}"
            assert restored.gpu_access.release_command == ("gpu", "release")
            assert restored.ssh_options == ("-J", "gateway.example.org")
            assert restored.command_prefix == ("env", "SITE_PROFILE=research")

            # Switching only the backend would make the persistent direct-host claim unsafe.
            # Validation must keep the editor open rather than partially writing the profile.
            editor.query_one("#cluster-scheduler").value = "slurm"
            await pilot.click("#save")
            await pilot.pause(0.05)
            assert app.screen is editor
            assert "cannot use a persistent" in str(
                editor.query_one("#cluster-editor-error").render()
            )

            # A corrected profile is persisted through the same catalog boundary.
            editor.query_one("#cluster-scheduler").value = "local"
            editor.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="save")))
            await pilot.pause(0.05)
            assert "profile" in saved, str(editor.query_one("#cluster-editor-error").render())
            persisted = saved["profile"]
            assert isinstance(persisted, ClusterProfile)
            assert persisted.gpu_access.command_prefix == ("gpu", "exec")
            assert persisted.connection.command_timeout == 7200
            assert saved["path"] == Path("/tmp/lambdaforge-clusters.yaml")

    asyncio.run(exercise())


def test_hidden_root_screens_do_not_start_remote_loaders() -> None:
    async def exercise() -> None:
        services = FakeServices()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.pause(0.15)
            assert services.calls.count(("overview_snapshot", None)) == 1
            assert not any(
                name in {"cluster_rows", "dataset_rows", "result_rows"}
                for name, _ in services.calls
            )

            # A pushed workspace leaves the root widgets mounted.  Its periodic
            # overview timer must nevertheless stay quiet so it cannot contend with
            # bootstrap or another explicit remote operation in that workspace.
            app.push_screen(ClusterWorkspace("gpu12", services))
            await pilot.pause(0.1)
            calls = services.calls.count(("overview_snapshot", None))
            app.query_one("#overview")._refresh_visible()
            await pilot.pause(0.05)
            assert services.calls.count(("overview_snapshot", None)) == calls

    asyncio.run(exercise())


def test_overview_separates_domains_and_keeps_failed_study_drilldown() -> None:
    snapshot = {
        "clusters": [
            {
                "cluster": "gpu12",
                "online": True,
                "scheduler": "local",
                "observed": {
                    "cpu_total": 32,
                    "cpu_load": 25.0,
                    "ram_total_bytes": 64 * 1024**3,
                    "ram_available_bytes": 48 * 1024**3,
                    "gpus": [
                        {
                            "utilization_percent": 50.0,
                            "memory_total_bytes": 80 * 1024**3,
                        }
                    ],
                },
                "personal": {
                    "active_jobs": 1,
                    "requested": {"cpu_cores": 4, "ram_bytes": 8 * 1024**3, "gpu_count": 1},
                    "observed": {
                        "cpu_percent": 100,
                        "rss_bytes": 2 * 1024**3,
                        "gpu_memory_bytes": 4 * 1024**3,
                        "job_count": 1,
                    },
                },
            }
        ],
        "work": {
            "items": [
                {
                    "work_id": "work-preprocess",
                    "name": "preprocess",
                    "cluster": "gpu12",
                    "state": "succeeded",
                    "study_expected": False,
                    "progress": {"completed": 10, "total": 10, "unit": "items"},
                    "attempt_history": [],
                },
                {
                    "work_id": "work-study",
                    "name": "training",
                    "primary_job_id": "job-study",
                    "study_job_id": "job-study",
                    "cluster": "gpu12",
                    "state": "failed",
                    "study_expected": True,
                    "study": {
                        "strategy": "adaptive",
                        "objective": {"metric": "val_auprc", "mode": "max"},
                        "planned_runs": 4,
                        "counts": {
                            "candidates": 2,
                            "completed_runs": 1,
                            "active_runs": 0,
                            "queued_runs": 0,
                            "pruned_runs": 1,
                        },
                        "candidates": [],
                    },
                },
            ]
        },
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause(0.15)
            assert app.query_one("#overview-clusters").row_count == 1
            assert app.query_one("#overview-work").row_count == 1
            studies = app.query_one("#overview-studies")
            assert studies.row_count == 1
            assert app.query_one("#nav-exit")

            app.query_one("#overview-clusters").focus()
            await pilot.pause(0.05)
            detail = str(app.query_one("#overview-detail").render())
            assert "Resource plots  CPU" in detail
            assert "Mine requested" in detail

            studies.focus()
            await pilot.press("enter")
            assert isinstance(app.screen, StudyWorkspace)
            assert "FAILED" in str(app.screen.query_one("#study-header").render())

    asyncio.run(exercise())


def test_failed_study_without_embedded_telemetry_opens_without_crashing() -> None:
    class MissingStudyServices(FakeServices):
        def study(self, job_id):
            raise KeyError(f"{job_id} has no reachable Study telemetry")

        def work_logs(self, job_id, *, tail=2_000):
            assert job_id == "job-failed-study"
            assert tail == 2_000
            self.calls.append(("work_logs", job_id))
            return {
                "text": "[preparation] failed\nRuntimeError: site GPU launcher rejected the request"
            }

    work = {
        "work_id": "work-failed-study",
        "name": "wisdom-v1",
        "primary_job_id": "job-failed-study",
        "cluster": "gpu12",
        "state": "failed",
        "study_expected": True,
        "study": None,
    }
    services = MissingStudyServices({"work": {"items": [work]}, "clusters": []})

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 34)) as pilot:
            app.show_screen("studies")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert isinstance(app.screen, StudyWorkspace)
            assert "wisdom-v1" in str(app.screen.query_one("#study-header").render())
            assert "FAILED" in str(app.screen.query_one("#study-header").render())
            tabs = app.screen.query_one("#study-tabs", TabbedContent)
            assert tabs.display is True
            assert tabs.active == "study-logs"
            assert ("work_logs", "job-failed-study") in services.calls
            assert "KeyError" not in str(app.screen.query_one("#study-header").render())

    asyncio.run(exercise())


def test_running_study_without_telemetry_can_cancel_its_semantic_work() -> None:
    class MissingStudyServices(FakeServices):
        def study(self, job_id):
            raise KeyError(f"{job_id} has not published Study telemetry")

    work = {
        "work_id": "work-preparing-study",
        "name": "wisdom-v1",
        "primary_job_id": "job-study",
        "cluster": "gpu12",
        "state": "running",
        "study_expected": True,
        "study": None,
    }
    services = MissingStudyServices({"work": {"items": [work]}, "clusters": []})

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 34)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause(0.1)
            cancel = app.screen.query_one("#study-cancel")
            assert cancel.disabled is False
            await pilot.click("#study-cancel")
            assert isinstance(app.screen, ExactConfirmation)
            await pilot.click("#exact-apply")
            await pilot.pause(0.15)
            assert ("cancel_work", "work-preparing-study") in services.calls
            status = str(app.screen.query_one("#study-action-status").render())
            assert "Cancellation complete" in status

    asyncio.run(exercise())


def test_terminal_study_delete_uses_exact_work_preview_and_confirmation() -> None:
    work = {
        "work_id": "work-terminal-study",
        "name": "wisdom-v1",
        "primary_job_id": "job-terminal-study",
        "cluster": "gpu12",
        "state": "failed",
        "study_expected": True,
        "study": None,
    }
    services = FakeServices({"work": {"items": [work]}, "clusters": []})

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 34)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause(0.1)
            delete = app.screen.query_one("#study-delete")
            assert delete.disabled is False
            await pilot.click("#study-delete")
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExactConfirmation)
            assert ("delete_work", ("work-terminal-study", False)) in services.calls
            await pilot.click("#exact-apply")
            await pilot.pause(0.15)
            assert ("delete_work", ("work-terminal-study", True)) in services.calls
            assert not isinstance(app.screen, StudyWorkspace)

    asyncio.run(exercise())


def test_overview_refresh_preserves_selected_cluster_and_preview() -> None:
    def cluster(name: str, cpu: float) -> dict[str, object]:
        return {
            "cluster": name,
            "online": True,
            "scheduler": "local",
            "observed": {
                "cpu_total": 8,
                "cpu_load": cpu,
                "ram_total_bytes": 16 * 1024**3,
                "ram_available_bytes": 8 * 1024**3,
                "gpus": [],
            },
            "personal": {"active_jobs": 0, "requested": {}, "observed": {}},
        }

    services = FakeServices(
        {
            "clusters": [cluster("local", 10.0), cluster("gpu12", 20.0)],
            "work": {"items": []},
        }
    )

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause(0.15)
            table = app.query_one("#overview-clusters")
            table.focus()
            table.move_cursor(row=1)
            await pilot.pause(0.05)
            assert "gpu12" in str(app.query_one("#overview-detail").render())

            services.snapshot = {
                "clusters": [cluster("local", 11.0), cluster("gpu12", 25.0)],
                "work": {"items": []},
            }
            app.query_one("#overview").reload()
            await pilot.pause(0.15)
            assert table.cursor_row == 1
            assert "gpu12" in str(app.query_one("#overview-detail").render())

    asyncio.run(exercise())


def test_run_dialog_has_one_submit_action_that_validates_before_submission(tmp_path: Path) -> None:
    services = FakeServices()
    config = tmp_path / "train.yaml"
    config.write_text("name: training\nrun: project.Training\n", encoding="utf-8")

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.click("#nav-run")
            assert isinstance(app.screen, WorkLaunchDialog)
            submit = app.screen.query_one("#launch-submit")
            assert submit.disabled
            app.screen.query_one("#launch-config").value = str(config)
            await pilot.pause()
            assert not submit.disabled
            assert len(app.screen.query("#launch-submit")) == 1
            assert len(app.screen.query("#launch-preview")) == 0
            assert len(app.screen.query("#launch-run-recent")) == 0
            await pilot.click("#launch-submit")
            await pilot.pause(0.15)
            assert ("validate_work", config) in services.calls
            assert ("explain_work", config) in services.calls
            assert ("submit_work", (config, "local")) in services.calls
            assert app.screen is app.screen_stack[0]

    asyncio.run(exercise())


def test_run_dialog_submits_even_if_the_validated_preview_cannot_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = FakeServices()
    config = tmp_path / "study.yaml"
    config.write_text("name: study\nrun: project.Study\n", encoding="utf-8")

    def broken_preview(*args, **kwargs):
        raise TypeError("unsupported display value")

    monkeypatch.setattr("lambdaforge.tui.App.structured_text", broken_preview)

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.click("#nav-run")
            app.screen.query_one("#launch-config").value = str(config)
            await pilot.click("#launch-submit")
            await pilot.pause(0.2)
            assert ("submit_work", (config, "local")) in services.calls
            assert app.screen is app.screen_stack[0]

    asyncio.run(exercise())


def test_run_dialog_treats_authored_explanation_as_literal_text(tmp_path: Path) -> None:
    class MarkupLikeExplanationServices(FakeServices):
        def explain_work(self, config):
            self.calls.append(("explain_work", config))
            return {
                "name": "study",
                "run_doc": "Use values in [0, 1] and omit ``score`` when unavailable.",
            }

    services = MarkupLikeExplanationServices()
    config = tmp_path / "study.yaml"
    config.write_text("name: study\nrun: project.Study\n", encoding="utf-8")

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.click("#nav-run")
            app.screen.query_one("#launch-config").value = str(config)
            await pilot.click("#launch-submit")
            await pilot.pause(0.2)
            assert ("submit_work", (config, "local")) in services.calls
            assert app.screen is app.screen_stack[0]

    asyncio.run(exercise())


def test_run_dialog_renders_yaml_failure_once_and_does_not_submit(tmp_path: Path) -> None:
    class InvalidYamlServices(FakeServices):
        def validate_work(self, config):
            self.calls.append(("validate_work", config))
            return {
                "valid": False,
                "errors": [
                    f"Invalid YAML in {config} at line 3, column 1:\n"
                    "     3 | º\n"
                    "Hint: This line is a standalone value."
                ],
            }

    services = InvalidYamlServices()
    config = tmp_path / "broken.yaml"
    config.write_text("name: broken\nrun: project.Training\nº\n", encoding="utf-8")

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.click("#nav-run")
            app.screen.query_one("#launch-config").value = str(config)
            await pilot.click("#launch-submit")
            await pilot.pause(0.1)

            status = str(app.screen.query_one("#launch-status").render())
            preview = str(app.screen.query_one("#launch-preview-content").render())
            assert status == "Validation failed · review the explanation below."
            assert "VALIDATION FAILED" in preview
            assert "line 3, column 1" in preview
            assert "standalone value" in preview
            assert "line 3, column 1" not in status
            assert not app.screen.query_one("#launch-submit").disabled
            assert not any(call[0] == "explain_work" for call in services.calls)
            assert not any(call[0] == "submit_work" for call in services.calls)

    asyncio.run(exercise())


def test_recent_work_store_is_atomic_bounded_and_ignores_missing_yaml(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yml"
    first.write_text("name: first\nrun: project.First\n", encoding="utf-8")
    second.write_text("name: second\nrun: project.Second\n", encoding="utf-8")
    store = RecentWorkStore(tmp_path / "state" / "recent-work.json")

    store.remember(first, name="First Work")
    store.remember(second, name="Second Work")
    store.remember(first, name="First Work")

    items = store.items()
    assert [item["name"] for item in items] == ["First Work", "Second Work"]
    assert store.path.read_text(encoding="utf-8").endswith("\n")
    first.unlink()
    assert [item["name"] for item in store.items()] == ["Second Work"]


def test_recent_work_choices_merge_existing_job_history(tmp_path: Path) -> None:
    stored = tmp_path / "stored.yaml"
    historical = tmp_path / "historical.yaml"
    stored.write_text("name: stored\nrun: project.Stored\n", encoding="utf-8")
    historical.write_text("name: historical\nrun: project.Historical\n", encoding="utf-8")
    service = ConsoleServices.__new__(ConsoleServices)
    service.recent_work = SimpleNamespace(
        items=lambda limit: (
            {
                "path": str(stored),
                "name": "stored",
                "last_used_utc": "2026-09-01T10:00:00+00:00",
            },
        )[:limit]
    )
    service.jobs = SimpleNamespace(
        list=lambda refresh: (
            SimpleNamespace(
                metadata={
                    "source_config_path": str(historical),
                    "name": "historical",
                },
                config_path="/remote/config.yaml",
                created_at_utc="2026-09-02T10:00:00+00:00",
            ),
        )
    )

    choices = service.recent_work_configs()

    assert [item["name"] for item in choices] == ["historical", "stored"]


def test_run_dialog_browses_yaml_and_remembers_valid_selection(tmp_path: Path) -> None:
    config = tmp_path / "experiments" / "train.yaml"
    config.parent.mkdir()
    config.write_text("name: training\nrun: project.Training\n", encoding="utf-8")
    services = FakeServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 38)) as pilot:
            await pilot.click("#nav-run")
            await pilot.click("#launch-browse")
            assert isinstance(app.screen, YamlFilePicker)
            location = app.screen.query_one("#yaml-picker-location")
            location.value = str(config)
            location.focus()
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert isinstance(app.screen, WorkLaunchDialog)
            assert app.screen.query_one("#launch-config").value == str(config)
            assert ("validate_work", config) not in services.calls
            assert ("explain_work", config) not in services.calls
            assert not app.screen.query_one("#launch-submit").disabled
            await pilot.click("#launch-submit")
            await pilot.pause(0.15)
            assert ("validate_work", config) in services.calls
            assert ("explain_work", config) in services.calls
            assert (
                "remember_work_config",
                (config, "training"),
            ) in services.calls
            assert ("submit_work", (config, "local")) in services.calls

    asyncio.run(exercise())


def test_recent_work_enter_selects_then_submit_validates_and_launches(
    tmp_path: Path,
) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("name: training\nrun: project.Training\n", encoding="utf-8")
    services = FakeServices(
        recent=(
            {
                "path": str(config),
                "name": "training",
                "last_used_utc": "2026-09-02T12:30:00+00:00",
            },
        )
    )

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 38)) as pilot:
            await pilot.click("#nav-run")
            dialog = app.screen
            assert isinstance(dialog, WorkLaunchDialog)
            assert dialog.query_one("#launch-recent-table").row_count == 1
            dialog.query_one("#launch-cluster").value = "gpu12"
            dialog.query_one("#launch-recent-table").focus()
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.screen is dialog
            assert dialog.query_one("#launch-config").value == str(config)
            assert not any(call[0] == "validate_work" for call in services.calls)
            assert not any(call[0] == "submit_work" for call in services.calls)
            await pilot.click("#launch-submit")
            await pilot.pause(0.2)
            assert app.screen is app.screen_stack[0]
            assert ("validate_work", config) in services.calls
            assert ("explain_work", config) in services.calls
            assert ("submit_work", (config, "gpu12")) in services.calls

    asyncio.run(exercise())


def test_enter_opens_real_work_workspace_and_escape_returns_to_table() -> None:
    snapshot = {
        "work": {
            "items": [
                {
                    "name": "training",
                    "execution_id": "execution-1",
                    "cluster": "gpu-cluster",
                    "state": "running",
                    "attempt_history": [],
                }
            ]
        },
        "clusters": [],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(80, 24)) as pilot:
            app.show_screen("work")
            await pilot.pause(0.1)
            assert app.focused is app.query_one("#work #screen-table")

            await pilot.press("enter")
            assert isinstance(app.screen, WorkWorkspace)
            assert "training" in str(app.screen.query_one(".workspace-header").render())

            await pilot.press("escape")
            assert app.screen is app.screen_stack[0]
            assert app.query_one("#work").display

    asyncio.run(exercise())


def test_background_refresh_keeps_the_previous_snapshot_visible() -> None:
    class BlockingRefreshServices(FakeServices):
        def __init__(self):
            super().__init__({"work": {"items": []}, "clusters": []})
            self.started = threading.Event()
            self.release = threading.Event()
            self.loads = 0

        def overview_snapshot(self):
            self.loads += 1
            if self.loads > 1:
                self.started.set()
                self.release.wait(timeout=2)
            return self.snapshot

    services = BlockingRefreshServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            overview = app.query_one("#overview")
            assert overview.last_success == services.snapshot
            overview.reload()
            await pilot.pause(0.05)
            assert services.started.is_set()
            assert not overview.query_one("#screen-loading").display
            assert "refreshing in the background" in str(
                overview.query_one("#screen-freshness").render()
            )
            assert overview.last_success == services.snapshot
            services.release.set()
            await pilot.pause(0.1)
            assert "Last successful update" in str(overview.query_one("#screen-freshness").render())

    asyncio.run(exercise())


def test_work_table_and_logs_accept_live_snapshots() -> None:
    preparing = {
        "work": {
            "items": [
                {
                    "work_id": "work-live",
                    "name": "preprocess",
                    "cluster": "local",
                    "state": "preparing",
                    "primary_job_id": "job-live",
                    "attempt_history": [],
                }
            ]
        },
        "clusters": [],
    }

    class LiveServices(FakeServices):
        def __init__(self):
            super().__init__(preparing)
            self.log_reads = 0

        def work_logs(self, job_id):
            self.log_reads += 1
            suffix = "\nsecond line" if self.log_reads > 1 else ""
            return {"text": f"first line{suffix}"}

    services = LiveServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            app.show_screen("work")
            await pilot.pause(0.1)
            table = app.query_one("#work #screen-table")
            assert str(table.get_row_at(0)[2]) == "preparing"
            services.snapshot = {
                **preparing,
                "work": {"items": [{**preparing["work"]["items"][0], "state": "running"}]},
            }
            app.query_one("#work")._refresh_visible()
            await pilot.pause(0.1)
            assert str(table.get_row_at(0)[2]) == "running"

            table.move_cursor(row=0)
            await pilot.press("enter")
            workspace = app.screen
            assert isinstance(workspace, WorkWorkspace)
            await pilot.pause(0.1)
            assert workspace._log_text == "first line"
            workspace._refresh_logs()
            await pilot.pause(0.1)
            assert workspace._log_text == "first line\nsecond line"
            assert "updated just now" in str(workspace.query_one("#work-log-status").render())

    asyncio.run(exercise())


def test_same_named_work_on_two_clusters_has_distinct_rows_and_navigation() -> None:
    snapshot = {
        "work": {
            "items": [
                {
                    "work_id": "work-local",
                    "name": "wisdom-dna-preprocess",
                    "cluster": "local",
                    "state": "cancelled",
                    "attempt_history": [],
                },
                {
                    "work_id": "work-remote",
                    "name": "wisdom-dna-preprocess",
                    "cluster": "citius-ctgpgpu12",
                    "state": "preparing",
                    "attempt_history": [],
                },
            ]
        },
        "clusters": [],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(100, 30)) as pilot:
            app.show_screen("work")
            await pilot.pause(0.1)
            table = app.query_one("#work #screen-table")
            assert table.row_count == 2
            assert str(table.get_row_at(0)[1]) == "local"
            assert str(table.get_row_at(1)[1]) == "citius-ctgpgpu12"

            table.move_cursor(row=1)
            await pilot.press("enter")
            assert isinstance(app.screen, WorkWorkspace)
            assert "citius-ctgpgpu12" in str(app.screen.query_one(".workspace-header").render())
            assert app.screen._selector == "work-remote"

    asyncio.run(exercise())


def test_cli_parity_inventory_covers_every_public_family() -> None:
    assert set(CLI_PARITY) == {
        "project/work",
        "system",
        "cluster",
        "job",
        "dataset",
        "result",
    }
    result_operations = {value["operation"] for value in CLI_PARITY["result"]}
    cluster_operations = {value["operation"] for value in CLI_PARITY["cluster"]}
    assert {"analyze", "report"} <= result_operations
    assert {"add", "bootstrap", "credentials set"} <= cluster_operations
    assert all(action.handler is not None for action in PALETTE_ACTIONS)
    assert all(action.status == "IMPLEMENTED" for action in PALETTE_ACTIONS)
    assert all(
        action.service_method is None
        or callable(getattr(ConsoleServices, action.service_method, None))
        for action in PALETTE_ACTIONS
    )


def test_pruned_trial_detail_marks_partial_evidence_as_censored() -> None:
    snapshot = {
        "work": {
            "items": [
                {
                    "name": "study",
                    "state": "running",
                    "study": {
                        "objective": {"metric": "score"},
                        "counts": {"candidates": 1, "pruned_runs": 1},
                        "candidates": [
                            {
                                "trial": 4,
                                "state": "pruned",
                                "best_objective": 0.42,
                                "partially_censored": True,
                            }
                        ],
                    },
                }
            ]
        },
        "clusters": [],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(100, 32)) as pilot:
            app.show_screen("studies")
            await pilot.pause(0.1)
            screen = app.query_one("#studies")
            assert "partial/censored" in screen.detail(0)
            assert "0.42†" in screen.detail(0)

    asyncio.run(exercise())


def test_pruned_trial_table_and_chart_show_partial_values_without_finalizing_them() -> None:
    work = {
        "work_id": "work-pruned",
        "name": "study",
        "state": "running",
        "study": {
            "objective": {"metric": "score", "mode": "max"},
            "counts": {"candidates": 1, "pruned_runs": 1},
            "candidates": [
                {
                    "trial": 4,
                    "state": "pruned",
                    "current_objective": 0.39,
                    "best_objective": 0.42,
                    "partially_censored": True,
                    "parameters": {"width": 32},
                    "runs": [
                        {
                            "key": "trial-00004-seed-7",
                            "seed": 7,
                            "state": "pruned",
                            "latest_step": 8,
                            "best_step": 6,
                            "current_observed_objective": 0.39,
                            "best_observed_objective": 0.42,
                        }
                    ],
                }
            ],
        },
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(StudyWorkspace(work, app.services))
            await pilot.pause(0.1)
            table = app.screen.query_one("#trial-table")
            row = [str(value) for value in table.get_row_at(0)]
            assert len(table.columns) == 9  # No meaningless all-empty final-SE column.
            assert row[0] == "†"
            assert row[2] == "censored †"
            assert row[3:5] == ["0.39 †", "0.42 †"]
            plot = app.screen.query_one(".study-objective-plot")
            assert any(point.series == "pruned candidate" for point in plot._inspection_points)

    asyncio.run(exercise())


def test_studies_refresh_live_without_expanding_admission_json() -> None:
    def snapshot(state: str) -> dict[str, object]:
        return {
            "work": {
                "items": [
                    {
                        "work_id": "work-study",
                        "name": "study",
                        "cluster": "gpu12",
                        "state": state,
                        "study_expected": True,
                        "study": {
                            "objective": {"metric": "score", "mode": "max"},
                            "counts": {"candidates": 10, "active_runs": 6},
                            "candidates": [],
                            "admission": {
                                "current": {
                                    "status": "limited",
                                    "reason": "launch stagger or lease",
                                    "devices": [{"gpu": 0, "available": 123}],
                                }
                            },
                        },
                    }
                ]
            },
            "clusters": [],
        }

    services = FakeServices(snapshot("staging"))

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(110, 34)) as pilot:
            app.show_screen("studies")
            await pilot.pause(0.1)
            screen = app.query_one("#studies")
            table = screen.query_one("#screen-table")
            assert str(table.get_row_at(0)[1]) == "staging"

            detail = screen.detail(0)
            assert "Admission   limited · launch stagger or lease" in detail
            assert '"devices"' not in detail
            assert len(detail.splitlines()) <= 8

            services.snapshot = snapshot("running")
            screen._refresh_visible()
            await pilot.pause(0.1)
            assert str(table.get_row_at(0)[1]) == "running"

    asyncio.run(exercise())


def test_study_trial_seed_epoch_navigation_uses_persisted_telemetry() -> None:
    run = {
        "key": "trial-00017-seed-54",
        "trial": 17,
        "seed": 54,
        "state": "running",
        "latest_step": 100,
        "best_step": 63,
        "current_observed_objective": 0.6,
        "best_observed_objective": 0.663,
        "final_objective": None,
        "gpu_index": 1,
    }
    work = {
        "name": "WISDOM",
        "execution_id": "execution-1",
        "primary_job_id": "job-1",
        "cluster": "gpu12",
        "state": "running",
        "study": {
            "objective": {"metric": "val_auprc", "mode": "max"},
            "counts": {"candidates": 1, "active_runs": 1},
            "candidates": [
                {
                    "trial": 17,
                    "state": "running",
                    "parameters": {"hidden_dim": 128},
                    "selection_objective": 0.61,
                    "current_objective": 0.6,
                    "best_objective": 0.663,
                    "runs": [run],
                }
            ],
        },
    }
    services = FakeServices({"work": {"items": [work]}, "clusters": []})

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 40)) as pilot:
            app.show_screen("studies")
            await pilot.pause(0.1)
            await pilot.press("enter")
            assert isinstance(app.screen, StudyWorkspace)
            assert "Running 1" in str(app.screen.query_one(".study-state-counts").render())

            app.screen.query_one("#trial-table").focus()
            await pilot.press("enter")
            assert isinstance(app.screen, TrialWorkspace)

            app.screen.query_one("#seed-table").focus()
            await pilot.press("enter")
            await pilot.pause(0.15)
            assert isinstance(app.screen, SeedWorkspace)
            assert "AUPRC" in str(app.screen.query_one("#curve-content").render())
            assert "best epoch 63" in str(app.screen.query_one("#seed-header").render())
            await pilot.click("#curve-next")
            assert "CURVES 2/2" in str(app.screen.query_one("#curve-content").render())
            assert "Train loss" in str(app.screen.query_one("#curve-content").render())

            artifacts = app.screen.query_one("#seed-artifact-table")
            assert artifacts.row_count == 2
            assert "1abc.html" in str(app.screen.query_one("#seed-artifact-detail").render())

            epochs = app.screen.query_one("#epoch-table")
            assert epochs.row_count == 100
            epochs.move_cursor(row=62)
            await pilot.pause(0.05)
            selected_row = list(epochs.get_row_at(62))
            assert selected_row[0] == "63"
            assert selected_row[1] == "0.563"
            assert len(selected_row) == 7  # epoch, five scalar metrics and marker
            assert "Back to Trial 17" in str(app.screen.query_one("#workspace-back").label)
            await pilot.click("#workspace-back")
            assert isinstance(app.screen, TrialWorkspace)

    asyncio.run(exercise())


def test_study_resources_show_exact_persisted_admission_reason() -> None:
    work = {
        "name": "capacity-study",
        "primary_job_id": "job-capacity",
        "state": "running",
        "study": {
            "objective": {"metric": "loss", "mode": "min"},
            "counts": {"candidates": 0},
            "candidates": [],
            "admission": {
                "current": {
                    "requested": {"gpus": 1, "runs_per_gpu": 5, "max_parallel": 5},
                    "potential_slots": 5,
                    "status": "limited",
                    "limiting_resource": "gpu_memory",
                    "reason": "only 3 Runs fit the current free VRAM",
                }
            },
        },
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test() as pilot:
            app.push_screen(StudyWorkspace(work, app.services))
            await pilot.pause(0.1)
            text = str(app.screen.query_one("#study-resource-content").render())
            assert "runs_per_gpu          5" in text
            assert "only 3 Runs fit the current free VRAM" in text

    asyncio.run(exercise())


def test_study_resources_render_v31_wait_explanation_without_raw_json() -> None:
    work = {
        "name": "capacity-study",
        "primary_job_id": "job-capacity",
        "state": "running",
        "study": {
            "objective": {"metric": "loss", "mode": "min"},
            "counts": {"candidates": 1},
            "candidates": [],
            "admission": {
                "current": {
                    "admission_version": 3,
                    "summary": "waiting_for_resources",
                    "pending_runs": 1,
                    "devices": [],
                    "admitted": [],
                    "resource_blocked": [],
                    "exploration_evaluations": [
                        {
                            "candidate": "candidate-1",
                            "gpu": 0,
                            "plan": "WAIT",
                            "rejection_reason": "ROLLBACK_DOMINATES",
                            "fit_probability": 0.84,
                            "tail_probability": 0.005,
                            "peak_hazard": 0.1,
                            "evidence_cycles": 12,
                            "phase_hazards": {"validation": 0.2},
                            "fit_uncertainty_sources": ["resident:a:phase:validation"],
                            "rollback_seconds": 300.0,
                            "resource_information_value": 0.2,
                            "wait_regret": 0.4,
                            "wait_regret_details": {
                                "current_idle_usable_fraction": 0.5,
                                "scientific_opportunity_rate": 0.01,
                            },
                            "final_delta_value": -0.1,
                        }
                    ],
                }
            },
        },
    }

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices())
        async with app.run_test() as pilot:
            app.push_screen(StudyWorkspace(work, app.services))
            await pilot.pause(0.1)
            text = str(app.screen.query_one("#study-resource-content").render())
            assert "WHY WAIT / WHY EXPLORE" in text
            assert "ROLLBACK_DOMINATES" in text
            assert "P(fit) / tail" in text
            assert "validation=0.2" in text
            assert "wait regret" in text

    asyncio.run(exercise())


def test_metric_labels_hide_internal_objective_names() -> None:
    assert metric_display_name("val_balanced_accuracy") == "Balanced accuracy"
    assert metric_display_name("train_loss") == "Train loss"
    assert metric_display_name("max_gpu_reserved_memory") == "Peak reserved GPU memory"
    assert objective_display_name({"metric": "__lambdaforge_utility__", "mode": "max"}) == (
        "Composite selection score"
    )


def test_seed_workspace_distinguishes_remote_loading_from_empty_telemetry() -> None:
    ready = threading.Event()

    class SlowServices(FakeServices):
        def study_run(self, job_id, run_key, *, tail=2_000, curve_points=200):
            ready.wait(timeout=2)
            return super().study_run(job_id, run_key, tail=tail, curve_points=curve_points)

    async def exercise() -> None:
        app = LambdaForgeApp(SlowServices())
        async with app.run_test(size=(100, 30)) as pilot:
            app.push_screen(
                SeedWorkspace(
                    {"name": "study", "primary_job_id": "job-1"},
                    {"trial": 1},
                    {"key": "run-1", "seed": 4, "state": "running"},
                    app.services,
                    objective={"metric": "score", "mode": "max"},
                )
            )
            await pilot.pause(0.05)
            assert app.screen.query_one("#seed-loading").display
            assert not app.screen.query_one("#seed-tabs").display
            assert "Loading persisted Run telemetry" in str(
                app.screen.query_one("#seed-loading-message").render()
            )
            ready.set()
            await pilot.pause(0.15)
            assert not app.screen.query_one("#seed-loading").display
            assert app.screen.query_one("#seed-tabs").display

    asyncio.run(exercise())


def test_hpo_workspace_exposes_parameter_evidence_actions_and_clickable_breadcrumbs() -> None:
    class HpoServices(FakeServices):
        def result_analysis(self, selector):
            return {
                "winner": {},
                "seed_analysis": {"status": "available"},
                "surrogate": {"quality": "medium", "observations": 3},
                "search_space": {
                    "hidden_dim": {
                        "kind": "numeric",
                        "low": 64.0,
                        "high": 256.0,
                        "values": [64, 128, 256],
                    }
                },
                "coverage": {
                    "marginal": {
                        "hidden_dim": {
                            "kind": "numeric",
                            "observed_range": [64.0, 128.0],
                        }
                    }
                },
                "parameter_importance": {
                    "hidden_dim": {"importance": 0.41, "reliability": "medium"}
                },
                "top_region_importance": {},
                "response_curves": {
                    "hidden_dim": {
                        "points": [
                            {
                                "x": 64.0,
                                "predicted_objective": 0.5,
                                "uncertainty": 0.1,
                                "support_count": 1,
                            },
                            {
                                "x": 128.0,
                                "predicted_objective": 0.7,
                                "uncertainty": 0.05,
                                "support_count": 2,
                            },
                        ],
                        "best_supported_point": {"x": 128.0},
                    }
                },
                "interactions": {"pairs": []},
                "candidates": [
                    {"parameters": {"hidden_dim": 64}, "mean": 0.5},
                    {"parameters": {"hidden_dim": 128}, "mean": 0.68},
                    {"parameters": {"hidden_dim": 128}, "mean": 0.72},
                ],
                "findings": [],
            }

    work = {
        "name": "visual-hpo",
        "execution_id": "execution-hpo",
        "primary_job_id": "job-hpo",
        "state": "running",
        "study": {
            "strategy": "adaptive",
            "objective": {"metric": "__lambdaforge_utility__", "mode": "max"},
            "counts": {"candidates": 1, "completed_runs": 1},
            "hpo_analysis": {
                "parameters": [
                    {
                        "parameter": "hidden_dim",
                        "pruning_signal": {
                            "groups": [
                                {
                                    "value": 64.0,
                                    "observations": 2,
                                    "pruned": 1,
                                    "pruned_rate": 0.5,
                                },
                                {
                                    "value": 128.0,
                                    "observations": 2,
                                    "pruned": 0,
                                    "pruned_rate": 0.0,
                                },
                            ]
                        },
                    }
                ]
            },
            "candidates": [
                {
                    "trial": 7,
                    "state": "succeeded",
                    "selection_objective": 0.7,
                    "pareto_optimal": True,
                    "parameters": {"hidden_dim": 128},
                    "runs": [],
                }
            ],
            "controller": {
                "recent": [
                    {
                        "action": "PROMOTE",
                        "trial": 7,
                        "reason": "credible improvement",
                        "controller_value": 0.8,
                    }
                ]
            },
        },
    }

    async def exercise() -> None:
        app = LambdaForgeApp(HpoServices())
        async with app.run_test(size=(140, 44)) as pilot:
            app.push_screen(StudyWorkspace(work, app.services))
            await pilot.pause(0.15)
            screen = app.screen
            assert isinstance(screen, StudyWorkspace)
            trials = screen.query_one("#trial-table")
            assert str(trials.get_row_at(0)[0]) == "★ ◆"
            assert str(trials.get_row_at(0)[1]) == "7"
            assert "Composite selection score" in str(
                screen.query_one("#hpo-objective-card").render()
            )
            screen.query_one("#study-tabs", TabbedContent).active = "study-hpo"
            await pilot.pause(0.15)
            parameters = screen.query_one("#hpo-parameter-table")
            assert parameters.row_count == 1
            assert "64, 128, 256" in str(parameters.get_row_at(0)[1])
            actions = screen.query_one("#hpo-action-table")
            assert actions.row_count == 1
            assert len(actions.columns) == 3

            screen._open_hpo_parameter(0)
            await pilot.pause(0.05)
            assert isinstance(app.screen, HpoParameterWorkspace)
            assert app.screen.query_one("#hpo-dispersion-table").row_count == 2
            coverage_plot = app.screen.query_one(".hpo-coverage-plot")
            assert any(point.series == "pruned" for point in coverage_plot._inspection_points)
            await pilot.click("#breadcrumb-1")
            assert isinstance(app.screen, StudyWorkspace)

            app.screen._open_hpo_action(0)
            await pilot.pause(0.05)
            assert isinstance(app.screen, HpoActionWorkspace)
            assert "credible improvement" in str(app.screen.query_one(".workspace-panel").render())

    asyncio.run(exercise())


def test_study_heavy_analysis_and_logs_are_loaded_only_for_their_tabs() -> None:
    class LazyServices(FakeServices):
        def result_analysis(self, selector):
            self.calls.append(("result_analysis", selector))
            return {
                "winner": {},
                "seed_analysis": {},
                "surrogate": {},
                "parameter_importance": {},
                "findings": [],
            }

        def work_logs(self, job_id, *, tail=2_000):
            self.calls.append(("work_logs", job_id))
            return {"text": "study output"}

        def study_actions(self, job_id):
            self.calls.append(("study_actions", job_id))
            return ({"action": "PROPOSE", "trial": 1, "reason": "coverage"},)

    work = {
        "work_id": "work-lazy",
        "name": "lazy-study",
        "execution_id": "execution-lazy",
        "primary_job_id": "job-lazy",
        "state": "succeeded",
        "study_expected": True,
        "study": {
            "study_telemetry_version": 1,
            "strategy": "adaptive",
            "objective": {"metric": "score", "mode": "max"},
            "counts": {"candidates": 0, "completed_runs": 0},
            "candidates": [],
        },
    }
    services = LazyServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 38)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause(0.1)
            assert not any(call[0] == "result_analysis" for call in services.calls)
            assert not any(call[0] == "work_logs" for call in services.calls)
            assert not any(call[0] == "study_actions" for call in services.calls)

            app.screen.query_one("#study-tabs", TabbedContent).active = "study-hpo"
            await pilot.pause(0.1)
            assert ("result_analysis", "execution-lazy") in services.calls
            assert not any(call[0] == "work_logs" for call in services.calls)
            assert not any(call[0] == "study_actions" for call in services.calls)

            app.screen.query_one("#hpo-tabs", TabbedContent).active = "hpo-actions-pane"
            await pilot.pause(0.1)
            assert ("study_actions", "job-lazy") in services.calls

            app.screen.query_one("#study-tabs", TabbedContent).active = "study-logs"
            await pilot.pause(0.1)
            assert ("work_logs", "job-lazy") in services.calls

    asyncio.run(exercise())


def test_pruned_seed_keeps_best_partial_evidence_without_final_score() -> None:
    class PrunedServices(FakeServices):
        def study_run(self, job_id, run_key, *, tail=2_000, curve_points=200):
            value = super().study_run(job_id, run_key, tail=tail, curve_points=curve_points)
            value.update(
                {
                    "state": "pruned",
                    "best_observed_objective": 0.642,
                    "best_step": 53,
                    "final_objective": None,
                }
            )
            return value

    run = {
        "key": "trial-00017-seed-54",
        "seed": 54,
        "state": "pruned",
        "latest_step": 60,
        "best_step": 53,
    }
    work = {"name": "study", "primary_job_id": "job-1"}
    candidate = {"trial": 17, "runs": [run]}

    async def exercise() -> None:
        services = PrunedServices()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            app.push_screen(
                SeedWorkspace(
                    work,
                    candidate,
                    run,
                    services,
                    objective={"metric": "val_auprc", "mode": "max"},
                )
            )
            await pilot.pause(0.15)
            header = str(app.screen.query_one("#seed-header").render())
            assert "PRUNED" in header and "CENSORED" in header
            assert "best 0.642" in header
            assert "final not final · pruned" in header
            assert "best epoch 53" in header

    asyncio.run(exercise())


def test_cluster_workspace_buttons_call_real_services() -> None:
    services = FakeServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(100, 32)) as pilot:
            app.push_screen(ClusterWorkspace("gpu12", services))
            await pilot.pause(0.1)
            assert ("cluster_detail", "gpu12") in services.calls
            workspace = app.screen
            workspace._set_operation_height(100)
            await pilot.pause(0.05)
            main = workspace.query_one("#cluster-workspace-main")
            log = workspace.query_one("#cluster-operation-log")
            assert main.size.height >= 8
            assert log.size.height <= max(6, workspace.size.height - 22)
            assert main.max_scroll_y > 0
            await pilot.click("#cluster-doctor")
            await pilot.click("#cluster-bootstrap-plan")
            await pilot.pause(0.15)
            assert ("doctor", "gpu12") in services.calls
            assert ("bootstrap", ("gpu12", True)) in services.calls
            assert "Resolving the managed runtime" in "\n".join(
                str(line) for line in app.screen.query_one("#cluster-operation-log").lines
            )
            await pilot.click("#cluster-bootstrap-apply")
            assert isinstance(app.screen, ExactConfirmation)
            confirmation = str(app.screen.query_one("#exact-preview").render())
            assert "This action changes persisted state" in confirmation
            assert "WILL CHANGE" in confirmation
            assert "system Python" in confirmation
            assert "{" not in confirmation
            await pilot.click("#exact-cancel")
            assert ("bootstrap", ("gpu12", False)) not in services.calls

    asyncio.run(exercise())


def test_dataset_workspace_uses_cards_and_bounded_member_table() -> None:
    services = FakeServices()
    dataset = {
        "name": "proteins",
        "version": "4",
        "dataset_id": "sha256:dataset",
        "sample_count": 2,
        "partitions": {"split": {"train": 1, "validation": 1}},
        "lineage": ["proteins@3"],
        "placements": [
            {
                "cluster": "gpu12",
                "root": "/datasets/proteins/4",
                "size_bytes": 2 * 1024**3,
            }
        ],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 38)) as pilot:
            app.push_screen(DatasetWorkspace(dataset, services))
            await pilot.pause(0.1)
            cards = app.screen.query(".metric-card")
            assert len(cards) == 4
            assert "2.0 GiB" in " ".join(str(card.render()) for card in cards)
            app.screen.query_one("#dataset-tabs").active = "dataset-members"
            await pilot.pause(0.2)
            table = app.screen.query_one("#dataset-member-table")
            assert table.row_count == 2
            assert "protein-1" in str(app.screen.query_one("#dataset-member-content").render())
            assert "Exact logical-index counts" in str(
                app.screen.query_one("#dataset-split-note").render()
            )

    asyncio.run(exercise())


def test_dataset_delete_requires_confirmation_and_handles_stale_placement() -> None:
    services = FakeServices()
    dataset = {
        "name": "wisdom",
        "version": "4",
        "dataset_id": "sha256:dataset",
        "sample_count": 10,
        "placements": [{"cluster": "gpu12", "root": "/missing/wisdom/4"}],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            app.push_screen(DatasetWorkspace(dataset, services))
            await pilot.pause(0.1)
            assert app.screen.query_one("#dataset-delete").display
            app.screen._preview_delete()
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExactConfirmation)
            assert "registered_but_missing" in str(app.screen.query_one("#exact-preview").render())
            await pilot.click("#exact-cancel")
            assert ("delete_dataset", ("wisdom@4", True)) not in services.calls

            app.screen._preview_delete()
            await pilot.pause(0.1)
            await pilot.click("#exact-apply")
            await pilot.pause(0.15)
            assert ("delete_dataset", ("wisdom@4", True)) in services.calls

    asyncio.run(exercise())


def test_dataset_delete_shows_immediate_progress_and_blocks_repeated_clicks() -> None:
    release = threading.Event()

    class SlowDeleteServices(FakeServices):
        def delete_dataset(self, selector, *, apply=False):
            self.calls.append(("delete_dataset", (selector, apply)))
            if not apply:
                release.wait(timeout=2)
            return {"dataset": selector, "safe": True, "applied": apply, "placements": []}

    services = SlowDeleteServices()
    dataset = {"name": "wisdom", "version": "4", "placements": []}

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            app.push_screen(DatasetWorkspace(dataset, services))
            await pilot.pause(0.05)
            workspace = app.screen
            assert isinstance(workspace, DatasetWorkspace)
            workspace._preview_delete()
            assert workspace.query_one("#dataset-delete").disabled
            assert "Preparing exact deletion preview" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            await pilot.pause(0.05)
            workspace._preview_delete()
            assert services.calls.count(("delete_dataset", ("wisdom@4", False))) == 1
            release.set()
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExactConfirmation)
            await pilot.click("#exact-cancel")
            await pilot.pause()
            assert "Deletion cancelled" in str(
                workspace.query_one("#dataset-operation-status").render()
            )

    asyncio.run(exercise())


def test_resource_dashboard_compares_like_for_like_percentages() -> None:
    values = ResourceDashboard._metric_values(
        {
            "cpu_total": 8,
            "cpu_load": 50.0,
            "ram_total_bytes": 1000,
            "ram_available_bytes": 600,
            "gpus": [
                {"memory_used_bytes": 25, "memory_total_bytes": 100},
                {"memory_used_bytes": 50, "memory_total_bytes": 100},
            ],
        },
        {"job_count": 1, "cpu_percent": 200, "rss_bytes": 100, "gpu_memory_bytes": 50},
    )
    assert values == {
        "cpu": (50.0, 25.0),
        "ram": (40.0, 10.0),
        "gpu": (37.5, 25.0),
    }
    assert ResourceDashboard._format_relative_time(-300.0) == "5m"
    assert ResourceDashboard._format_relative_time(-75.0) == "1m15s"
    assert ResourceDashboard._format_relative_time(0.0) == "now"
    assert ResourceDashboard._series(((0.0, 10.0), (100.0, 20.0), (400.0, 30.0)), 400.0) == (
        [-300.0, 0.0],
        [20.0, 30.0],
    )


def test_resource_chart_export_bubbles_to_the_app_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        "clusters": [
            {
                "cluster": "gpu12",
                "online": True,
                "scheduler": "local",
                "observed": {
                    "cpu_total": 8,
                    "cpu_load": 30.0,
                    "ram_total_bytes": 1000,
                    "ram_available_bytes": 600,
                    "gpus": [{"memory_total_bytes": 1000, "memory_used_bytes": 500}],
                },
                "personal": {"active_jobs": 0, "requested": {}, "observed": {}},
            }
        ],
        "work": {"items": []},
    }
    captured: dict[str, object] = {}

    def write(cluster: str, series: object, _destination: Path) -> Path:
        captured.update(cluster=cluster, series=series)
        destination = tmp_path / "resources.html"
        destination.write_text("report", encoding="utf-8")
        return destination

    monkeypatch.setattr("lambdaforge.tui.App.write_resource_html", write)
    monkeypatch.setattr("lambdaforge.tui.App.webbrowser.open", lambda _uri: False)

    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(snapshot))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.15)
            await pilot.click("#overview-resource-dashboard .resource-export")
            await pilot.pause(0.1)
            assert captured["cluster"] == "gpu12"
            assert "total_cpu" in captured["series"]  # type: ignore[operator]

    asyncio.run(exercise())


def test_metric_page_change_restores_data_driven_plot_limits() -> None:
    class Plot:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def set_xlimits(self, lower=None, upper=None) -> None:
            self.calls.append((lower, upper))

        def set_ylimits(self, lower=None, upper=None) -> None:
            self.calls.append((lower, upper))

    plot = Plot()
    MetricDashboard._reset_viewport(plot)
    assert plot.calls == [(None, None), (None, None)]


def test_hpo_summary_exposes_terminal_controller_reason() -> None:
    assert (
        StudyWorkspace._controller_stop_summary(
            {
                "last": {"action": "FINISH", "reason": "candidate-budget-reached"},
                "recent": [],
            }
        )
        == "Stop: Candidate budget reached"
    )
    assert (
        StudyWorkspace._controller_stop_summary(
            {
                "last": {"action": "CONFIRM"},
                "recent": [
                    {"action": "STOP_PROPOSING", "reason": "full-fidelity-evidence-convergence"},
                    {"action": "CONFIRM"},
                ],
            }
        )
        == "Stop: Full fidelity evidence convergence"
    )


def test_live_study_hpo_parameters_do_not_require_a_final_execution() -> None:
    live_parameter = {
        "parameter": "learning_rate",
        "confidence": 0.64,
        "confidence_label": "medium",
        "best_observed_value": 0.001,
        "response": {
            "kind": "numeric-binned",
            "points": [
                {"parameter": 0.0001, "objective": 0.51, "samples": 2},
                {"parameter": 0.001, "objective": 0.63, "samples": 3},
            ],
        },
        "joint_relationships": [
            {
                "parameter": "dropout",
                "gain": 0.12,
                "confidence": 0.55,
                "confidence_label": "low",
                "observations": 5,
            }
        ],
    }
    study = {
        "objective": {"metric": "val_auprc", "mode": "max"},
        "counts": {"candidates": 5, "completed_runs": 3},
        "candidates": [],
        "hpo_analysis": {"parameters": [live_parameter]},
    }
    analysis = {
        "search_space": {"learning_rate": {"kind": "numeric", "low": 0.00001, "high": 0.003}},
        "coverage": {"marginal": {"learning_rate": {"observed_range": [0.0001, 0.001]}}},
        "parameter_importance": {},
        "response_curves": {},
        "interactions": {"pairs": []},
        "candidates": [],
        "live_hpo": {"parameters": [live_parameter]},
        "winner": {},
        "seed_analysis": {},
        "surrogate": {},
        "findings": [],
    }

    class LiveAnalysisServices(FakeServices):
        def result_analysis(self, selector):
            raise AssertionError("A live Study without an Execution must not request final results")

        def live_study_analysis(self, value, *, job_id=None):
            self.calls.append(("live_study_analysis", job_id))
            assert value == study
            return analysis

    work = {
        "name": "live-study",
        "primary_job_id": "job-live",
        "state": "running",
        "study": study,
    }
    services = LiveAnalysisServices()

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 42)) as pilot:
            app.push_screen(StudyWorkspace(work, services))
            await pilot.pause(0.15)
            app.screen.query_one("#study-tabs", TabbedContent).active = "study-hpo"
            await pilot.pause(0.15)
            table = app.screen.query_one("#hpo-parameter-table")
            assert table.row_count == 1
            row = [str(value) for value in table.get_row_at(0)]
            assert "Learning Rate" in row
            assert "Near 0.001 (observed)" in row
            assert "64% · Medium" in row

            table.focus()
            await pilot.press("enter")
            assert isinstance(app.screen, HpoParameterWorkspace)
            reliability = str(app.screen.query_one("#hpo-detail-reliability").render())
            assert "64%" in reliability
            assert app.screen.query_one("#hpo-dispersion-table").row_count == 2
            assert app.screen.query_one("#hpo-relation-table").row_count == 1

    asyncio.run(exercise())


def test_console_service_builds_provisional_analysis_from_live_snapshot() -> None:
    study = {
        "execution_id": "execution-live",
        "objective": {"metric": "score", "mode": "max"},
        "candidates": [
            {"trial": 1, "parameters": {"width": 64}, "runs": []},
            {"trial": 2, "parameters": {"width": 128}, "runs": []},
        ],
        "hpo_analysis": {
            "parameters": [{"parameter": "width", "confidence": 0.25, "confidence_label": "low"}]
        },
    }

    result = ConsoleServices.live_study_analysis(object.__new__(ConsoleServices), study)

    assert result["source"]["status"] == "provisional"
    assert result["search_space"]["width"]["low"] == 64.0
    assert result["search_space"]["width"]["high"] == 128.0
    assert result["search_space_source"] == "observed candidates"
    assert result["coverage"]["marginal"]["width"]["observed_range"] == [64.0, 128.0]
    assert result["live_hpo"] == study["hpo_analysis"]


def test_result_delete_requires_exact_preview_and_cancel_does_not_apply() -> None:
    services = FakeServices()
    result = {"name": "study", "execution_id": "execution-7", "status": "succeeded"}

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            app.push_screen(ResultWorkspace(result, services))
            await pilot.pause(0.05)
            await pilot.click("#result-delete")
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExactConfirmation)
            assert "result envelope" in str(app.screen.query_one("#exact-preview").render())
            await pilot.click("#exact-cancel")
            assert ("delete_result", ("execution-7", True)) not in services.calls

            await pilot.click("#result-delete")
            await pilot.pause(0.1)
            await pilot.click("#exact-apply")
            await pilot.pause(0.1)
            assert ("delete_result", ("execution-7", True)) in services.calls

    asyncio.run(exercise())


def test_narrow_layout_and_worker_failure_keep_console_alive() -> None:
    async def exercise() -> None:
        app = LambdaForgeApp(FakeServices(fail=True))
        async with app.run_test(size=(70, 24)) as pilot:
            await pilot.pause(0.1)
            assert app.screen.has_class("narrow")
            assert "Temporarily unavailable" in str(
                app.query_one("#overview #screen-status").render()
            )
            app.show_screen("clusters")
            await pilot.pause(0.05)
            assert app.query_one("#clusters").display

    asyncio.run(exercise())


def test_palette_never_advertises_an_action_without_a_real_handler() -> None:
    unsupported_destructive = next(
        value
        for value in CONSOLE_ACTIONS
        if value.family == "dataset" and value.operation == "delete"
    )
    assert unsupported_destructive.destructive is True
    assert unsupported_destructive.status == "CLI-ONLY BY DESIGN"
    assert unsupported_destructive not in PALETTE_ACTIONS


def test_no_command_dispatch_uses_console_only_for_a_tty(monkeypatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    stdout = SimpleNamespace(isatty=lambda: True, write=lambda value: None)
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("lambdaforge.tui.App.run_console", lambda: calls.append("console") or 0)
    assert CommandLineInterface._dispatch((), json_output=False) == 0
    assert calls == ["console"]

    monkeypatch.undo()
    assert CommandLineInterface._dispatch((), json_output=False) == 0
    assert "usage: lf" in capsys.readouterr().out
