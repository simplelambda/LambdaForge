"""Cross-project operations share hardware, never another project's scientific history."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.jobs import JobRecord, JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.tui.RecentWorkStore import RecentWorkStore
from lambdaforge.work.ResultStore import ResultStore


def project(root: Path, identifier: str | None = None) -> ProjectContext:
    root.mkdir(parents=True)
    content = '[project]\nname = "same-package"\nversion = "1.0"\n'
    if identifier is not None:
        content += f'[tool.lambdaforge]\nproject_id = "{identifier}"\n'
    (root / "pyproject.toml").write_text(content)
    (root / "experiments").mkdir()
    (root / "experiments" / "work.yaml").write_text("name: same-work\n")
    return ProjectContext.discover(root)


def record(identifier: str, context: ProjectContext) -> JobRecord:
    source = str(context.root / "experiments" / "work.yaml")
    return JobRecord(
        identifier,
        "gpu",
        "local",
        identifier,
        JobState.SUCCEEDED,
        ("python", "-m", "lambdaforge"),
        f"/remote/.lambdaforge/jobs/{identifier}/work",
        {},
        "2026-09-05T10:00:00+00:00",
        "2026-09-05T11:00:00+00:00",
        config_path=source,
        metadata={"source_config_path": source, "name": "same-work"},
    )


def test_distinct_checkouts_same_package_and_subdirectory_identity(tmp_path: Path) -> None:
    first = project(tmp_path / "a" / "same")
    second = project(tmp_path / "b" / "same")
    assert first.project_id != second.project_id
    assert ProjectContext.discover(first.root / "experiments") == first
    assert not first.owns_source(str(second.root / "experiments" / "work.yaml"))
    nested = project(first.root / "nested")
    assert not first.owns_source(str(nested.root / "experiments" / "work.yaml"))


def test_explicit_project_id_survives_checkout_relocation_and_rejects_paths(tmp_path: Path) -> None:
    first = project(tmp_path / "old", "protein-study")
    second = project(tmp_path / "new", "protein-study")
    assert first.project_id == second.project_id
    with pytest.raises(ValueError, match="project_id"):
        project(tmp_path / "unsafe", "../other")


def test_scaffold_persists_a_stable_project_identity(tmp_path: Path) -> None:
    from lambdaforge.cli.scaffold import initialize

    original = tmp_path / "new-study"
    assert initialize(original, force=False) == 0
    before = ProjectContext.discover(original)
    moved = tmp_path / "moved-study"
    original.rename(moved)
    after = ProjectContext.discover(moved)
    assert before.project_id == after.project_id
    assert before.root != after.root


def test_shared_controller_index_filters_legacy_jobs_and_rejects_foreign_actions(
    tmp_path: Path,
) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    raw = JobStore(tmp_path / "jobs")
    raw.write(record("job-first", first))
    raw.write(record("job-second", second))
    a = JobStore(raw.root, project=first)
    b = JobStore(raw.root, project=second)
    assert [item.job_id for item in a.records()] == ["job-first"]
    assert [item.job_id for item in b.records()] == ["job-second"]
    with pytest.raises(FileNotFoundError):
        a.delete("job-second")
    with pytest.raises(FileNotFoundError):
        a.write(record("job-second", first))
    a.write(a.get("job-first"))
    assert raw.get("job-first").metadata["project_id"] == first.project_id
    with pytest.raises(ValueError, match="another project"):
        b.write(raw.get("job-first"))
    assert raw.get("job-second").job_id == "job-second"


def test_catalog_namespaces_remote_storage_but_keeps_grants_and_auth_shared(tmp_path: Path) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    profile = ClusterProfile("gpu", transport="ssh", host="gpu.invalid", workspace="/remote")
    a = ClusterCatalog({"gpu": profile}, project=first)
    b = ClusterCatalog({"gpu": profile}, project=second)
    left, right = a.get("gpu"), b.get("gpu")
    assert left.storage and right.storage
    assert left.storage.run_root == f"/remote/.lambdaforge/projects/{first.project_id}/jobs"
    assert left.storage.cache_root != right.storage.cache_root
    assert left.storage.state_root != right.storage.state_root
    assert left.storage.lease_root == right.storage.lease_root == "/remote/.lambdaforge/state"
    assert left.auth == right.auth
    assert a.inspect("gpu")["profile"] == profile.to_dict()
    assert ClusterProfile.from_mapping("gpu", left.to_dict()).storage == left.storage
    assert a.get("gpu").storage == left.storage  # No repeated nesting on refresh.


def test_custom_remote_storage_is_scoped_while_local_paths_stay_project_relative(
    tmp_path: Path,
) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    policy = ClusterStoragePolicy("/state", "/scratch", "/jobs", "/datasets")
    a = policy.for_project(first.project_id, workspace="/remote")
    b = policy.for_project(second.project_id, workspace="/remote")
    assert a.dataset_root == f"/datasets/projects/{first.project_id}"
    assert a.dataset_root != b.dataset_root
    profile = ClusterProfile("local", storage=policy)
    local_a = ClusterCatalog({"local": profile}, project=first).get("local").storage
    local_b = ClusterCatalog({"local": profile}, project=second).get("local").storage
    assert local_a and local_b
    assert local_a == local_b == policy


def test_legacy_remote_job_retains_its_paths_after_project_scoping(tmp_path: Path) -> None:
    context = project(tmp_path / "a")
    profile = ClusterProfile("gpu", transport="ssh", host="gpu.invalid", workspace="/remote")
    jobs = JobService(
        ClusterCatalog({"gpu": profile}, project=context), JobStore(tmp_path / "jobs")
    )
    legacy = record("job-old", context)
    assert jobs.job_root(legacy) == "/remote/.lambdaforge/jobs"
    snapshot = replace(profile.storage, run_root="/previous/jobs")
    saved = legacy.with_updates(
        metadata={**legacy.metadata, "provider_storage": snapshot.to_dict()}
    )
    assert jobs.job_root(saved) == "/previous/jobs"


def test_recents_and_results_remain_project_local_from_subdirectories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(first.root)
    a = RecentWorkStore()
    a.remember(first.root / "experiments" / "work.yaml")
    monkeypatch.chdir(second.root)
    b = RecentWorkStore()
    b.remember(second.root / "experiments" / "work.yaml")
    assert len(b.items()) == 1
    assert len(a.items()) == 1
    assert a.items()[0]["path"] != b.items()[0]["path"]
    assert a.path.parent.name == first.project_id
    assert b.path.parent.name == second.project_id
    monkeypatch.chdir(first.root / "experiments")
    assert ResultStore().root == first.root / ".lambdaforge" / "runs"


def test_project_cli_reports_identity_without_creating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lambdaforge.cli.CommandLineInterface import CommandLineInterface

    context = project(tmp_path / "a")
    monkeypatch.chdir(context.root)
    assert CommandLineInterface.main(["project", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["project_id"] == context.project_id
    assert not (context.root / ".lambdaforge").exists()


def test_project_catalog_overrides_only_its_mirror_and_preserves_user_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    ClusterCatalog.add(
        ClusterCatalog.user_path(),
        ClusterProfile(
            "gpu",
            transport="ssh",
            host="same-host.invalid",
            workspace="/remote",
        ),
    )
    (first.root / "lambdaforge.clusters.yaml").write_text(
        "clusters:\n  gpu:\n    project_root: /remote/project-a\n"
    )
    monkeypatch.chdir(first.root / "experiments")
    a = ClusterCatalog.load()
    monkeypatch.chdir(second.root)
    b = ClusterCatalog.load()
    assert a.get("gpu").host == b.get("gpu").host == "same-host.invalid"
    assert a.get("gpu").project_root == "/remote/project-a"
    assert b.get("gpu").project_root is None
    assert a.get("gpu").storage != b.get("gpu").storage
    assert a.get("gpu").storage == ClusterCatalog.load(project=first).get("gpu").storage


def test_detached_request_preserves_project_and_resolved_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    from types import SimpleNamespace

    from lambdaforge.configuration.ConfigurationDescriptor import ConfigurationDescriptor
    from lambdaforge.controlplane.SubmissionService import SubmissionService
    from lambdaforge.execution.ResourceRequest import ResourceRequest

    context = project(tmp_path / "a")
    profile = ClusterProfile("gpu", transport="ssh", host="host.invalid", workspace="/remote")
    catalog = ClusterCatalog({"gpu": profile}, project=context)
    store = JobStore(tmp_path / "jobs", project=context)
    jobs = JobService(catalog, store)
    source = context.root / "experiments" / "work.yaml"
    monkeypatch.setattr(
        ConfigurationDescriptor,
        "from_path",
        staticmethod(
            lambda _path: SimpleNamespace(
                scientific_identity="sha256:test",
                name="same-work",
                job_type="work",
                metadata=lambda: {"source_config_path": str(source)},
            )
        ),
    )
    submission_module = importlib.import_module("lambdaforge.controlplane.SubmissionService")
    monkeypatch.setattr(submission_module.subprocess, "Popen", lambda *a, **k: None)
    handle = SubmissionService(catalog, jobs).enqueue(
        source, cluster="gpu", resources=ResourceRequest()
    )
    payload = json.loads((store.root / "submissions" / handle.job_id / "request.json").read_text())
    assert payload["project"] == context.to_dict()
    resolved = ClusterProfile.from_mapping("gpu", payload["profile"])
    assert resolved.storage == catalog.get("gpu").storage
    assert store.get(handle.job_id).metadata["project_id"] == context.project_id


def test_local_supervisor_uses_project_registry_and_common_host_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.controlplane.LocalTransport import LocalTransport
    from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler

    context = project(tmp_path / "a")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    profile = JobService._submission_profile(
        ClusterProfile("local"),
        config_path=str(context.root / "experiments" / "work.yaml"),
        work_dir=context.root,
    )
    scheduler = ProcessScheduler(LocalTransport(), profile)
    assert scheduler._dataset_registry(context.root / "experiments") == str(
        context.root / ".lambdaforge" / "datasets.json"
    )
    assert profile.storage and profile.storage.lease_root
    assert profile.storage.lease_root == str(tmp_path / "state" / "lambdaforge" / "leases")


def test_default_controller_store_is_namespaced_and_reads_only_owned_legacy_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = project(tmp_path / "a"), project(tmp_path / "b")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    legacy = JobStore(tmp_path / "state" / "lambdaforge" / "jobs")
    legacy.write(record("job-first-old", first))
    legacy.write(record("job-second-old", second))
    monkeypatch.chdir(first.root)
    a = JobStore()
    monkeypatch.chdir(second.root)
    b = JobStore()
    assert a.root != b.root
    assert a.root.parent.name == first.project_id
    assert [item.job_id for item in a.records()] == ["job-first-old"]
    assert [item.job_id for item in b.records()] == ["job-second-old"]
    a.write(record("job-first-new", first))
    assert (a.root / "job-first-new.json").is_file()
    assert not (legacy.root / "job-first-new.json").exists()
