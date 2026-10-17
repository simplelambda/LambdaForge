"""Real scalar Work evidence and read-only extension plans in disposable projects."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from lambdaforge.cli.parser import build_parser
from lambdaforge.cli.study_extensions import run_study_extension_command
from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.StudyExtensionService import StudyExtensionService
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.ResultStore import ResultStore
from lambdaforge.work.StudySeriesStore import StudySeriesStore


@pytest.fixture
def project(tmp_path, monkeypatch):
    name = "extension_fixture_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "extension-test"\nversion = "1.0.0"\n'
    )
    (tmp_path / ".gitignore").write_text(".lambdaforge/\n__pycache__/\n")
    source = tmp_path / "src" / name
    source.mkdir(parents=True)
    code = source / "__init__.py"
    code.write_text(
        "from pathlib import Path\nfrom lambdaforge import Work\n"
        "class Sweep(Work):\n"
        "    def run(self, strength: float, source: Path | None = None, fail: bool = False):\n"
        "        if fail: raise RuntimeError('test failure')\n"
        "        score = 1.0 - strength + self.seed / 100000.0\n"
        "        self.metrics.log('score', score, step=1)\n"
        "        self.checkpoints.save_json('state', {'strength': strength, 'seed': self.seed})\n"
        "        self.outputs.file('toy', filename='toy.json').write_json({'score': score})\n"
        "        return {'score': score}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    path = experiments / "sweep.yaml"
    raw = {
        "name": "regularization",
        "run": name + ".Sweep",
        "seeds": [11, 23, 47, 59, 71],
        "resources": {"cpu": 1},
        "objective": {"metric": "score", "mode": "max"},
        "sweep": {"space": {"strength": [0.0, 0.1, 0.2, 0.3]}},
    }
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    for key in (
        "LAMBDAFORGE_RUN_ROOT",
        "LAMBDAFORGE_CACHE_ROOT",
        "LAMBDAFORGE_STUDY_PATH",
        "LAMBDAFORGE_PROGRESS_PATH",
        "LAMBDAFORGE_BUNDLE",
        "LAMBDAFORGE_JOB_RESULT_PATH",
        "LAMBDAFORGE_PRODUCT_ROOT",
        "LAMBDAFORGE_STORAGE_POLICY",
    ):
        monkeypatch.delenv(key, raising=False)
    runner = importlib.import_module("lambdaforge.work.runner")
    monkeypatch.setattr(runner, "_seed", lambda seed: None)
    monkeypatch.setattr(runner.EnvironmentManifest, "_torch_info", staticmethod(lambda: {}))

    def inline(prepared, *, results, on_result=None, **kwargs):
        queue = list(prepared)
        while queue:
            value = queue.pop(0)
            result = runner._execute_run(value)
            results.append(result)
            if on_result:
                queue = list(on_result(result, queue, ()))

    monkeypatch.setattr(runner, "_execute_cpu_isolated_runs", inline)
    return {"root": tmp_path, "path": path, "raw": raw, "code": code}


def execute(project):
    return WorkRunner().run(WorkConfig.from_yaml(project["path"]))


def service(project):
    from lambdaforge.controlplane.JobService import JobService
    from lambdaforge.controlplane.JobStore import JobStore

    context = ProjectContext.discover(project["root"])
    catalog = ClusterCatalog(
        {"local": ClusterProfile("local", python=sys.executable)}, project=context
    )
    return StudyExtensionService(
        project=context,
        catalog=catalog,
        results=ResultStore(project["root"] / ".lambdaforge/runs"),
        jobs=JobService(
            catalog,
            store=JobStore(project["root"] / ".lambdaforge/controller/jobs", project=context),
        ),
    )


def extend_yaml(project):
    raw = {**project["raw"], "sweep": {"space": {"strength": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]}}}
    project["path"].write_text(yaml.safe_dump(raw, sort_keys=False))


def test_reservation_is_atomic_idempotent_and_protects_native_evidence(project):
    original = execute(project)
    extend_yaml(project)
    plan = service(project).preview(original.execution_id, project["path"])
    store = StudySeriesStore(ProjectContext.discover(project["root"]))
    first = store.reserve(plan, request_id="extension-" + "a" * 32)
    assert first["created"] and first["state"] == "reserved"
    assert first["job_id"] is None and first["execution_id"] is None
    files = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    repeated = store.reserve(plan, request_id="extension-" + "a" * 32)
    duplicate = store.reserve(plan, request_id="extension-" + "b" * 32)
    assert not repeated["created"] and duplicate == repeated
    assert files == {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    record = store.read(plan["study_series_id"])
    assert len(record["revisions"]) == 2 and len(record["history"]) == 1
    assert len(record["revisions"][0]["evidence"]) == 20
    assert len(record["revisions"][1]["new_cells"]) == 15
    for apply in (False, True):
        with pytest.raises(ValueError, match="retained by a Study series"):
            service(project).results.delete(original.execution_id, apply=apply)
    assert (original.execution_dir / "result.json").is_file()


def test_reserved_revision_rejects_other_plan_and_reused_request(project):
    original = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(original.execution_id, project["path"])
    store = StudySeriesStore(svc.project)
    store.reserve(plan, request_id="extension-" + "a" * 32)
    raw = yaml.safe_load(project["path"].read_text())
    raw["sweep"]["space"]["strength"].append(1.5)
    project["path"].write_text(yaml.safe_dump(raw))
    different = svc.preview(original.execution_id, project["path"])
    with pytest.raises(ValueError, match="reused for another plan"):
        store.reserve(different, request_id="extension-" + "a" * 32)
    with pytest.raises(ValueError, match="pending Study extension"):
        store.reserve(different, request_id="extension-" + "b" * 32)


def test_series_journal_corruption_and_missing_authority_fail_closed(project):
    original = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(original.execution_id, project["path"])
    store = StudySeriesStore(svc.project)
    store.reserve(plan, request_id="extension-" + "a" * 32)
    path = store.root / plan["study_series_id"] / "series.json"
    contents = path.read_bytes()
    value = json.loads(contents)
    value["revisions"][0]["evidence"][0]["metrics"]["score"] = -1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Corrupt"):
        store.read(plan["study_series_id"])
    with pytest.raises(ValueError, match="Corrupt"):
        svc.results.delete(original.execution_id)
    path.unlink()  # Disposable fixture only: simulate loss of the authoritative journal.
    with pytest.raises(ValueError, match="journal is missing"):
        store.reserve(plan, request_id="extension-" + "a" * 32)
    with pytest.raises(ValueError, match="journal is missing"):
        svc.results.delete(original.execution_id)


def test_reservation_rejects_symlink_and_changed_attempt_without_writes(project, tmp_path):
    original = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(original.execution_id, project["path"])
    store = StudySeriesStore(svc.project)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    store.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic paths"):
        store.reserve(plan, request_id="extension-" + "a" * 32)
    store.root.unlink()
    path = Path(plan["reused_evidence"][0]["result_path"])
    value = json.loads(path.read_text())
    value["metrics"]["score"] = -1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Attempt changed"):
        store.reserve(plan, request_id="extension-" + "a" * 32)
    assert not store.root.exists()


def test_concurrent_reservations_keep_one_revision_and_one_request(project):
    original = execute(project)
    extend_yaml(project)
    svc = service(project)
    plan = svc.preview(original.execution_id, project["path"])
    store = StudySeriesStore(svc.project)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda letter: store.reserve(plan, request_id="extension-" + letter * 32),
                ("a", "b"),
            )
        )
    assert sum(result["created"] for result in results) == 1
    assert results[0]["request_id"] == results[1]["request_id"]
    assert len(store.read(plan["study_series_id"])["history"]) == 1


def test_cli_preview_is_native_read_only_and_shows_exact_cell_counts(project, monkeypatch, capsys):
    original = execute(project)
    extend_yaml(project)
    svc = service(project)
    monkeypatch.setattr(ClusterCatalog, "load", classmethod(lambda cls, *a, **kw: svc.catalog))
    args = build_parser().parse_args(
        [
            "studies",
            "extend",
            original.execution_id,
            "--config",
            str(project["path"]),
            "--dry-run",
            "--json",
        ]
    )
    assert run_study_extension_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["comparison"]["counts"] == {"reused": 20, "new": 15, "total": 35}
    assert result["apply_supported"] and not result["series_persisted"]
    assert not (project["root"] / ".lambdaforge/study-series").exists()


def test_cli_apply_refuses_without_loading_any_target_or_work(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("Unsupported apply must not inspect a provider or import Work.")

    monkeypatch.setattr(ClusterCatalog, "load", unexpected)
    args = build_parser().parse_args(
        [
            "studies",
            "extend",
            "execution-unknown",
            "--config",
            str(tmp_path / "absent.yaml"),
            "--apply",
        ]
    )
    with pytest.raises(ValueError, match="apply requires --plan-hash"):
        run_study_extension_command(args)
    assert list(tmp_path.iterdir()) == []


def test_no_op_reservation_writes_nothing(project):
    original = execute(project)
    svc = service(project)
    plan = svc.preview(original.execution_id, project["path"])
    store = StudySeriesStore(svc.project)
    result = store.reserve(plan, request_id="extension-" + "a" * 32)
    assert result == {"state": "no-op", "request_id": None, "created": False}
    assert not store.root.exists()


def test_twenty_real_cells_preview_fifteen_new_without_mutation_or_asset_reads(
    project, monkeypatch
):
    original = execute(project)
    assert original.status == "succeeded" and len(original.runs) == 20
    extend_yaml(project)
    files = {
        path: path.read_bytes() for path in original.execution_dir.rglob("*") if path.is_file()
    }
    plan = service(project).preview(original.execution_id, project["path"])
    assert plan["comparison"]["counts"] == {"reused": 20, "new": 15, "total": 35}
    assert {
        (row["execution_id"], row["run_id"], row["attempt_id"]) for row in plan["reused_evidence"]
    } == {(run.execution_id, run.run_id, run.attempt_id) for run in original.runs}
    assert files == {
        path: path.read_bytes() for path in original.execution_dir.rglob("*") if path.is_file()
    }
    assert not (project["root"] / ".lambdaforge/study-series").exists()
    assert plan["apply_supported"] is True and plan["series_persisted"] is False
    assert (
        plan["plan_hash"]
        == service(project).preview(original.execution_id, project["path"])["plan_hash"]
    )
    native_read = Path.read_bytes

    def read(path):
        assert "checkpoints" not in path.parts and path.name != "toy.json"
        return native_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    service(project).preview(original.execution_id, project["path"])


def git(root, *args):
    return subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)


def commit(project):
    root = project["root"]
    git(root, "init")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "disposable fixture",
    )


def test_clean_git_origin_same_head_only_authorized_yaml_is_compatible(project):
    commit(project)
    original = execute(project)
    extend_yaml(project)
    plan = service(project).preview(original.execution_id, project["path"])
    assert plan["code_compatibility"]["kind"] == "authorized-yaml-only"
    assert plan["code_compatibility"]["origin"]["dirty"] is False
    assert plan["code_compatibility"]["current"]["dirty"] is True
    assert (
        json.loads((original.execution_dir / "execution.json").read_text())["code_identity"][
            "dirty"
        ]
        is False
    )


@pytest.mark.parametrize("change", ["code", "dependency", "unknown"])
def test_git_yaml_edit_plus_any_other_change_rejects(project, change):
    commit(project)
    original = execute(project)
    extend_yaml(project)
    path = {
        "code": project["code"],
        "dependency": project["root"] / "pyproject.toml",
        "unknown": project["root"] / "other.txt",
    }[change]
    path.write_text(path.read_text() + "\n# changed\n" if path.exists() else "unknown\n")
    with pytest.raises(ValueError, match="Executable/dependency|unknown Git"):
        service(project).preview(original.execution_id, project["path"], code_policy="strict")


def test_dirty_origin_cannot_be_reconstructed_after_yaml_edit(project):
    commit(project)
    project["code"].write_text(project["code"].read_text() + "\n# preexisting dirty science\n")
    original = execute(project)
    extend_yaml(project)
    with pytest.raises(ValueError, match="dirty origin"):
        service(project).preview(original.execution_id, project["path"], code_policy="strict")


def test_nongit_source_change_rejects_even_when_package_version_unchanged(project):
    original = execute(project)
    extend_yaml(project)
    project["code"].write_text(project["code"].read_text() + "\n# changed science\n")
    with pytest.raises(ValueError, match="Changed/unverifiable"):
        service(project).preview(original.execution_id, project["path"], code_policy="strict")


def test_failed_origin_requires_recovery_not_extension(project):
    raw = {**project["raw"], "with": {"fail": True}, "execution": {"failure_retries": 0}}
    project["path"].write_text(yaml.safe_dump(raw))
    original = execute(project)
    extend_yaml(project)
    with pytest.raises(ValueError, match="native retry"):
        service(project).preview(original.execution_id, project["path"])


def test_symlink_yaml_and_other_project_reject_before_launch(project, tmp_path):
    original = execute(project)
    linked = project["path"].with_name("linked.yaml")
    linked.symlink_to(project["path"])
    with pytest.raises(ValueError, match="non-symbolic"):
        service(project).preview(original.execution_id, linked)
    nested = project["root"] / "other-project"
    nested.mkdir()
    (nested / "pyproject.toml").write_text('[project]\nname="other"\nversion="1"\n')
    path = nested / "study.yaml"
    path.write_text(project["path"].read_text())
    with pytest.raises(ValueError, match="another project"):
        service(project).preview(original.execution_id, path)


def test_relative_inputs_same_content_can_be_rebound_but_changed_bytes_reject(project):
    data = project["root"] / "data"
    data.mkdir()
    source = data / "input.txt"
    source.write_text("immutable input\n")
    raw = {**project["raw"], "with": {"source": {"file": "../data/input.txt"}}}
    project["path"].write_text(yaml.safe_dump(raw))
    original = execute(project)
    changed = {
        **raw,
        "with": {"source": {"file": str(source)}},
        "sweep": {"space": {"strength": [0.0, 0.1, 0.2, 0.3, 0.5]}},
    }
    project["path"].write_text(yaml.safe_dump(changed))
    assert (
        service(project).preview(original.execution_id, project["path"])["comparison"]["counts"][
            "new"
        ]
        == 5
    )
    source.write_text("changed input\n")
    with pytest.raises(ValueError, match="unchanged configuration"):
        service(project).preview(original.execution_id, project["path"])


@pytest.mark.parametrize("corruption", ["metrics", "fingerprint", "missing-aggregate-cell"])
def test_corrupt_or_unattested_evidence_is_not_reused(project, corruption):
    original = execute(project)
    extend_yaml(project)
    path = original.runs[0].run_dir / "result.json"
    if corruption == "missing-aggregate-cell":
        path = original.execution_dir / "result.json"
    value = json.loads(path.read_text())
    if corruption == "metrics":
        value["metrics"]["score"] = 999
    elif corruption == "fingerprint":
        value["scientific_fingerprint"] = "sha256:" + "0" * 64
    else:
        value["runs"].pop(0)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="scientific fingerprint|authoritative Attempt"):
        service(project).preview(original.execution_id, project["path"])


def test_runtime_dependency_change_cannot_be_waived(project, monkeypatch):
    original = execute(project)
    extend_yaml(project)
    compatibility = importlib.import_module("lambdaforge.work.extension_compatibility")
    native = compatibility.version
    monkeypatch.setattr(
        compatibility, "version", lambda name: "different" if name == "numpy" else native(name)
    )
    with pytest.raises(ValueError, match="Dependency.*changed"):
        service(project).preview(original.execution_id, project["path"])


def test_preview_noop_and_full_budget_delta_are_explicit(project):
    original = execute(project)
    preview = service(project).preview(original.execution_id, project["path"])
    assert preview["comparison"]["kind"] == "no-op"
    assert preview["comparison"]["counts"]["new"] == 0
    extend_yaml(project)
    raw = yaml.safe_load(project["path"].read_text())
    raw["execution"] = {"max_runs": 35}
    project["path"].write_text(yaml.safe_dump(raw))
    preview = service(project).preview(original.execution_id, project["path"])
    assert preview["budget"]["historical_attempts"] == 20
    assert preview["budget"]["remaining_runs"] == 15
    assert preview["comparison"]["delta_configuration"]["execution"]["max_runs"] == 15


def test_active_controller_lock_blocks_preview_without_creating_metadata(project):
    from lambdaforge.runtime.CrossProcessFileLock import CrossProcessFileLock

    original = execute(project)
    extend_yaml(project)
    with CrossProcessFileLock(
        original.execution_dir / ".controller.lock",
        shared=False,
        create=False,
        timeout_seconds=0.1,
        poll_interval_seconds=0.01,
    ):
        with pytest.raises(TimeoutError):
            service(project).preview(original.execution_id, project["path"])


def test_unknown_job_ownership_is_not_permission_to_extend(project, monkeypatch):
    from types import SimpleNamespace

    from lambdaforge.controlplane.JobService import JobService

    original = execute(project)
    extend_yaml(project)
    manifest = original.execution_dir / "execution.json"
    value = json.loads(manifest.read_text())
    value["job_id"] = "job-fixture-unknown"
    manifest.write_text(json.dumps(value))
    monkeypatch.setattr(
        JobService,
        "get",
        lambda self, *args, **kwargs: SimpleNamespace(
            cluster="local", state=SimpleNamespace(value="unknown"), metadata={}
        ),
    )
    with pytest.raises(ValueError, match="UNKNOWN is not permission"):
        service(project).preview(original.execution_id, project["path"])
