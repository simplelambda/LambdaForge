"""Loopback provider acceptance: real native metadata reads and retained-interpreter guard."""

import hashlib
import json
import os
import site
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ControlPlaneFactory import ControlPlaneFactory
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.JobService import JobService
from lambdaforge.controlplane.JobStore import JobStore
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StudyExtensionDispatch import prepared_entrypoint, retained_profile
from lambdaforge.controlplane.StudyExtensionService import StudyExtensionService
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work.extension_runtime import check_worker_runtime, runtime_contract
from lambdaforge.work.ResultStore import ResultStore
from tests.controlplane.test_study_extension_apply import capture_preparation


@pytest.fixture(name="project")
def fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    value = project.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".lambdaforge/fixture-state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".lambdaforge/fixture-config"))
    monkeypatch.delenv("LAMBDAFORGE_CLUSTERS", raising=False)
    # Small CPU-only, disposable scientific fixture. No SSH, GPUs or dependency install.
    value["raw"]["seeds"] = [11, 23]
    value["raw"]["sweep"]["space"]["strength"] = [0.0, 0.1]
    value["path"].write_text(yaml.safe_dump(value["raw"]))
    return value


def remote_fixture(project, monkeypatch):
    context = ProjectContext.discover(project["root"])
    remote = ClusterProfile(
        "loopback",
        transport="ssh",
        host="not-a-real-cluster.invalid",
        workspace=str(project["root"] / "provider"),
        environment="managed",
    )
    catalog = ClusterCatalog({"loopback": remote}, project=context)
    profile = catalog.get("loopback")
    environment_id = "env-" + "a" * 24
    prefix = Path(profile.storage.environment_root) / environment_id
    venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
    marker = prefix / ".lambdaforge-environment.json"
    marker.write_text(json.dumps({"environment_id": environment_id, "packages": ["fixture.whl"]}))
    # Reuse already installed test dependencies via PYTHONPATH; install nothing.
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [
                str(project["root"] / "src"),
                str(Path(__file__).resolve().parents[2] / "src"),
                *site.getsitepackages(),
            ]
        ),
    )

    class Loopback(ControlPlaneFactory):
        def transport(self, profile):
            return LocalTransport()

    jobs = JobService(
        catalog,
        factory=Loopback(),
        store=JobStore(project["root"] / ".lambdaforge/controller/jobs", project=context),
    )
    svc = StudyExtensionService(
        project=context,
        catalog=catalog,
        jobs=jobs,
        results=ResultStore(project["root"] / ".lambdaforge/runs"),
    )
    handle = jobs.reserve(
        cluster="loopback",
        resources=ResourceRequest(cpu_cores=1),
        config_path=project["path"],
        metadata={"environment_id": environment_id},
    )
    work = Path(profile.storage.run_root) / handle.job_id / "work"
    work.mkdir(parents=True)
    env = {
        "LAMBDAFORGE_JOB_ID": handle.job_id,
        "LAMBDAFORGE_CLUSTER": "loopback",
        "LAMBDAFORGE_RUN_ROOT": str(work / ".lambdaforge/runs"),
        "LAMBDAFORGE_JOB_RESULT_PATH": str(work.parent / "result.json"),
    }
    script = (
        "from lambdaforge.work import WorkConfig,WorkRunner\n"
        "import lambdaforge.work.runner as runner\n"
        "def inline(prepared,*,results,on_result=None,**kwargs):\n"
        " for item in prepared:\n"
        "  result=runner._execute_run(item); results.append(result)\n"
        "runner._execute_cpu_isolated_runs=inline\n"
        "import sys\nWorkRunner().run(WorkConfig.from_yaml(sys.argv[1]))\n"
    )
    result = LocalTransport().run(
        (
            "env",
            *(f"{key}={value}" for key, value in env.items()),
            str(prefix / "bin/python"),
            "-c",
            script,
            str(project["path"]),
        ),
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    record = jobs.get(handle.job_id, refresh=False)
    pointer = json.loads((work.parent / "result.json").read_text())
    jobs.store.write(
        record.with_updates(
            state=JobState.SUCCEEDED,
            work_dir=str(work),
            command=(str(prefix / "bin/python"), "-m", "lambdaforge", "run", str(project["path"])),
            metadata={
                **record.metadata,
                "provider_storage": profile.storage.to_dict(),
                "source_config_path": str(project["path"]),
                "scientific_identity": pointer["scientific_fingerprint"],
            },
        )
    )
    return svc, handle.job_id, prefix, work


@pytest.mark.parametrize("conditional", [False, True])
def test_remote_preview_apply_and_actual_native_guard_reuse_exact_prefix(
    project, monkeypatch, conditional
):
    if conditional:
        project["raw"]["sweep"]["space"]["fail"] = {
            "values": [False], "when": {"strength": 0.0},
        }
        project["path"].write_text(yaml.safe_dump(project["raw"]))
    svc, job_id, prefix, work = remote_fixture(project, monkeypatch)
    raw = yaml.safe_load(project["path"].read_text())
    if conditional:
        raw["sweep"]["include"] = [{"strength": 0.2}]
    else:
        raw["sweep"]["space"]["strength"].append(0.2)
    project["path"].write_text(yaml.safe_dump(raw))
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    plan = svc.preview(job_id, project["path"], target="loopback")
    assert plan["placement"] == "remote" and plan["apply_supported"]
    assert plan["comparison"]["counts"] == {"reused": 4, "new": 2, "total": 6}
    assert plan["runtime_compatibility"]["retained_runtime"]["python"] == str(prefix / "bin/python")
    assert (
        bool(plan["runtime_compatibility"]["retained_runtime"].get("sparse_sweeps")) == conditional
    )
    assert before == {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    capture_preparation(monkeypatch)
    accepted = svc.apply(job_id, project["path"], target="loopback", plan_hash=plan["plan_hash"])
    proof = plan["runtime_compatibility"]["retained_runtime"]
    narrowed = retained_profile(svc.catalog.get("loopback"), proof)
    assert (
        narrowed.environment == "existing"
        and narrowed.gpu_access == svc.catalog.get("loopback").gpu_access
    )
    delta_work = Path(svc.catalog.get("loopback").storage.run_root) / accepted["job_id"] / "work"
    delta_work.mkdir(parents=True)
    # Mirror the native bundle: its own project namespace and embedded submitting code.
    from lambdaforge.reproducibility.CodeIdentity import CodeIdentity

    (delta_work / "pyproject.toml").write_bytes((project["root"] / "pyproject.toml").read_bytes())
    (delta_work / "code-identity.json").write_text(json.dumps(
        CodeIdentity.capture(project["root"]).to_dict()
    ))
    config = delta_work / "config.yaml"
    config.write_bytes(Path(accepted["snapshot"]).read_bytes())
    prepared = SimpleNamespace(
        profile=svc.catalog.get("loopback"),
        python=str(prefix / "bin/python"),
        work_dir=str(delta_work),
        transport=LocalTransport(),
        config=str(config),
        dry_run=False,
    )
    # Exercise the actual durable worker's retained-profile hook. Only provider preparation
    # is intercepted here; the generated guard below runs native science in a subprocess.
    import importlib

    worker = importlib.import_module("lambdaforge.controlplane.SubmissionWorker")
    dispatched = []

    class ControlPlaneProbe:
        def __init__(self, catalog, **kwargs):
            self.catalog = catalog

        def submit(self, config, **kwargs):
            actual_profile = self.catalog.get("loopback")
            assert actual_profile.python == proof["python"]
            assert actual_profile.environment == "existing"
            assert actual_profile.gpu_access == svc.catalog.get("loopback").gpu_access
            assert kwargs["run_arguments"] == ()
            dispatched.append(kwargs["entrypoint_builder"](prepared))

    monkeypatch.setattr(worker, "ControlPlane", ControlPlaneProbe)
    request = svc.jobs.store.root / "submissions" / accepted["job_id"] / "request.json"
    assert worker.serve(request) == 0 and len(dispatched) == 1
    attestation = svc.jobs.get(accepted["job_id"], refresh=False).metadata[
        "prepared_extension_attestation"
    ]
    entry = prepared_entrypoint(prepared, proof, attestation=attestation)
    assert tuple(dispatched[0]) == tuple(entry)
    marker = prefix / ".lambdaforge-environment.json"
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == proof["marker_sha256"]
    result = LocalTransport().run(
        (
            "env",
            "LAMBDAFORGE_EXECUTION_MODE=worker",
            "LAMBDAFORGE_BUNDLE=1",
            f"LAMBDAFORGE_JOB_ID={accepted['job_id']}",
            "LAMBDAFORGE_CLUSTER=loopback",
            f"LAMBDAFORGE_RUN_ROOT={delta_work / '.lambdaforge/runs'}",
            f"LAMBDAFORGE_JOB_RESULT_PATH={delta_work.parent / 'result.json'}",
            str(prefix / "bin/python"),
            "-m",
            *entry,
        ),
        cwd=delta_work,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    record = svc.jobs.get(accepted["job_id"], refresh=False)
    svc.jobs.store.write(
        record.with_updates(
            state=JobState.SUCCEEDED,
            work_dir=str(delta_work),
            metadata={
                **record.metadata,
                "environment_id": proof["environment_id"],
                "provider_storage": svc.catalog.get("loopback").storage.to_dict(),
            },
        )
    )
    final = svc.refresh(accepted["study_series_id"])
    assert final["operations"][0]["state"] == "succeeded"
    assert svc.overview(accepted["study_series_id"])["counts"]["completed"] == 6
    assert before == {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}


def test_worker_hardware_check_ignores_gpu_ordinal_but_not_numerics(monkeypatch):
    from lambdaforge.EnvironmentManifest import EnvironmentManifest

    env = {
        "lambdaforge_version": "test",
        "python": {"version": "test"},
        "platform": {},
        "packages": {"torch": "x"},
        "torch": {
            "version": "x",
            "devices": [
                {"index": 0, "name": "H200", "capability": [9, 0], "total_memory_bytes": 140000},
            ],
        },
    }
    expected = runtime_contract(env)
    env["torch"]["devices"][0]["index"] = 3
    monkeypatch.setattr(
        EnvironmentManifest,
        "capture",
        classmethod(lambda cls: SimpleNamespace(to_dict=lambda: env)),
    )
    check_worker_runtime(expected)
    env["torch"]["devices"][0]["name"] = "H100"
    with pytest.raises(ValueError, match="hardware differs"):
        check_worker_runtime(expected)


def test_old_retained_parser_fails_before_sparse_submission(tmp_path, monkeypatch):
    import platform
    import sys
    from importlib.metadata import version

    from lambdaforge.work import config
    from lambdaforge.work.extension_runtime import inspect_retained_runtime

    (tmp_path / ".lambdaforge-environment.json").write_text(json.dumps({
        "environment_id": "env-fixture", "packages": ["fixture"],
    }))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    contract = {
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "processor": platform.processor()},
        "packages": {"lambdaforge": version("lambdaforge")},
        "framework": version("lambdaforge"),
    }
    monkeypatch.setattr(config, "FIXED_SWEEP_POINTS_VERSION", 0)
    with pytest.raises(ValueError, match="lacks native sparse sweep support"):
        inspect_retained_runtime(str(tmp_path), "env-fixture", [contract], require_sparse=True)
    assert "sparse_sweeps" not in inspect_retained_runtime(
        str(tmp_path), "env-fixture", [contract]
    )
