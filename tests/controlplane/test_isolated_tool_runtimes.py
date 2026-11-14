"""Native isolated tools: disposable offline prefixes, no project or cluster mutation."""

from __future__ import annotations

import ensurepip
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import pytest

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterService import ClusterService
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.controlplane.ToolRuntimeService import (
    ToolRuntimeService,
    validate_verification,
    verification_code,
    verification_policy,
)
from lambdaforge.controlplane.ToolRuntimeSpecification import ToolRuntimeSpecification
from lambdaforge.work.tools import ToolExecutionError, ToolService


class OfflineTransport(LocalTransport):
    """Use real venv/pip, but forbid networking in these regression fixtures."""

    def __init__(self):
        self.installs = 0

    def run(self, command, **kwargs):
        if "pip" in command and "install" in command:
            self.installs += 1
            command = (*command, "--no-index")
        return super().run(command, **kwargs)


class Log:
    def __init__(self):
        self.lines = []

    def emit(self, message, **kwargs):
        self.lines.append(str(message))


def declaration(**changes):
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "requirements": [f"pip=={ensurepip.version()}"],
        "imports": ["json"],
        **changes,
    }


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("isolated-tools")
    project = root / "project"
    project.mkdir()
    (project / "worker.py").write_text(
        "import json,os,pathlib,sys\n"
        "source=pathlib.Path(sys.argv[1]);output=pathlib.Path(sys.argv[2])\n"
        "output.write_text(source.read_text().upper())\n"
        "print(json.dumps({'prefix':sys.prefix,'cuda':os.getenv('CUDA_VISIBLE_DEVICES'),"
        "'threads':os.getenv('OMP_NUM_THREADS'),'pythonpath':os.getenv('PYTHONPATH')}))\n"
        "print('tool stderr',file=sys.stderr)\n"
    )
    profile = ClusterProfile("local", workspace=str(root / "host"), python=sys.executable)
    spec = ToolRuntimeSpecification.from_mapping(
        "example", declaration(entrypoints={"extract": "worker.py"}), project
    )
    transport = OfflineTransport()
    service = ToolRuntimeService()
    (binding,) = service.prepare((spec,), profile, transport)
    return root, project, profile, spec, transport, service, binding


@pytest.mark.parametrize(
    "changes",
    [
        {"requirements": ["numpy>=1"]},
        {"requirements": ["pkg @ git+https://example.org/x.git@main"]},
        {"source_packages": ["unreviewed"]},
        {"install_command": "pip install x"},
        {"extra_index_urls": "https://example.org/simple"},
        {"index_url": "https://user:secret@example.org/simple"},
        {"requirements": ["torch==2.6.0"]},
        {"entrypoints": {"x": "../outside.py"}},
    ],
)
def test_unsafe_or_unpinned_declarations_fail(tmp_path, changes):
    with pytest.raises(ValueError):
        ToolRuntimeSpecification.from_mapping("example", declaration(**changes), tmp_path)


def test_identity_separates_entrypoint_code_from_packages(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("print(1)")
    first = ToolRuntimeSpecification.from_mapping(
        "example", declaration(entrypoints={"worker": "worker.py"}), tmp_path
    )
    descriptor = first.descriptor()
    script.write_text("print(2)")
    assert (
        first.contract_id
        == ToolRuntimeSpecification.from_mapping(
            "example", declaration(entrypoints={"worker": "worker.py"}), tmp_path
        ).contract_id
    )
    assert descriptor != first.descriptor()
    with pytest.raises(TypeError):
        first.policy["imports"].append("other")


def test_native_provision_reuse_and_independent_inventory(prepared):
    root, _, profile, spec, transport, service, binding = prepared
    installs = transport.installs
    (again,) = service.prepare((spec,), profile, transport)
    assert again["reused"] and transport.installs == installs
    assert again["environment_id"] == binding["environment_id"]
    assert again["resolution_id"] == binding["resolution_id"]
    marker = json.loads(
        (Path(binding["python"]).parent.parent / ".lambdaforge-environment.json").read_text()
    )
    assert "lambdaforge" not in {p["name"] for p in marker["installed_packages"]}
    assert not (root / "host/.lambdaforge/state/active-environment").exists()


def test_public_execution_file_exchange_logs_grants_and_provenance(prepared, monkeypatch):
    root, _, profile, _, transport, service, binding = prepared
    path = service.bind_job(profile, transport, "job-fixture", (binding,))
    monkeypatch.setenv("LAMBDAFORGE_TOOL_RUNTIMES_PATH", path)
    grant = "GPU-opaque-a,GPU-opaque-c"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", grant)
    monkeypatch.setenv("PYTHONPATH", "/must/not/inject")
    log = Log()
    owner = ToolService(log, max_threads=2)
    runtime = owner.runtime("example")
    source, output = root / "input with space.txt", root / "output.txt"
    source.write_text("original")
    result = runtime.run(["python", runtime.entrypoint("extract"), source, output])
    observed = json.loads(result.stdout)
    assert observed == {
        "prefix": str(Path(binding["python"]).parent.parent),
        "cuda": grant,
        "threads": "2",
        "pythonpath": None,
    }
    assert source.read_text() == "original" and output.read_text() == "ORIGINAL"
    assert "tool stderr" in result.stderr
    assert any("tool stderr" in v for v in log.lines)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == grant
    assert owner.provenance[0]["resolution_identity"] == binding["resolution_id"]
    for env in ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, {"OMP_NUM_THREADS": "99"}):
        with pytest.raises(ValueError, match="override"):
            runtime.run(["python", "-c", "pass"], env=env)
    with pytest.raises(ValueError, match="CPU"):
        runtime.run(["python", "-c", "pass"], threads=3)
    with pytest.raises(ValueError, match="named members"):
        runtime.run(["../outside", "argument"])
    with pytest.raises(ToolExecutionError) as error:
        runtime.run(["python", "-c", "import sys;print('failure');sys.exit(7)"])
    assert error.value.result.returncode == 7
    assert "failure" in error.value.result.stdout
    with pytest.raises(RuntimeError, match="timeout"):
        runtime.run(["python", "-c", "import time;time.sleep(20)"], timeout=0.15)


def test_cancellation_stops_only_owned_process(prepared, monkeypatch):
    root, _, profile, _, transport, service, binding = prepared
    monkeypatch.setenv(
        "LAMBDAFORGE_TOOL_RUNTIMES_PATH",
        service.bind_job(profile, transport, "job-cancel", (binding,)),
    )
    stop = Event()
    owner = ToolService(Log(), max_threads=1, stop_check=stop.is_set)
    runtime = owner.runtime("example")
    runtime.preflight()
    sentinel = root / "ready.txt"
    errors = []

    def run():
        try:
            runtime.run(
                [
                    "python",
                    "-c",
                    "import pathlib,sys,time;"
                    "pathlib.Path(sys.argv[1]).write_text('ready');time.sleep(30)",
                    sentinel,
                ]
            )
        except InterruptedError as error:
            errors.append(error)

    worker = Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 10
    while not sentinel.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sentinel.exists()
    stop.set()
    worker.join(timeout=10)
    assert not worker.is_alive() and errors


def test_missing_runtime_has_bootstrap_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMBDAFORGE_TOOL_RUNTIMES_PATH", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="lf clusters bootstrap"):
        ToolService(Log()).runtime("slae")


def test_source_policy_rejects_wrong_abi_or_source(tmp_path):
    spec = ToolRuntimeSpecification.from_mapping(
        "example",
        declaration(
            requirements=["torch-scatter==2.1.2+pt26cpu"],
            wheel_sources=[
                {
                    "url": "https://data.pyg.org/whl/torch-2.6.0+cpu.html",
                    "packages": ["torch-scatter"],
                    "build_tag": "pt26cpu",
                }
            ],
        ),
        tmp_path,
    )
    item = {
        "metadata": {"name": "torch-scatter", "version": "2.1.2+pt26cpu"},
        "download_info": {"url": "https://data.pyg.org/whl/pkg.whl"},
    }
    ToolRuntimeService.validate_sources(spec.policy, {"install": [item]})
    item["download_info"]["url"] = "https://other.org/whl/pkg.whl"
    with pytest.raises(ValueError, match="origin"):
        ToolRuntimeService.validate_sources(spec.policy, {"install": [item]})
    item["download_info"]["url"] = "https://data.pyg.org/whl/pkg.tar.gz"
    with pytest.raises(ValueError, match="compilation"):
        ToolRuntimeService.validate_sources(spec.policy, {"install": [item]})


def test_bootstrap_dry_run_exposes_tool_without_preparation(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="isolated-fixture"\nversion="0.0.0"\n'
        '[tool.lambdaforge.tool_runtimes.example]\npython="3.10"\n'
        'requirements=["pip==23.0.1"]\n'
    )
    profile = ClusterProfile("local", workspace=str(tmp_path / "host"), python=sys.executable)
    monkeypatch.setattr(ToolRuntimeService, "prepare", lambda *a, **k: pytest.fail("installed"))
    result = ClusterService(ClusterCatalog({"local": profile})).bootstrap(
        "local", project=tmp_path, dry_run=True
    )
    assert result.planned and result.to_dict()["tool_runtimes"][0]["name"] == "example"
    assert not (tmp_path / "host").exists()


def test_storage_protects_alias_and_live_use_then_allows_explicit_gc(prepared):
    _, _, profile, _, _, _, binding = prepared
    descriptor = profile.storage.to_dict()
    roots = StorageOperations._roots(descriptor)
    references = StorageOperations.environment_references(roots)
    assert binding["environment_id"] in references["environments"]
    assert binding["entrypoint_bundle_id"] in references["tool_bundles"]
    from lambdaforge.work.tool_runtime import IsolatedToolRuntime

    runtime = IsolatedToolRuntime(binding, ToolService(Log()))
    with runtime._lease():
        assert StorageOperations._tool_runtime_busy(runtime.prefix)
    assert not StorageOperations._tool_runtime_busy(runtime.prefix)
    assert binding["environment_id"] not in {
        row["environment_id"]
        for row in StorageOperations.prune_environments(descriptor, (), apply=False)["candidates"]
    }


def test_real_bootstrap_existing_environment_prepares_declared_tools(tmp_path, prepared):
    _, _, profile, _, transport, service, _ = prepared
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="tool-fixture"\nversion="0.0.0"\n'
        "[tool.lambdaforge.tool_runtimes.bootstrap]\n"
        f'python="{sys.version_info.major}.{sys.version_info.minor}"\n'
        f'requirements=["pip=={ensurepip.version()}"]\n'
    )
    # Reuse a native offline transport; the main prefix is the existing test environment.
    cluster_service = ClusterService(ClusterCatalog({"local": profile}))
    cluster_service.factory.transport = lambda _: transport
    result = cluster_service.bootstrap("local", project=tmp_path)
    assert result.tool_runtimes[0]["name"] == "bootstrap"
    assert result.environment_id == "existing"
    assert result.tool_runtimes[0]["python"] != sys.executable


@pytest.mark.parametrize(
    "grant,available,expected",
    [
        ("", True, "provider-owned"),
        ("GPU-a", False, "no CPU fallback"),
        ("GPU-a,MIG-opaque", True, None),
    ],
)
def test_cuda_preflight_fail_closed_without_rewriting_grants(grant, available, expected):
    # Stub only the GPU driver interaction; real CUDA acceptance is explicitly not claimed.
    code = (
        "import sys,types;from types import SimpleNamespace as N;"
        "sys.modules['torch']=N(version=N(cuda='12.4'),"
        f"cuda=N(is_available=lambda:{available!r}),"
        "ones=lambda *a,**k:N(item=lambda:1));"
    )
    policy = {"requirements": [], "imports": [], "wheel_sources": [], "torch_cuda": "12.4"}
    command = (
        sys.executable,
        "-I",
        "-c",
        code + "exec(sys.argv[3]);print(__import__('os').environ['CUDA_VISIBLE_DEVICES'])",
        json.dumps(verification_policy(policy)),
        "true",
        verification_code(),
    )
    result = subprocess.run(
        command, env={**os.environ, "CUDA_VISIBLE_DEVICES": grant}, capture_output=True, text=True
    )
    if expected:
        assert result.returncode and expected in result.stderr
    else:
        assert result.returncode == 0 and grant in result.stdout


def test_local_relative_storage_yields_absolute_binding(tmp_path, prepared, monkeypatch):
    _, _, _, spec, transport, service, _ = prepared
    monkeypatch.chdir(tmp_path)
    profile = ClusterProfile("local", workspace="relative", python=sys.executable)
    (binding,) = service.prepare((spec,), profile, transport)
    assert Path(binding["python"]).is_absolute()
    assert all(Path(v["path"]).is_absolute() for v in binding["entrypoints"].values())


def test_missing_import_is_not_silently_installed(tmp_path, prepared):
    _, project, profile, _, transport, service, _ = prepared
    spec = ToolRuntimeSpecification.from_mapping(
        "missing", declaration(imports=["nonexistent_isolated_tool_fixture"]), project
    )
    with pytest.raises(RuntimeError, match="lf clusters bootstrap"):
        service.prepare((spec,), profile, transport)
    registry = json.loads((Path(profile.storage.state_root) / "tool-runtimes.json").read_text())
    assert "missing" not in registry["runtimes"]


def test_unsafe_remote_job_binding_is_rejected_without_transport():
    profile = ClusterProfile(
        "remote", transport="ssh", host="example.org", workspace="/remote/project", python="python"
    )

    class NoTransport(LocalTransport):
        def run(self, *args, **kwargs):
            pytest.fail("Unsafe identity reached transport")

    with pytest.raises(ValueError, match="identity"):
        ToolRuntimeService.bind_job(profile, NoTransport(), "job-../../escape", ())


def test_native_work_records_tool_provenance_and_managed_checkpoint(
    tmp_path, prepared, monkeypatch
):
    from lambdaforge.work import WorkConfig
    from lambdaforge.work.runner import WorkRunner

    _, _, profile, _, transport, service, binding = prepared
    monkeypatch.setenv(
        "LAMBDAFORGE_TOOL_RUNTIMES_PATH",
        service.bind_job(profile, transport, "job-work-fixture", (binding,)),
    )
    config = WorkConfig.from_mapping(
        {"run": "tests.work.isolated_tool_cases.IsolatedProducer", "resources": {"cpu": 2}},
        source=tmp_path / "work.yaml",
    )
    result = WorkRunner().run(config)
    assert result.status == "succeeded"
    run = result.runs[0]
    assert run.primary_result == {"content": "FIXTURE"}
    environment = json.loads((run.run_dir / "environment.json").read_text())
    assert any(
        v.get("resolution_identity") == binding["resolution_id"]
        for v in environment["external_tools"]
    )
    assert any(p.read_text() == "FIXTURE" for p in tmp_path.rglob("produced.txt"))


def test_recovery_owner_retains_old_tool_snapshot(tmp_path):
    profile = ClusterProfile("local", workspace=str(tmp_path), python=sys.executable)
    roots = StorageOperations._roots(profile.storage.to_dict())
    previous = roots["job_workspaces"] / "job-original"
    active = roots["job_workspaces"] / "job-recovery"
    previous.mkdir(parents=True)
    active.mkdir()
    binding = {
        "environment_id": "env-tool-historical",
        "entrypoint_bundle_id": "tool-code-historical",
    }
    (previous / "state.json").write_text(json.dumps({"state": "failed"}))
    (previous / "request.json").write_text(json.dumps({"command": [sys.executable]}))
    (previous / "tool-runtimes.json").write_text(
        json.dumps({"tool_registry_version": 1, "runtimes": {"old": binding}})
    )
    (active / "state.json").write_text(json.dumps({"state": "running"}))
    (active / "request.json").write_text(
        json.dumps(
            {
                "command": [
                    sys.executable,
                    "--resume-execution",
                    str(previous / "work/execution-original"),
                ]
            }
        )
    )
    references = StorageOperations.environment_references(roots)
    assert "job-original" in references["protected_jobs"]
    assert binding["environment_id"] in references["environments"]
    assert binding["entrypoint_bundle_id"] in references["tool_bundles"]


@pytest.mark.parametrize("broken", ["{", '{"tool_registry_version":1,"runtimes":[]}'])
def test_corrupt_tool_registry_protects_only_tool_family(tmp_path, broken):
    profile = ClusterProfile("local", workspace=str(tmp_path), python=sys.executable)
    roots = StorageOperations._roots(profile.storage.to_dict())
    roots["state"].mkdir(parents=True)
    (roots["state"] / "tool-runtimes.json").write_text(broken)
    for name in ("env-tool-example", "env-main-example"):
        (roots["environments"] / name).mkdir(parents=True)
    (roots["bundles"] / "tool-code-example").mkdir(parents=True)
    references = StorageOperations.environment_references(roots)
    assert "env-tool-example" in references["environments"]
    assert "env-main-example" not in references["environments"]
    assert "tool-code-example" in references["tool_bundles"]
    assert any(v["reason"] == "missing-or-corrupt-tool-bindings" for v in references["issues"])


def test_verifier_does_not_import_pip_before_scientific_setuptools():
    # Regression for the real Triton -> setuptools path. Importing pip first reproduces
    # ensure_local_distutils's assertion on supported Python 3.10/3.11 installations.
    # No package install, CUDA allocation or application import is involved.
    policy = {
        "requirements": [f"setuptools=={importlib.metadata.version('setuptools')}"],
        "imports": ["setuptools"],
        "wheel_sources": [],
    }
    script = (
        verification_code() + "\nassert not any("
        "name=='pip' or name.startswith('pip.') for name in sys.modules)\n"
        "import distutils.core;assert '_distutils' in distutils.core.__file__\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, json.dumps(verification_policy(policy)), "false"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    validate_verification(policy, result.stdout)
    # Exercise the same bootstrap path, not only the generated script.
    ToolRuntimeService._verify(policy, LocalTransport(), sys.executable)


@pytest.mark.parametrize(
    "pin,actual,success",
    [
        ("1.0", "1.0.0", True),
        ("1.0", "1.1", False),
        ("2.6.0+cu124", "2.6.0+cpu", False),
        ("2.6.0+cu124", "2.6.0+cu124", True),
    ],
)
def test_host_version_validation_preserves_pep440_and_local_build_checks(pin, actual, success):
    policy = {"requirements": [f"example=={pin}"]}
    report = "upstream import output\n" + json.dumps({"versions": {"example": actual}})
    if success:
        validate_verification(policy, report)
    else:
        with pytest.raises(RuntimeError, match="Pinned tool package mismatch"):
            validate_verification(policy, report)


@pytest.mark.parametrize("report", ["", "not-json", "{}", '{"versions": []}'])
def test_corrupt_probe_report_never_waives_version_checks(report):
    with pytest.raises(RuntimeError, match="invalid metadata report"):
        validate_verification({"requirements": ["example==1.0"]}, report)


def test_work_preflight_keeps_version_validation_and_unverified_state(prepared, monkeypatch):
    _, _, profile, _, transport, service, binding = prepared
    monkeypatch.setenv(
        "LAMBDAFORGE_TOOL_RUNTIMES_PATH",
        service.bind_job(profile, transport, "job-bad-version", (binding,)),
    )
    runtime = ToolService(Log()).runtime("example")
    runtime.binding = {
        **runtime.binding,
        "policy": {**runtime.binding["policy"], "requirements": ["pip==0.0.0"]},
    }
    with pytest.raises(RuntimeError, match="Pinned tool package mismatch"):
        runtime.preflight()
    assert not runtime._verified
