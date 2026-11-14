"""Real loader/import regressions; no WISDOM, network, package install or GPU allocation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.controlplane.ClusterBootstrapResult import ClusterBootstrapResult
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.jobs import JobState
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.NativeLibraryPolicy import (
    POLICY_VARIABLE,
    NativeLibraryPolicy,
)
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work.tool_runtime import IsolatedToolRuntime
from lambdaforge.work.tools import ToolService


@pytest.fixture
def abi_fixture(tmp_path: Path) -> tuple[NativeLibraryPolicy, Path, Path]:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("Native loader regression requires a C compiler")
    old = tmp_path / "host/lib"
    selected = tmp_path / "dependencies"
    new = selected / "lib"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    source = tmp_path / "runtime.c"
    source.write_text("int baseline(void){return 7;} int modern(void){return 15;}\n")
    for directory, modern in ((old, False), (new, True)):
        script = directory / "versions.map"
        script.write_text(
            "CXXABI_1.3.13 { global: baseline; };\n"
            + ("CXXABI_1.3.15 { global: modern; } CXXABI_1.3.13;\n" if modern else "")
        )
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(directory / "libstdc++.so.6"),
                "-Wl,-soname,libstdc++.so.6",
                f"-Wl,--version-script={script}",
            ],
            check=True,
            capture_output=True,
        )
    bindings = []
    for directory, function in ((old, "baseline"), (new, "modern")):
        binding = tmp_path / f"{function}.so"
        wrapper = tmp_path / f"{function}.c"
        wrapper.write_text(
            f"extern int {function}(void); int value(void){{return {function}();}}\n"
        )
        subprocess.run(
            [
                compiler,
                "-shared",
                "-fPIC",
                str(wrapper),
                "-o",
                str(binding),
                f"-L{directory}",
                "-l:libstdc++.so.6",
                f"-Wl,--enable-new-dtags,-rpath,{directory}",
            ],
            check=True,
            capture_output=True,
        )
        bindings.append(binding)
    return NativeLibraryPolicy(str(selected)), bindings[0], bindings[1]


def _child(command: tuple[str, ...], *, extra: dict[str, str] | None = None):
    environment = dict(os.environ)
    environment.pop(POLICY_VARIABLE, None)
    environment.pop("LD_PRELOAD", None)
    environment.pop("LD_LIBRARY_PATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = ""  # Every real Torch test remains CPU-only.
    environment.update(extra or {})
    return subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("reverse", [False, True])
def test_real_cxxabi_order_conflict_is_resolved_before_startup(abi_fixture, reverse):
    policy, old, new = abi_fixture
    first, second = (new, old) if reverse else (old, new)
    code = (
        "import ctypes,json,os\n"
        f"one=ctypes.CDLL({str(first)!r});two=ctypes.CDLL({str(second)!r})\n"
        "from lambdaforge.controlplane.NativeLibraryPolicy import native_library_report\n"
        "print(json.dumps(native_library_report(verify=True)))\n"
    )
    baseline = _child((sys.executable, "-c", code))
    if not reverse:
        assert baseline.returncode != 0
        assert "CXXABI_1.3.15" in baseline.stderr
    else:
        assert baseline.returncode == 0, baseline.stderr
    safe = _child(policy.wrap((sys.executable, "-c", code), python=sys.executable))
    assert safe.returncode == 0, safe.stderr
    report = json.loads(safe.stdout)
    assert report["loaded"]["libstdc++"] == [
        str(Path(policy.dependency_prefix) / "lib/libstdc++.so.6")
    ]
    assert report["validated_libraries"] == report["loaded"]["libstdc++"]


def test_conflicting_preload_is_detected_not_misreported_as_verified(abi_fixture):
    policy, old, _ = abi_fixture
    code = (
        "from lambdaforge.controlplane.NativeLibraryPolicy import native_library_report\n"
        "native_library_report(verify=True)\n"
    )
    result = _child(
        policy.wrap((sys.executable, "-c", code), python=sys.executable),
        extra={"LD_PRELOAD": str(old.parent / "host/lib/libstdc++.so.6")},
    )
    assert result.returncode != 0
    assert "Native library origin mismatch" in result.stderr


def test_policy_is_scoped_inherited_and_preserves_provider_gpu_grants(tmp_path: Path):
    prefix = tmp_path / "dependencies"
    (prefix / "lib").mkdir(parents=True)
    policy = NativeLibraryPolicy(str(prefix))
    parent_before = dict(os.environ)
    code = (
        "import json,os,subprocess,sys\n"
        "keys=('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES',"
        "'LAMBDAFORGE_NATIVE_LIBRARY_POLICY')\n"
        "snapshot={k:os.environ.get(k) for k in keys}\n"
        "nested='import json,os;print(json.dumps({k:os.environ.get(k) for k in '+repr(keys)+'}))'\n"
        "child=json.loads(subprocess.check_output([sys.executable,'-c',nested],text=True))\n"
        "assert child==snapshot\nprint(json.dumps(snapshot))\n"
    )
    result = _child(
        policy.wrap((sys.executable, "-c", code), python=sys.executable),
        extra={
            "CUDA_VISIBLE_DEVICES": "GPU-provider-owned,3",
            "NVIDIA_VISIBLE_DEVICES": "provider-grant",
            "LD_LIBRARY_PATH": ":.:relative:/provider/driver/lib:/provider/driver/lib:",
        },
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(result.stdout)
    assert snapshot["LD_LIBRARY_PATH"] == f"{prefix}/lib:/provider/driver/lib"
    assert snapshot["CUDA_VISIBLE_DEVICES"] == "GPU-provider-owned,3"
    assert snapshot["NVIDIA_VISIBLE_DEVICES"] == "provider-grant"
    isolated = NativeLibraryPolicy.isolated_environment(snapshot)
    assert isolated["LD_LIBRARY_PATH"] == "/provider/driver/lib"
    assert POLICY_VARIABLE not in isolated
    assert isolated["CUDA_VISIBLE_DEVICES"] == snapshot["CUDA_VISIBLE_DEVICES"]
    assert dict(os.environ) == parent_before


def test_directory_escape_is_rejected_without_mutating_parent(tmp_path: Path):
    prefix = tmp_path / "dependencies"
    prefix.mkdir()
    outside = tmp_path / "foreign"
    outside.mkdir()
    (prefix / "lib").symlink_to(outside)
    result = _child(
        NativeLibraryPolicy(str(prefix)).wrap((sys.executable, "-c", "pass"), python=sys.executable)
    )
    assert result.returncode != 0
    assert "Unsafe/missing selected native library directory" in result.stderr


def test_bootstrap_does_not_import_sitecustomize_before_loader_selection(tmp_path: Path):
    prefix = tmp_path / "dependencies"
    (prefix / "lib").mkdir(parents=True)
    hook = tmp_path / "hooks"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os\n"
        "if not os.environ.get('LAMBDAFORGE_NATIVE_LIBRARY_POLICY'): os._exit(91)\n"
        "os.environ['FIXTURE_HOOK_RAN']='yes'\n"
    )
    policy = NativeLibraryPolicy(str(prefix))
    command = policy.wrap(
        (sys.executable, "-c", "import os;assert os.environ['FIXTURE_HOOK_RAN']=='yes'"),
        python=sys.executable,
    )
    result = _child(command, extra={"PYTHONPATH": str(hook)})
    assert result.returncode == 0, result.stderr


def test_native_envelope_reaches_the_supervisor_and_module_entrypoint(tmp_path: Path):
    root = tmp_path / "dependencies"
    (root / "lib").mkdir(parents=True)
    policy = NativeLibraryPolicy(str(root))
    profile = ClusterProfile("fixture", python=sys.executable, workspace=str(tmp_path))
    scheduler = ProcessScheduler(LocalTransport(), profile)
    command = (
        "gpu",
        "exec",
        *policy.wrap(
            (sys.executable, "-m", "lambdaforge", "run", "fixture.yaml"), python=sys.executable
        ),
    )
    submission = scheduler.submit(
        command, ResourceRequest(), work_dir=tmp_path, dry_run=True, job_id="job-native-fixture"
    )
    assert NativeLibraryPolicy.from_command(submission.command) == policy
    assert scheduler._control_python(command) == sys.executable
    assert "lambdaforge.controlplane.ProcessSupervisor" in submission.command
    assert NativeLibraryPolicy.from_command((sys.executable, "-m", "lambdaforge")) is None


@pytest.mark.parametrize("key", ["LD_LIBRARY_PATH", "LD_PRELOAD"])
def test_tools_cannot_override_active_native_policy_and_isolated_tools_detach(
    tmp_path: Path, monkeypatch, key
):
    policy = NativeLibraryPolicy(str(tmp_path / "dependencies"))
    monkeypatch.setenv(POLICY_VARIABLE, json.dumps(policy.to_dict()))
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{policy.dependency_prefix}/lib:/provider/driver/lib")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "provider-grant")
    owner = ToolService(SimpleNamespace(emit=lambda *args, **kwargs: None))
    with pytest.raises(ValueError, match="framework-owned"):
        owner.run((sys.executable, "-c", "pass"), env={key: "/other/lib"})
    detached = IsolatedToolRuntime({"name": "fixture", "python": sys.executable}, owner)
    isolated = detached._runner._environment
    assert isolated is not None
    assert isolated["LD_LIBRARY_PATH"] == "/provider/driver/lib"
    assert POLICY_VARIABLE not in isolated
    assert isolated["CUDA_VISIBLE_DEVICES"] == "provider-grant"


@pytest.fixture
def real_cpp_prefix(tmp_path: Path):
    # Discover the installed CPU-capable Torch/SciPy runtime without installing anything.
    probe = _child(
        (
            sys.executable,
            "-c",
            "import torch,scipy.spatial,json;"
            "from lambdaforge.controlplane.NativeLibraryPolicy import native_library_report;"
            "print(json.dumps(native_library_report()))",
        )
    )
    assert probe.returncode == 0, probe.stderr
    loaded = json.loads(probe.stdout)["loaded"]
    assert loaded.get("libstdc++"), "Real C++ runtime must be visible for this regression"
    selected = tmp_path / "dependencies"
    (selected / "lib").mkdir(parents=True)
    for family, soname in (("libstdc++", "libstdc++.so.6"), ("libgcc_s", "libgcc_s.so.1")):
        if loaded.get(family):
            shutil.copyfile(loaded[family][0], selected / "lib" / soname)
    return NativeLibraryPolicy(str(selected))


@pytest.mark.parametrize("reverse", [False, True])
def test_real_torch_scipy_and_loaded_origins_in_either_order(real_cpp_prefix, reverse):
    policy = real_cpp_prefix
    result = _child(
        policy.wrap(
            (sys.executable, "-c", policy.probe_code(scipy_first=reverse)), python=sys.executable
        )
    )
    assert result.returncode == 0, result.stderr
    report = policy.read_report(result.stdout)
    assert report["loaded"]["libstdc++"] == [f"{policy.dependency_prefix}/lib/libstdc++.so.6"]
    assert report["validated_libraries"]


def test_provider_preparation_checks_both_orders_and_runtime_provenance(
    real_cpp_prefix, monkeypatch
):
    policy = real_cpp_prefix
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    evidence = {}
    checked = ManagedEnvironmentProvider._verify(
        ClusterProfile("fixture", python=sys.executable),
        LocalTransport(),
        sys.executable,
        None,
        None,
        library_policy=policy,
        library_evidence=evidence,
    )
    assert checked.returncode == 0, checked.stderr
    assert set(evidence) == {"torch_first", "scipy_first"}
    bootstrap = ClusterBootstrapResult(
        "fixture",
        "env-fixture",
        sys.executable,
        True,
        native_library_policy=policy.to_dict(),
        native_libraries=evidence,
    )
    serialized = json.loads(json.dumps(bootstrap.to_dict()))
    assert (
        serialized["native_libraries"]["torch_first"]["loaded"] == evidence["torch_first"]["loaded"]
    )
    code = (
        "import json,scipy.spatial\n"
        "from lambdaforge.EnvironmentManifest import EnvironmentManifest\n"
        "print(json.dumps(EnvironmentManifest.capture().native_libraries))\n"
    )
    observed = _child(policy.wrap((sys.executable, "-c", code), python=sys.executable))
    assert observed.returncode == 0, observed.stderr
    provenance = json.loads(observed.stdout)
    assert provenance["policy"]["dependency_prefix"] == policy.dependency_prefix
    assert provenance["loaded"]["libstdc++"] == evidence["torch_first"]["loaded"]["libstdc++"]


def test_detached_supervisor_and_child_load_selected_cpp(
    real_cpp_prefix, tmp_path: Path, monkeypatch
):
    policy = real_cpp_prefix
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    profile = ClusterProfile(
        "fixture",
        python=sys.executable,
        environment="existing",
        workspace=str(tmp_path),
        storage=ClusterStoragePolicy(
            str(tmp_path / "state"), str(tmp_path / "cache"), str(tmp_path / "jobs")
        ),
    )
    scheduler = ProcessScheduler(LocalTransport(), profile)
    code = (
        "import torch,scipy.spatial,json,time\n"
        "from lambdaforge.controlplane.NativeLibraryPolicy import native_library_report\n"
        "print('CHILD_NATIVE='+json.dumps(native_library_report(verify=True)),flush=True)\n"
        "time.sleep(1)\n"
    )
    command = policy.wrap((sys.executable, "-c", code), python=sys.executable)
    job_id = "job-disposable-native-libraries"
    scheduler.submit(command, ResourceRequest(), work_dir=tmp_path, job_id=job_id)
    try:
        state_path = tmp_path / "jobs" / job_id / "state.json"
        state = json.loads(state_path.read_text())
        pid = state["supervisor_identity"]["pid"]
        assert (
            str(Path(policy.dependency_prefix) / "lib/libstdc++.so.6")
            in Path(f"/proc/{pid}/maps").read_text()
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state_value = scheduler.state(job_id)
            if state_value in {JobState.SUCCEEDED, JobState.FAILED}:
                break
            time.sleep(0.1)
        assert scheduler.state(job_id) is JobState.SUCCEEDED, scheduler.logs(job_id)
        reports = [
            line.split("=", 1)[1]
            for line in scheduler.logs(job_id).splitlines()
            if line.startswith("CHILD_NATIVE=")
        ]
        assert len(reports) == 1
        assert json.loads(reports[0])["loaded"]["libstdc++"] == [
            f"{policy.dependency_prefix}/lib/libstdc++.so.6"
        ]
    finally:
        if not scheduler.state(job_id).terminal:
            scheduler.cancel(job_id)


@pytest.mark.parametrize(
    "value", ["/", "relative", "/tmp/../other", "/tmp/lib\nwrong", "/tmp/lib:other"]
)
def test_invalid_selected_prefix_is_not_interpreted_locally_or_as_shell(value):
    with pytest.raises(ValueError, match="absolute, safe"):
        NativeLibraryPolicy(value)
