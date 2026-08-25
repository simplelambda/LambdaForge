"""Native project dependencies remain declarative, exact and managed as one prefix."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.controlplane import (
    ClusterProfile,
    CommandResult,
    EnvironmentIdentity,
    ExecutionBundle,
    ManagedEnvironmentProvider,
    NativeEnvironmentError,
    NativeEnvironmentPlan,
    NativeEnvironmentPlanner,
    NativeEnvironmentSpecification,
    PythonRuntime,
    TorchInstallationPlan,
    Transport,
)
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.work.tools import ToolService


def _project(tmp_path: Path, environment: str, declaration: str | None = None) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='consumer'\nversion='1.0'\nrequires-python='>=3.10'\n\n"
        + (
            declaration
            or "[tool.lambdaforge.environment]\n"
            "manager='conda'\nfile='environment.yml'\n"
            "required_executables=['mmseqs', 'foldseek']\n"
        ),
        encoding="utf-8",
    )
    (root / "environment.yml").write_text(environment, encoding="utf-8")
    return root


def _runtime(architecture: str = "x86_64") -> PythonRuntime:
    return PythonRuntime(
        "runtime",
        "/remote/runtime/bin/python",
        "3.11.9",
        "CPython",
        "Linux",
        architecture,
        "micromamba",
        "2.8.1-0",
        True,
        True,
        "reuse",
    )


def _profile() -> ClusterProfile:
    return ClusterProfile(
        "gpu",
        transport="ssh",
        host="gpu.example",
        workspace="/remote/work",
        environment="managed",
    )


def _solver_inventory() -> list[dict[str, str]]:
    return [
        {
            "name": "python",
            "version": "3.11.9",
            "build_string": "h123",
            "channel": "conda-forge",
            "subdir": "linux-64",
        },
        {
            "name": "pip",
            "version": "25.1",
            "build_string": "pyh8",
            "channel": "conda-forge",
            "subdir": "noarch",
        },
        {
            "name": "mmseqs2",
            "version": "18.8cc5c",
            "build_string": "pl5321h6a68c12_0",
            "channel": "bioconda",
            "subdir": "linux-64",
        },
        {
            "name": "foldseek",
            "version": "10.941cd33",
            "build_string": "h9ee0642_0",
            "channel": "bioconda",
            "subdir": "linux-64",
        },
    ]


class PlanningRuntimeResolver:
    """Expose the pinned manager without performing an installer download in unit tests."""

    artifacts = SimpleNamespace(VERSION="2.8.1-0")

    def ensure_micromamba(
        self, profile: ClusterProfile, transport: Transport, platform: str
    ) -> tuple[str, str, str]:
        del profile, transport
        return "micromamba", f"/remote/micromamba-{platform}", "2.8.1-0"


class PlanningTransport(Transport):
    """Record one exact solver call and model content-addressed specification staging."""

    def __init__(self, inventory: Sequence[dict[str, str]] | None = None) -> None:
        self.inventory = list(inventory or _solver_inventory())
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.commands: list[tuple[str, ...]] = []
        self.solve_count = 0

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        values = tuple(command)
        self.commands.append(values)
        if values[:2] == ("test", "-f"):
            return CommandResult(0 if values[2] in self.files else 1)
        if values and values[0] == "mkdir":
            self.directories.add(values[-1])
            return CommandResult(0)
        if values and values[0] == "sha256sum":
            content = self.files.get(values[1])
            return (
                CommandResult(0, f"{hashlib.sha256(content).hexdigest()}  {values[1]}\n")
                if content is not None
                else CommandResult(1)
            )
        if values and values[0] == "mv":
            old, new = values[-2:]
            if old in self.files:
                self.files[new] = self.files.pop(old)
            for path, content in tuple(self.files.items()):
                if path.startswith(f"{old}/"):
                    self.files[f"{new}{path[len(old) :]}"] = content
                    del self.files[path]
            if old in self.directories:
                self.directories.remove(old)
                self.directories.add(new)
            return CommandResult(0)
        if values and values[0] == "rm":
            self.files.pop(values[-1], None)
            return CommandResult(0)
        if "--dry-run" in values and "--json" in values:
            self.solve_count += 1
            return CommandResult(0, json.dumps({"actions": {"LINK": self.inventory}}))
        if "Path(sys.argv[1]).write_text" in " ".join(values):
            self.files[values[-2]] = values[-1].encode()
            return CommandResult(0)
        return CommandResult(0)

    def put(self, source: str | Path, destination: str | Path) -> None:
        path = Path(source)
        if path.is_file():
            self.files[str(destination)] = path.read_bytes()
        elif path.is_dir():
            self.directories.add(str(destination))
            for item in path.rglob("*"):
                if item.is_file():
                    relative = item.relative_to(path).as_posix()
                    self.files[f"{destination}/{relative}"] = item.read_bytes()


def test_environment_file_is_strict_and_exposes_python_and_tools(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        """name: wisdom
channels: [conda-forge, bioconda]
dependencies: [python=3.11, pip, biopython>=1.84, foldseek, mmseqs2]
""",
    )

    specification = NativeEnvironmentSpecification.discover(root)

    assert specification is not None
    assert specification.python_requirement == "==3.11.*"
    assert specification.required_executables == ("mmseqs", "foldseek")
    assert specification.channels == ("conda-forge", "bioconda")
    assert specification.kind == "environment-file"


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            "channels: [conda-forge]\ndependencies:\n  - python=3.11\n  - pip:\n"
            "      - arbitrary-package\n",
            "nested pip sections",
        ),
        (
            "channels: [conda-forge]\ndependencies: [python=3.11, pytorch]\n",
            "PyTorch/CUDA plan",
        ),
        (
            "channels: [conda-forge]\ndependencies: [python=3.11]\nvariables: {X: Y}\n",
            "hooks, variables and prefixes",
        ),
        (
            "channels: [ftp://untrusted.example/channel]\ndependencies: [python=3.11]\n",
            "Conda channels",
        ),
    ],
)
def test_environment_file_rejects_unsafe_or_conflicting_features(
    tmp_path: Path, environment: str, message: str
) -> None:
    root = _project(tmp_path, environment)

    with pytest.raises((TypeError, ValueError), match=message):
        NativeEnvironmentSpecification.discover(root)


def test_explicit_lock_requires_matching_offline_package_bytes(tmp_path: Path) -> None:
    root = tmp_path / "locked"
    cache = root / "native-packages"
    cache.mkdir(parents=True)
    packages = {
        "python-3.11.9-h123.conda": b"python-package",
        "pip-25.1-pyh8.conda": b"pip-package",
        "mmseqs2-18.8cc5c-h456.conda": b"mmseqs-package",
    }
    for name, content in packages.items():
        (cache / name).write_bytes(content)
    entries = [
        f"https://conda.example/linux-64/{name}#sha256={hashlib.sha256(content).hexdigest()}"
        for name, content in packages.items()
    ]
    (root / "native.lock").write_text(
        "# platform: linux-64\n@EXPLICIT\n" + "\n".join(entries) + "\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='locked'\nversion='1'\nrequires-python='>=3.10'\n\n"
        "[tool.lambdaforge.environment]\nmanager='conda'\nlockfile='native.lock'\n"
        "package_cache='native-packages'\nrequired_executables=['mmseqs']\n",
        encoding="utf-8",
    )

    specification = NativeEnvironmentSpecification.discover(root)

    assert specification is not None and specification.kind == "explicit-lock"
    assert specification.lock_platform == "linux-64"
    assert specification.python_requirement == "==3.11.*"
    assert specification.package_cache_id

    offline_inventory = [
        {
            "name": "python",
            "version": "3.11.9",
            "build_string": "h123",
            "channel": "conda-forge",
            "subdir": "linux-64",
        },
        {
            "name": "pip",
            "version": "25.1",
            "build_string": "pyh8",
            "channel": "conda-forge",
            "subdir": "noarch",
        },
        {
            "name": "mmseqs2",
            "version": "18.8cc5c",
            "build_string": "h456",
            "channel": "bioconda",
            "subdir": "linux-64",
        },
    ]
    transport = PlanningTransport(offline_inventory)
    plan = NativeEnvironmentPlanner(
        tmp_path / "offline-plans",
        PlanningRuntimeResolver(),  # type: ignore[arg-type]
    ).plan(specification, _profile(), transport, _runtime())
    solve = next(command for command in transport.commands if "--dry-run" in command)
    assert plan.offline and plan.package_cache_id == specification.package_cache_id
    assert "--offline" in solve and "--file" in solve
    assert any("native-packages" in value for value in solve)

    (cache / "mmseqs2-18.8cc5c-h456.conda").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        NativeEnvironmentSpecification.discover(root)


def test_exact_native_plan_is_cached_and_participates_in_environment_identity(
    tmp_path: Path,
) -> None:
    specification = NativeEnvironmentSpecification.discover(
        _project(
            tmp_path,
            "channels: [conda-forge, bioconda]\n"
            "dependencies: [python=3.11, pip, mmseqs2, foldseek]\n",
        )
    )
    assert specification is not None
    transport = PlanningTransport()
    planner = NativeEnvironmentPlanner(
        tmp_path / "plans",
        PlanningRuntimeResolver(),  # type: ignore[arg-type]
    )

    first = planner.plan(specification, _profile(), transport, _runtime())
    second = planner.plan(specification, _profile(), transport, _runtime())

    assert first == second
    assert transport.solve_count == 1
    assert first.platform == "linux-64"
    assert {item["name"] for item in first.package_inventory} >= {"python", "mmseqs2"}
    wheel = ({"name": "consumer.whl", "sha256": "sha256:x", "size_bytes": 1},)
    native_identity = EnvironmentIdentity.create(
        wheel,
        python_requirement="==3.11.*",
        offline=False,
        dependency_policy={"native_environment": first.to_dict()},
    )
    plain_identity = EnvironmentIdentity.create(
        wheel,
        python_requirement="==3.11.*",
        offline=False,
        dependency_policy={},
    )
    assert native_identity.environment_id != plain_identity.environment_id


def test_native_plan_rejects_platform_python_and_torch_conflicts(tmp_path: Path) -> None:
    specification = NativeEnvironmentSpecification.discover(
        _project(
            tmp_path,
            "channels: [conda-forge]\ndependencies: [python=3.11, mmseqs2]\n",
        )
    )
    assert specification is not None
    planner = NativeEnvironmentPlanner(
        tmp_path / "plans",
        PlanningRuntimeResolver(),  # type: ignore[arg-type]
    )
    wrong_python = PlanningTransport(
        [{**_solver_inventory()[0], "version": "3.12.4"}, *_solver_inventory()[1:]]
    )
    with pytest.raises(ValueError, match="selected Python"):
        planner.plan(specification, _profile(), wrong_python, _runtime())

    torch_inventory = PlanningTransport(
        [
            *_solver_inventory(),
            {
                "name": "pytorch",
                "version": "2.8",
                "build": "gpu",
                "channel": "conda-forge",
                "subdir": "linux-64",
            },
        ]
    )
    with pytest.raises(ValueError, match="conflict"):
        NativeEnvironmentPlanner(
            tmp_path / "torch-plans",
            PlanningRuntimeResolver(),  # type: ignore[arg-type]
        ).plan(specification, _profile(), torch_inventory, _runtime())

    arm_transport = PlanningTransport(
        [
            {
                **item,
                "subdir": "noarch" if item["subdir"] == "noarch" else "linux-aarch64",
            }
            for item in _solver_inventory()
        ]
    )
    arm_plan = NativeEnvironmentPlanner(
        tmp_path / "arm-plans",
        PlanningRuntimeResolver(),  # type: ignore[arg-type]
    ).plan(specification, _profile(), arm_transport, _runtime("aarch64"))
    assert arm_plan.platform == "linux-aarch64"


class NativeInstallTransport(Transport):
    """Model concurrent immutable native prefix construction and executable validation."""

    def __init__(self, *, tools_available: bool = True) -> None:
        self.tools_available = tools_available
        self.list_inventory = [
            {key: value for key, value in item.items() if key != "subdir"}
            for item in _solver_inventory()
        ]
        self.metadata_inventory = [
            {
                "name": item["name"],
                "version": item["version"],
                "build": item["build_string"],
                "channel": item["channel"],
                "subdir": item["subdir"],
            }
            for item in _solver_inventory()
        ]
        self.post_pip_list_inventory: list[dict[str, str]] | None = None
        self.post_pip_metadata_inventory: list[dict[str, str]] | None = None
        self.pip_installed = False
        self.commands: list[tuple[str, ...]] = []
        self.files: dict[str, str] = {}
        self.directories: set[str] = set()
        self.create_count = 0
        self._lock = threading.RLock()

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, timeout
        values = tuple(command)
        with self._lock:
            self.commands.append(values)
            if values[:2] == ("test", "-f"):
                return CommandResult(0 if values[2] in self.files else 1)
            if values and values[0] == "mkdir":
                target = values[-1]
                if "-p" not in values and target in self.directories:
                    return CommandResult(1)
                self.directories.add(target)
                return CommandResult(0)
            if values and values[0] == "rmdir":
                self.directories.discard(values[-1])
                return CommandResult(0)
            if values and values[0] == "mv":
                source, destination = values[-2:]
                for path, content in tuple(self.files.items()):
                    if path == source or path.startswith(f"{source}/"):
                        self.files[f"{destination}{path[len(source) :]}"] = content
                        del self.files[path]
                self.directories.add(destination)
                return CommandResult(0)
            if values and values[0] == "rm":
                target = values[-1]
                self.directories = {
                    item
                    for item in self.directories
                    if item != target and not item.startswith(f"{target}/")
                }
                for path in tuple(self.files):
                    if path == target or path.startswith(f"{target}/"):
                        del self.files[path]
                return CommandResult(0)
            if "create" in values and any("micromamba-" in item for item in values):
                self.create_count += 1
                self.directories.add(values[values.index("--prefix") + 1])
                time.sleep(0.05)
                return CommandResult(0)
            if "list" in values and any("micromamba-" in item for item in values):
                inventory = (
                    self.post_pip_list_inventory
                    if self.pip_installed and self.post_pip_list_inventory is not None
                    else self.list_inventory
                )
                return CommandResult(0, json.dumps(inventory))
            source = " ".join(values)
            if "lambdaforge-conda-meta-inventory" in source:
                inventory = (
                    self.post_pip_metadata_inventory
                    if self.pip_installed and self.post_pip_metadata_inventory is not None
                    else self.metadata_inventory
                )
                return CommandResult(0, json.dumps(inventory))
            if "conda-meta" in source:
                if not self.tools_available:
                    return CommandResult(1, stderr="missing executable: mmseqs")
                return CommandResult(
                    0,
                    json.dumps(
                        [
                            {
                                "name": "mmseqs",
                                "path": "/remote/env/bin/mmseqs",
                                "version": "18.8cc5c",
                                "package": "mmseqs2",
                                "package_version": "18.8cc5c",
                                "build": "pl5321h6a68c12_0",
                                "channel": "bioconda",
                                "subdir": "linux-64",
                            }
                        ]
                    ),
                )
            if "Path(sys.argv[1]).write_text" in source:
                self.files[values[-2]] = values[-1]
                return CommandResult(0)
            if "os.rename" in source:
                old, new = values[-2:]
                for path, content in tuple(self.files.items()):
                    if path == old or path.startswith(f"{old}/"):
                        self.files[f"{new}{path[len(old) :]}"] = content
                        del self.files[path]
                self.directories.add(new)
                return CommandResult(0)
            if "import lambdaforge" in source:
                return CommandResult(0, "0.12.0 2.1.0 None False\n")
            if "pip" in values:
                if "install" in values:
                    self.pip_installed = True
                return CommandResult(0)
            return CommandResult(0)

    def put(self, source: str | Path, destination: str | Path) -> None:
        del source, destination


def _change_installed_package(
    transport: NativeInstallTransport,
    field: str,
    value: str,
    *,
    after_pip: bool = False,
) -> None:
    """Change the same installed MMseqs record in list and conda-meta fixtures."""
    listed = [dict(item) for item in transport.list_inventory]
    metadata = [dict(item) for item in transport.metadata_inventory]
    list_field = "build_string" if field == "build" else field
    for item in listed:
        if item["name"] == "mmseqs2" and field != "subdir":
            item[list_field] = value
    for item in metadata:
        if item["name"] == "mmseqs2":
            item[field] = value
    if after_pip:
        transport.post_pip_list_inventory = listed
        transport.post_pip_metadata_inventory = metadata
    else:
        transport.list_inventory = listed
        transport.metadata_inventory = metadata


def _native_plan() -> NativeEnvironmentPlan:
    inventory = tuple(
        {
            "name": item["name"],
            "version": item["version"],
            "build": item["build_string"],
            "channel": item["channel"],
            "subdir": item["subdir"],
        }
        for item in _solver_inventory()
    )
    return NativeEnvironmentPlan(
        f"sha256:{'a' * 64}",
        "environment.yml",
        "environment-file",
        "linux-64",
        "2.8.1-0",
        ("conda-forge", "bioconda"),
        ("python=3.11", "pip", "mmseqs2", "foldseek"),
        tuple(
            f"{item['channel']}::{item['name']}={item['version']}={item['build']}"
            for item in inventory
        ),
        inventory,
        ("mmseqs",),
        False,
    )


def _native_bundle(tmp_path: Path) -> ExecutionBundle:
    plan = _native_plan()
    torch = TorchInstallationPlan(
        "cpu",
        "2.1.0",
        "https://download.pytorch.org/whl/cpu",
        "cpu",
        python_version="3.11",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return ExecutionBundle(
        "bundle",
        tmp_path,
        manifest,
        manifest,
        2,
        environment_id="env-native",
        package_names=("lambdaforge.whl", "consumer.whl"),
        environment_policy={
            "pytorch": torch.to_dict(),
            "native_environment": plan.to_dict(),
        },
    )


def test_native_inventory_recovers_linux_and_noarch_subdirs_from_conda_meta() -> None:
    transport = NativeInstallTransport()

    result = ManagedEnvironmentProvider._verify_native_inventory(
        _profile(),
        transport,
        PurePosixPath("/remote/environment"),
        _native_plan(),
        None,
    )

    assert result.returncode == 0
    assert all("subdir" not in item for item in transport.list_inventory)
    assert {item["subdir"] for item in transport.metadata_inventory} == {"linux-64", "noarch"}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("version", "999"),
        ("build", "different_0"),
        ("channel", "untrusted"),
        ("subdir", "noarch"),
    ),
)
def test_native_inventory_rejects_real_package_differences_compactly(
    field: str, value: str
) -> None:
    transport = NativeInstallTransport()
    _change_installed_package(transport, field, value)

    result = ManagedEnvironmentProvider._verify_native_inventory(
        _profile(),
        transport,
        PurePosixPath("/remote/environment"),
        _native_plan(),
        None,
    )

    assert result.returncode == 1
    assert "1 package difference(s)" in result.stderr
    assert f"changed 'mmseqs2': {field}" in result.stderr
    assert "expected [" not in result.stderr and "observed [" not in result.stderr


def test_native_inventory_reports_missing_essential_conda_metadata_compactly() -> None:
    transport = NativeInstallTransport()
    transport.metadata_inventory[0].pop("subdir")

    result = ManagedEnvironmentProvider._verify_native_inventory(
        _profile(),
        transport,
        PurePosixPath("/remote/environment"),
        _native_plan(),
        None,
    )

    assert result.returncode == 1
    assert result.stderr == (
        "Invalid native package inventory: conda-meta lacks subdir for package 'python'"
    )


def test_native_environment_is_one_prefix_reused_concurrently_and_records_tools(
    tmp_path: Path,
) -> None:
    transport = NativeInstallTransport()
    provider = ManagedEnvironmentProvider()
    bundle = _native_bundle(tmp_path)
    profile = _profile()

    results: list[Any] = []

    def prepare() -> None:
        results.append(
            provider.prepare(profile, transport, bundle, remote_bundle_dir="/remote/bundle")
        )

    threads = [threading.Thread(target=prepare) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert transport.create_count == 1
    assert {result.reused for result in results} == {False, True}
    assert not any("venv" in command for command in transport.commands)
    create = next(command for command in transport.commands if "create" in command)
    assert "--platform" in create and "linux-64" in create
    created_prefix = create[create.index("--prefix") + 1]
    pip = [command for command in transport.commands if "pip" in command]
    assert pip and all(f"{created_prefix}/bin/python" in " ".join(command) for command in pip)
    marker = next(
        json.loads(value)
        for key, value in transport.files.items()
        if key.endswith("env-native/.lambdaforge-environment.json")
    )
    assert marker["native_tools"][0]["package"] == "mmseqs2"
    metadata_reads = [
        command
        for command in transport.commands
        if "lambdaforge-conda-meta-inventory" in " ".join(command)
    ]
    assert len(metadata_reads) == 3  # before pip, after pip and idempotent cache reuse


def test_post_pip_inventory_change_fails_and_cleans_only_temporary_prefix(
    tmp_path: Path,
) -> None:
    transport = NativeInstallTransport()
    _change_installed_package(transport, "build", "pip_changed_0", after_pip=True)
    profile = _profile()

    with pytest.raises(
        NativeEnvironmentError,
        match="Pip installation changed.*changed 'mmseqs2': build",
    ):
        ManagedEnvironmentProvider().prepare(
            profile,
            transport,
            _native_bundle(tmp_path),
            remote_bundle_dir="/remote/bundle",
        )

    cleanup = [command for command in transport.commands if command[:2] == ("rm", "-rf")]
    assert profile.storage is not None
    assert cleanup and cleanup[-1][-1].startswith(f"{profile.storage.environment_root}/.env-")
    assert ".tmp-" in cleanup[-1][-1]
    assert not any(
        path.endswith("env-native/.lambdaforge-environment.json") for path in transport.files
    )


def test_missing_required_native_executable_prevents_publication(tmp_path: Path) -> None:
    transport = NativeInstallTransport(tools_available=False)

    with pytest.raises(RuntimeError, match="verification failed.*missing executable"):
        ManagedEnvironmentProvider().prepare(
            _profile(),
            transport,
            _native_bundle(tmp_path),
            remote_bundle_dir="/remote/bundle",
        )

    assert not any(
        key.endswith("env-native/.lambdaforge-environment.json") for key in transport.files
    )


def test_tool_service_naturally_searches_the_active_python_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "environment"
    executable = prefix / "bin" / "native-tool"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\necho native-tool 1.0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("lambdaforge.work.tools.sys.executable", str(prefix / "bin" / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    messages: list[str] = []
    service = ToolService(SimpleNamespace(emit=lambda message, **kwargs: messages.append(message)))

    tool = service.require("native-tool", version_args=("--version",))

    assert tool.path == executable.resolve()
    assert tool.version == "native-tool 1.0"


def test_native_package_cache_is_reconstructible_gc_storage(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    native = cache / "native-packages" / "cache-identity"
    native.mkdir(parents=True)
    (native / ".lambdaforge-native-cache.json").write_text("{}", encoding="utf-8")
    (native / "packages.conda").write_bytes(b"cached")
    state = tmp_path / "state"
    jobs = tmp_path / "jobs"
    state.mkdir()
    jobs.mkdir()
    descriptor = {
        "state_root": str(state),
        "cache_root": str(cache),
        "run_root": str(jobs),
        "cache_max_age": 0,
    }

    preview = StorageOperations.gc(descriptor, {}, apply=False)

    selected = [item for item in preview["candidates"] if item["category"] == "native_packages"]
    assert selected[0]["name"] == "cache-identity"
    assert native.is_dir()
