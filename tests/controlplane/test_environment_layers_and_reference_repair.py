"""Native immutable code layers and reference GC use disposable, tiny CPU fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import venv
import zipfile
from pathlib import Path

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.DependencyLayer import DependencyLayer
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.PreparedEnvironment import PreparedEnvironment
from lambdaforge.controlplane.StorageOperations import StorageOperations
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion
from tests.controlplane.test_storage_management import descriptor, environment


def wheel(root: Path, value: int = 1, requirements: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "disposable_fixture-1.0-py3-none-any.whl"
    metadata = "disposable_fixture-1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        entries = {
            "disposable_fixture/__init__.py": f"VALUE = {value}\ndef main():print(VALUE)\n",
            f"{metadata}/METADATA": "Metadata-Version: 2.1\n"
            "Name: disposable-fixture\nVersion: 1.0\n" + requirements,
            f"{metadata}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{metadata}/entry_points.txt": (
                "[console_scripts]\nfixture-code = disposable_fixture:main\n"
            ),
        }
        for name, content in entries.items():
            archive.writestr(name, content)
        archive.writestr(
            f"{metadata}/RECORD",
            "".join(f"{name},,\n" for name in entries) + f"{metadata}/RECORD,,\n",
        )
    return path


def test_dependency_identity_ignores_code_but_not_contract_or_runtime(tmp_path):
    first, second = wheel(tmp_path / "one", 1), wheel(tmp_path / "two", 2)
    policy = {
        "python_runtime": {
            "version": "3.11.16",
            "architecture": "x86_64",
            "action": "create",
            "ready": False,
        }
    }
    identity = DependencyLayer.policy([first], policy)
    assert identity == DependencyLayer.policy([second], policy)
    reused = {"python_runtime": {**policy["python_runtime"], "action": "reuse", "ready": True}}
    assert identity == DependencyLayer.policy([second], reused)
    gpu_policy = {
        "pytorch": {
            "version": "2.9+cu128",
            "channel": "cu128",
            "driver_version": "575",
            "compute_capabilities": ["9.0"],
            "require_cuda": True,
        }
    }
    gpu_other = {
        "pytorch": {
            **gpu_policy["pytorch"],
            "driver_version": "580",
            "compute_capabilities": ["8.0"],
            "require_cuda": False,
        }
    }
    assert DependencyLayer.policy([first], gpu_policy) == DependencyLayer.policy([first], gpu_other)
    gpu_changed = {"pytorch": {**gpu_other["pytorch"], "channel": "cu121", "version": "2.9+cu121"}}
    assert DependencyLayer.policy([first], gpu_policy) != DependencyLayer.policy(
        [first], gpu_changed
    )
    for replacement in (
        {"python_runtime": {"version": "3.12.1"}},
        {"pytorch": {"version": "2.9+cu128"}},
        {"native_environment": {"exact_packages": ["python=3.11=newabi"]}},
    ):
        assert identity != DependencyLayer.policy([second], replacement)
    changed = wheel(tmp_path / "three", 3, "Requires-Dist: numpy>=2\n")
    assert identity != DependencyLayer.policy([changed], policy)
    for selected in (first, second):
        env = EnvironmentIdentity.create(
            [
                {
                    "name": selected.name,
                    "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
                    "size_bytes": selected.stat().st_size,
                }
            ],
            python_requirement=">=3.10",
            offline=False,
        )
        if selected == first:
            initial = env.environment_id
        else:
            assert initial != env.environment_id  # exact code identity is NOT lost.


class TinyBaseProvider(ManagedEnvironmentProvider):
    """Replace only heavy dependency resolution, not the actual code-layer installer."""

    builds = 0

    def _prepare(self, profile, transport, bundle, **kwargs):
        assert not kwargs["activate"]
        self.last_pins = kwargs.get("dependency_pins", ())
        base = Path(profile.storage.environment_root) / bundle.environment_id
        if not base.exists():
            self.builds += 1
            venv.EnvBuilder(with_pip=True, symlinks=True).create(base)
            pure = subprocess.check_output(
                [
                    str(base / "bin/python"),
                    "-c",
                    'import sysconfig;print(sysconfig.get_path("purelib"))',
                ],
                text=True,
            ).strip()
            site = Path(pure)
            (site / "lambdaforge").mkdir()
            (site / "lambdaforge/__init__.py").write_text(
                f"__version__ = {LambdaForgeVersion.CURRENT!r}\n"
            )
            (site / "torch").mkdir()
            (site / "torch/__init__.py").write_text(
                "from types import SimpleNamespace\nversion=SimpleNamespace(cuda=None)\n"
                "__version__='fixture'\nclass cuda:\n @staticmethod\n"
                " def is_available():return False\n"
            )
            inventory = json.loads(
                subprocess.check_output(
                    [str(base / "bin/python"), "-m", "pip", "list", "--format=json"], text=True
                )
            )
            (base / ".lambdaforge-environment.json").write_text(
                json.dumps({"installed_packages": inventory})
            )
            command = base / "bin/dependency-command"
            command.write_text(
                f"#!{base}/bin/python\nfrom disposable_fixture import VALUE\nprint(VALUE)\n"
            )
            command.chmod(0o755)
        return PreparedEnvironment(bundle.environment_id, str(base / "bin/python"), self.builds > 1)


def test_real_code_layers_reuse_dependencies_and_do_not_mutate_base(tmp_path):
    config = descriptor(tmp_path)
    Path(config["state_root"]).mkdir()
    profile = ClusterProfile(
        "fixture",
        transport="local",
        environment="managed",
        python=sys.executable,
        workspace=str(tmp_path),
        storage=ClusterStoragePolicy.from_mapping(config, workspace=str(tmp_path)),
    )
    provider = TinyBaseProvider()
    paths = [wheel(tmp_path / "one/packages", 1), wheel(tmp_path / "two/packages", 2)]
    base_before = None
    for index, selected in enumerate(paths, 1):
        layer = DependencyLayer.policy([selected], {})
        bundle = ExecutionBundle(
            f"bundle-{index}",
            selected.parent.parent,
            selected,
            selected,
            1,
            environment_id=f"env-code-{index}",
            package_names=(selected.name,),
            environment_policy={"dependency_layer": layer},
        )
        prepared = provider.prepare(
            profile, LocalTransport(), bundle, remote_bundle_dir=selected.parent.parent
        )
        observed = subprocess.check_output(
            [prepared.python, "-c", "import disposable_fixture;print(disposable_fixture.VALUE)"],
            text=True,
        ).strip()
        assert observed == str(index)
        assert not (Path(prepared.python).parent / "fixture-code").is_symlink()
        script = (Path(prepared.python).parent / "fixture-code").read_text()
        assert str(Path(prepared.python).parent.parent) in script
        assert ".tmp-" not in "\n".join(script.splitlines()[:4])
        assert subprocess.check_output(
            [str(Path(prepared.python).parent / "fixture-code")], text=True
        ).strip() == str(index)
        assert subprocess.check_output(
            [str(Path(prepared.python).parent / "dependency-command")], text=True
        ).strip() == str(index)
        base = Path(profile.storage.environment_root) / layer["identity"]
        snapshot = {
            str(p.relative_to(base)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in base.rglob("*")
            if p.is_file() and not p.is_symlink() and "__pycache__" not in str(p)
        }
        if base_before is None:
            base_before = snapshot
        else:
            assert base_before == snapshot
        assert provider.builds == 1
        again = provider.prepare(
            profile, LocalTransport(), bundle, remote_bundle_dir=selected.parent.parent
        )
        assert again.reused
    # Live layers retain the shared base, while obsolete code layers are collectable.
    preview = StorageOperations.gc(config, {})
    assert layer["identity"] not in {item["name"] for item in preview["candidates"]}
    receipt = Path(config["state_root"]) / "dependency-resolutions" / f"{layer['identity']}.json"
    frozen = json.loads(receipt.read_text())
    (Path(config["state_root"]) / "active-environment").unlink()
    StorageOperations.gc({**config, "environments_only": True}, {}, apply=True)
    assert receipt.is_file() and not base.exists()
    rebuilt = provider.prepare(
        profile, LocalTransport(), bundle, remote_bundle_dir=selected.parent.parent
    )
    assert provider.builds == 2 and not rebuilt.reused
    assert rebuilt.dependency_resolution_id == frozen["resolution_id"]
    assert set(provider.last_pins) == {
        f"{item['name']}=={item['version']}" for item in frozen["packages"]
    }
    assert json.loads(receipt.read_text()) == frozen
    receipt.write_text('{"version":1,"identity":"corrupt"}')
    with pytest.raises(RuntimeError, match="frozen dependency resolution"):
        provider.prepare(
            profile, LocalTransport(), bundle, remote_bundle_dir=selected.parent.parent
        )


@pytest.mark.parametrize("state", [None, "", "{bad-json", "{}"])
def test_missing_state_with_known_request_only_protects_its_own_environment(tmp_path, state):
    environment(tmp_path, "env-in-use")
    environment(tmp_path, "env-obsolete")
    job = tmp_path / "jobs/job-orphan"
    job.mkdir(parents=True)
    if state is not None:
        (job / "state.json").write_text(state)
    (job / "request.json").write_text(
        json.dumps({"command": [str(tmp_path / "cache/environments/env-in-use/bin/python")]})
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    plan = StorageOperations.gc(descriptor(tmp_path), {})
    assert [item["name"] for item in plan["candidates"]] == ["env-obsolete"]
    assert plan["reference_issues"][0]["job_id"] == "job-orphan"
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_explicit_reference_repair_is_audited_and_never_removes_scientific_files(tmp_path):
    environment(tmp_path, "env-obsolete")
    job = tmp_path / "jobs/job-incomplete"
    job.mkdir(parents=True)
    (job / "checkpoint.npz").write_bytes(b"science")
    config = descriptor(tmp_path)
    assert not StorageOperations.gc(config, {})["candidates"]
    selection = {"retire_jobs": ["job-incomplete"]}
    preview = StorageOperations.repair_references(config, selection)
    assert not Path(preview["audit_path"]).exists()
    assert preview["retired_jobs"][0]["job_id"] == "job-incomplete"
    StorageOperations.repair_references(config, selection, apply=True)
    plan = StorageOperations.gc(config, {})
    assert len(plan["candidates"]) == 1
    applied = StorageOperations.gc(config, {}, apply=True)
    assert applied["outcome"] == "removed"
    assert (job / "checkpoint.npz").read_bytes() == b"science"
    (job / "state.json").write_text('{"state":"running"}')
    with pytest.raises(ValueError, match="active or unresolved"):
        StorageOperations.repair_references(config, selection, apply=True)


def test_physical_reclaim_does_not_double_count_hardlinks_or_retained_links(tmp_path):
    first = environment(tmp_path, "env-a")
    second = environment(tmp_path, "env-b")
    (second / "heavy.bin").unlink()
    os.link(first / "heavy.bin", second / "heavy.bin")
    allocated = first.joinpath("heavy.bin").stat().st_blocks * 512
    assert StorageOperations._reclaimable_allocation([first / "heavy.bin"]) == 0
    assert (
        StorageOperations._reclaimable_allocation([first / "heavy.bin", second / "heavy.bin"])
        == allocated
    )
    os.link(first / "heavy.bin", tmp_path / "retained-payload")
    assert (
        StorageOperations._reclaimable_allocation([first / "heavy.bin", second / "heavy.bin"]) == 0
    )


def test_build_repair_requires_controller_and_remote_descendant_proof(tmp_path):
    cache = tmp_path / "cache"
    marker = cache / ".environment-build-env-old.lock"
    marker.mkdir(parents=True)
    owner = {"host": socket.gethostname(), "pid": 2147483647, "heartbeat": time.time() - 600}
    (marker / "owner.json").write_text(json.dumps(owner))
    selection = {"release_builds": [{"name": marker.name, "owner": owner}]}
    plan = StorageOperations.repair_references(descriptor(tmp_path), selection)
    assert plan["released_builds"] and marker.exists()
    prefix = cache / "environments/.env-old.tmp-owned/bin/python"
    with subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", str(prefix)]
    ) as process:
        try:
            with pytest.raises(ValueError, match="descendants"):
                StorageOperations.repair_references(descriptor(tmp_path), selection, apply=True)
        finally:
            process.terminate()
            process.wait(timeout=5)
    StorageOperations.repair_references(descriptor(tmp_path), selection, apply=True)
    assert not marker.exists()
    foreign = {**owner, "host": "unavailable-controller"}
    marker.mkdir()
    (marker / "owner.json").write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="unverified"):
        StorageOperations.repair_references(
            descriptor(tmp_path), {"release_builds": [{"name": marker.name, "owner": foreign}]}
        )


def test_native_transported_collector_and_hypothetical_repair_are_read_only(tmp_path):
    from types import SimpleNamespace

    from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
    from lambdaforge.controlplane.StorageService import StorageService

    config = descriptor(tmp_path)
    environment(tmp_path, "env-old")
    job = tmp_path / "jobs/job-missing"
    job.mkdir(parents=True)
    profile = ClusterProfile(
        "remote-fixture",
        transport="ssh",
        host="not-contacted.invalid",
        workspace=str(tmp_path),
        python=sys.executable,
        storage=ClusterStoragePolicy.from_mapping(config, workspace=str(tmp_path)),
    )
    service = StorageService(
        ClusterCatalog({"remote-fixture": profile}),
        SimpleNamespace(transport=lambda _: LocalTransport()),
        jobs=SimpleNamespace(list=lambda **_: (), store=SimpleNamespace(project=None)),
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    preview = service.repair("remote-fixture", retire_jobs=["job-missing"])
    assert preview["cleanup_preview"]["hypothetical_after_repairs"]
    assert len(preview["cleanup_preview"]["candidates"]) == 1
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="cannot authorize deletion"):
        service.gc("remote-fixture", apply=True, reference_preview=preview)
    assert environment_path(tmp_path, "env-old").exists()
    service.repair("remote-fixture", retire_jobs=["job-missing"], apply=True)
    result = service.gc("remote-fixture", apply=True, environments_only=True)
    assert result.reclaimed_bytes > 0 and result.outcome == "removed"
    assert job.is_dir()


def environment_path(root, name):
    return root / "cache/environments" / name


def test_repair_snapshot_and_live_owner_are_revalidated_on_clean(tmp_path):
    config = descriptor(tmp_path)
    environment(tmp_path, "env-old")
    job = tmp_path / "jobs/job-old"
    job.mkdir(parents=True)
    selection = {"retire_jobs": ["job-old"]}
    StorageOperations.repair_references(config, selection, apply=True)
    assert StorageOperations.gc(config, {})["candidates"]
    (job / "request.json").write_text('{"command":["python"]}')
    (job / "state.json").write_text("{bad")
    observed = StorageOperations.gc(config, {})
    assert observed["reference_issues"]
    (job / "request.json").unlink()
    with subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)", str(job / "work")]
    ) as process:
        try:
            with pytest.raises(ValueError, match="active or unresolved"):
                StorageOperations.repair_references(config, selection, apply=True)
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_remote_recovery_dependency_retains_original_environment_and_workspace(
    tmp_path, monkeypatch
):
    environment(tmp_path, "env-original")
    environment(tmp_path, "env-recovery")
    old = tmp_path / "jobs/job-old"
    new = tmp_path / "jobs/job-new"
    for path, state, command in (
        (old, "failed", [str(environment_path(tmp_path, "env-original") / "bin/python")]),
        (
            new,
            "running",
            [
                str(environment_path(tmp_path, "env-recovery") / "bin/python"),
                "--resume-execution",
                str(old / "work/execution-test"),
            ],
        ),
    ):
        path.mkdir(parents=True)
        (path / "state.json").write_text(json.dumps({"state": state}))
        (path / "request.json").write_text(json.dumps({"command": command}))
    monkeypatch.setattr(
        StorageOperations,
        "compact_job",
        lambda *_a, **_kw: pytest.fail("Retained original must not be compacted"),
    )
    plan = StorageOperations.gc(descriptor(tmp_path), {})
    assert not plan["candidates"]
    assert {item["name"] for item in plan["protected_items"]} == {"env-original", "env-recovery"}


def test_refresh_metadata_does_not_walk_environment_files(tmp_path, monkeypatch):
    config = descriptor(tmp_path)
    environment(tmp_path, "env-large")
    monkeypatch.setattr(
        StorageOperations,
        "_usage",
        lambda *_: pytest.fail("Live refresh must not walk payload trees"),
    )
    report = StorageOperations.status({**config, "metadata_only": True})
    assert report["environments"]["measurement"] == "not-measured"
    assert "bytes" not in report["environments"]


def test_explicit_clean_keeps_shared_base_until_last_reference_disappears(tmp_path):
    base = environment(tmp_path, "deps-shared")
    code = environment(tmp_path, "env-code")
    (code / ".lambdaforge-environment.json").write_text(
        json.dumps({"dependency_environment": "deps-shared"})
    )
    config = descriptor(tmp_path)
    active = StorageOperations.gc(config, {"environments": ["env-code"]})
    assert not active["candidates"]
    obsolete = StorageOperations.gc(config, {})
    assert {item["name"] for item in obsolete["candidates"]} == {"env-code", "deps-shared"}
    StorageOperations.gc(config, {}, apply=True)
    assert not base.exists() and not code.exists()


def test_preparation_guard_protects_environment_until_durable_provider_handoff(tmp_path):
    from pathlib import PurePosixPath

    from lambdaforge.controlplane.CacheBuildLease import CacheBuildLease

    code = environment(tmp_path, "env-preparing")
    other = environment(tmp_path, "env-other")
    state = tmp_path / "state"
    state.mkdir()
    (state / "active-environment").write_text(str(other / "bin/python"))
    config = descriptor(tmp_path)
    cache = tmp_path / "cache"
    # Another activation can move the pointer before this Job has a request.json.
    with CacheBuildLease.preparation(
        LocalTransport(),
        cache_root=PurePosixPath(str(cache)),
        environment=code.name,
        bundle_id="bundle-fixture",
        job_id="job-pending",
        work_dir=str(tmp_path / "jobs/job-pending/work"),
        python=sys.executable,
    ):
        plan = StorageOperations.gc({**config, "environments_only": True}, {}, apply=True)
        assert not plan["candidates"] and code.exists()
    plan = StorageOperations.gc({**config, "environments_only": True}, {})
    assert [item["name"] for item in plan["candidates"]] == ["env-preparing"]


def test_bad_heartbeat_foreign_owner_and_reused_pid_are_not_conflated(tmp_path):
    import psutil

    marker = tmp_path / "cache/.environment-build-env-fixture.lock"
    marker.mkdir(parents=True)
    owner = {"host": socket.gethostname(), "pid": 2147483647, "heartbeat": float("nan")}
    (marker / "owner.json").write_text(json.dumps(owner))
    assert not StorageOperations._orphan_lease(marker)
    owner.update(
        pid=os.getpid(),
        heartbeat=time.time() - 600,
        process_birth=psutil.Process().create_time() - 3600,
    )
    (marker / "owner.json").write_text(json.dumps(owner))
    assert StorageOperations._orphan_lease(marker)
    owner["host"] = "another-host"
    (marker / "owner.json").write_text(json.dumps(owner))
    assert not StorageOperations._orphan_lease(marker)


def test_native_installer_rejects_rebuild_resolution_drift_before_publication(tmp_path):
    from lambdaforge.controlplane.CommandResult import CommandResult
    from tests.controlplane.test_cuda_environment_resolution import InstallationTransport

    class DifferentResolution(InstallationTransport):
        def run(self, command, **kwargs):
            if tuple(command[-4:]) == ("-m", "pip", "list", "--format=json"):
                self.commands.append(tuple(command))
                return CommandResult(0, '[{"name":"numpy","version":"2.0"}]')
            return super().run(command, **kwargs)

    profile = ClusterProfile(
        "fixture",
        transport="ssh",
        host="not-contacted.invalid",
        workspace=str(tmp_path),
        environment="managed",
    )
    bundle = ExecutionBundle(
        "bundle",
        tmp_path,
        tmp_path / "config.yaml",
        tmp_path / "manifest.json",
        1,
        environment_id="deps-fixture",
        package_names=("framework.whl",),
    )
    transport = DifferentResolution()
    with pytest.raises(RuntimeError, match="frozen resolution"):
        ManagedEnvironmentProvider()._prepare(
            profile,
            transport,
            bundle,
            remote_bundle_dir="/fixture",
            activate=False,
            framework_check=False,
            dependency_pins=("numpy==1.0",),
            frozen_packages=[{"name": "numpy", "version": "1.0"}],
        )
    assert any("--constraint" in command for command in transport.commands if "install" in command)
    assert not any("os.rename" in " ".join(command) for command in transport.commands)
