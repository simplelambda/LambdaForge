"""Stable-prefix publication and recovery use only disposable local fixture wheels."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import venv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterStoragePolicy import ClusterStoragePolicy
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.DependencyLayer import DependencyLayer
from lambdaforge.controlplane.EnvironmentIdentity import EnvironmentIdentity
from lambdaforge.controlplane.EnvironmentPublication import PUBLICATION_CONTRACT
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.NativeEnvironment import NativeEnvironmentPlan
from lambdaforge.controlplane.StorageOperations import StorageOperations
from tests.controlplane.test_environment_layers_and_reference_repair import TinyBaseProvider, wheel
from tests.controlplane.test_native_tool_probes import RuntimeCheckedTransport
from tests.controlplane.test_storage_management import descriptor


class CondaFixtureProvider(ManagedEnvironmentProvider):
    """Replace only Conda downloads; run pip, all tool probes, inventory and leases."""

    builds = 0
    broken = False

    def _create_native(self, profile, transport, native, temporary, remote_bundle, trust):
        root = Path(temporary)
        assert root.is_dir() and not list(root.iterdir())  # micromamba's create contract.
        self.builds += 1
        venv.EnvBuilder(with_pip=True, symlinks=True).create(root)
        (root / "conda-meta").mkdir()
        interpreter = root / "bin/python"
        if self.broken:
            interpreter = root.parent / ".deleted.tmp-123/bin/python3.11"
        scripts = {
            "pdb2pqr": f"#!{interpreter}\nprint('pdb2pqr 3.7.1')\n",
            "pdb2pqr-shell": (
                f"#!/bin/sh\n'''exec' '{interpreter}' \"$0\" \"$@\"\n' '''\n"
                "print('pdb2pqr 3.7.1')\n"
            ),
            "pdb2pqr-sh": f"#!/bin/sh\nexec '{interpreter}' -c \"print('pdb2pqr 3.7.1')\"\n",
        }
        for name, content in scripts.items():
            path = root / "bin" / name
            path.write_text(content)
            path.chmod(0o755)
        metadata = {**dict(native.package_inventory[0]), "files": [f"bin/{v}" for v in scripts]}
        (root / "conda-meta/package.json").write_text(json.dumps(metadata))
        # Opaque package content is never edited by the framework's publication protocol.
        (root / "opaque.bin").write_bytes(b"\x00\xffprefix-placeholder\x00")
        return CommandResult(0)


def _fixture(tmp_path: Path) -> tuple[ClusterProfile, ExecutionBundle, NativeEnvironmentPlan]:
    policy = descriptor(tmp_path)
    Path(str(policy["state_root"])).mkdir()
    profile = ClusterProfile(
        "fixture",
        transport="local",
        environment="managed",
        python=sys.executable,
        workspace=str(tmp_path),
        storage=ClusterStoragePolicy.from_mapping(policy, workspace=str(tmp_path)),
    )
    selected = wheel(tmp_path / "bundle/packages")
    native = NativeEnvironmentPlan(
        f"sha256:{'a' * 64}",
        "environment.yml",
        "environment-file",
        "linux-64",
        "2.8.1-0",
        ("conda-forge",),
        ("pdb2pqr",),
        ("conda-forge::pdb2pqr=3.7.1=test_0",),
        (
            {
                "name": "pdb2pqr",
                "version": "3.7.1",
                "build": "test_0",
                "channel": "conda-forge",
                "subdir": "linux-64",
            },
        ),
        ("pdb2pqr", "pdb2pqr-shell", "pdb2pqr-sh"),
        False,
    )
    bundle = ExecutionBundle(
        "bundle-fixture",
        selected.parent.parent,
        selected,
        selected,
        1,
        environment_id="env-publication-fixture",
        package_names=(selected.name,),
        environment_policy={"native_environment": native.to_dict()},
    )
    return profile, bundle, native


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(v.relative_to(root)): v.read_bytes()
        for v in root.rglob("*")
        if v.is_file() and not v.is_symlink() and "__pycache__" not in str(v)
    }


def test_real_rename_reproduces_missing_interpreter_and_no_version_is_accepted(tmp_path: Path):
    profile, bundle, native = _fixture(tmp_path)
    temporary = tmp_path / ".old.tmp-123"
    temporary.mkdir()
    provider = CondaFixtureProvider()
    provider._create_native(profile, None, native, temporary, None, None)
    # A real pip console script also binds to its installation prefix.
    subprocess.run(
        [
            str(temporary / "bin/python"),
            "-m",
            "pip",
            "install",
            "--no-index",
            str(bundle.directory / "packages" / bundle.package_names[0]),
        ],
        check=True,
        capture_output=True,
    )
    assert subprocess.check_output([str(temporary / "bin/fixture-code")], text=True).strip() == "1"
    final = tmp_path / "env-old"
    temporary.rename(final)
    with pytest.raises(FileNotFoundError):
        subprocess.run([str(final / "bin/fixture-code")], check=True)
    shell = subprocess.run(
        [str(final / "bin/pdb2pqr-shell"), "--version"], capture_output=True, text=True
    )
    assert shell.returncode == 127
    assert str(temporary) in shell.stderr
    result = provider._native_tools(RuntimeCheckedTransport(), str(final / "bin/python"), native)
    assert result.returncode != 0
    assert "missing-interpreter" in result.stderr
    assert str(temporary / "bin/python") in result.stderr


def test_real_final_prefix_tools_and_pip_console_survive_publication_and_reuse(tmp_path: Path):
    profile, bundle, native = _fixture(tmp_path)
    provider = CondaFixtureProvider()
    transport = RuntimeCheckedTransport()
    result = provider.prepare(profile, transport, bundle, remote_bundle_dir=bundle.directory)
    prefix = Path(result.python).parent.parent
    receipt = json.loads((prefix / ".lambdaforge-environment.json").read_text())
    assert receipt["publication_contract"] == PUBLICATION_CONTRACT
    assert receipt["installation_prefix"] == str(prefix)
    for name in (*native.required_executables, "fixture-code"):
        value = subprocess.check_output([str(prefix / "bin" / name), "--version"], text=True)
        assert value.strip() == ("1" if name == "fixture-code" else "pdb2pqr 3.7.1")
    assert (prefix / "opaque.bin").read_bytes() == b"\x00\xffprefix-placeholder\x00"
    before = _tree(prefix)
    again = provider.prepare(profile, transport, bundle, remote_bundle_dir=bundle.directory)
    assert again.reused and provider.builds == 1
    assert before == _tree(prefix)
    assert not list(prefix.parent.parent.glob(".environment-build-*.lock"))


def test_preparations_are_serialized_and_gc_cannot_collect_a_build(tmp_path: Path, monkeypatch):
    profile, bundle, _ = _fixture(tmp_path)
    provider = CondaFixtureProvider()
    original = provider._create_native

    def observed(*args):
        preview = StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
        protected = {item["name"] for item in preview["protected_items"]}
        assert bundle.environment_id in protected
        return original(*args)

    monkeypatch.setattr(provider, "_create_native", observed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                provider.prepare,
                profile,
                RuntimeCheckedTransport(),
                bundle,
                remote_bundle_dir=bundle.directory,
            )
            for _ in range(2)
        ]
        results = [future.result() for future in futures]
    assert provider.builds == 1
    assert sorted(value.reused for value in results) == [False, True]


def test_failed_publication_is_diagnosed_not_ready_and_gc_respects_references(tmp_path: Path):
    profile, bundle, _ = _fixture(tmp_path)
    provider = CondaFixtureProvider()
    provider.broken = True
    with pytest.raises(RuntimeError, match="missing-interpreter"):
        provider.prepare(
            profile, RuntimeCheckedTransport(), bundle, remote_bundle_dir=bundle.directory
        )
    root = tmp_path / "cache/environments" / str(bundle.environment_id)
    failure = json.loads((root / ".lambdaforge-publication.json").read_text())
    assert failure["state"] == "failed"
    assert "missing-interpreter" in failure["error"]
    assert not (root / ".lambdaforge-environment.json").exists()
    assert not (tmp_path / "state/active-environment").exists()
    before = _tree(root)
    # No destructive repair, even with a new caller holding the build lease.
    with pytest.raises(RuntimeError, match="already exists"):
        provider.prepare(
            profile, RuntimeCheckedTransport(), bundle, remote_bundle_dir=bundle.directory
        )
    assert _tree(root) == before
    refs = {"environments": [str(bundle.environment_id)]}
    plan = StorageOperations.gc(descriptor(tmp_path), refs, apply=True)
    assert root.exists() and bundle.environment_id not in {v["name"] for v in plan["candidates"]}
    eligible = StorageOperations.gc(descriptor(tmp_path), {})
    assert bundle.environment_id in {v["name"] for v in eligible["candidates"]}
    StorageOperations.gc(descriptor(tmp_path), {}, apply=True)
    assert not root.exists()
    recovered = provider
    recovered.broken = False
    assert not recovered.prepare(
        profile, RuntimeCheckedTransport(), bundle, remote_bundle_dir=bundle.directory
    ).reused


def test_published_broken_prefix_is_detected_and_never_modified(tmp_path: Path):
    profile, bundle, _ = _fixture(tmp_path)
    provider = CondaFixtureProvider()
    prepared = provider.prepare(
        profile, RuntimeCheckedTransport(), bundle, remote_bundle_dir=bundle.directory
    )
    prefix = Path(prepared.python).parent.parent
    (prefix / "bin/pdb2pqr").write_text("#!/definitely/missing/python\n")
    before = _tree(prefix)
    with pytest.raises(RuntimeError, match="preserved|preserve"):
        provider.prepare(
            profile, RuntimeCheckedTransport(), bundle, remote_bundle_dir=bundle.directory
        )
    assert provider.builds == 1 and _tree(prefix) == before


def test_shell_probe_reports_both_127_exits_and_the_missing_interpreter(tmp_path: Path):
    profile, _, native = _fixture(tmp_path)
    native = replace(native, required_executables=("pdb2pqr-shell",))
    prefix = tmp_path / "env-broken"
    prefix.mkdir()
    provider = CondaFixtureProvider()
    provider.broken = True
    provider._create_native(profile, None, native, prefix, None, None)
    result = provider._native_tools(RuntimeCheckedTransport(), str(prefix / "bin/python"), native)
    assert result.returncode != 0
    probes = json.loads(result.stderr.split("; probes=", 1)[1])
    assert [probe["exit_code"] for probe in probes] == [127, 127]
    assert all(probe["diagnostic"] == "missing-interpreter" for probe in probes)
    assert all(probe["interpreter"].endswith(".deleted.tmp-123/bin/python3.11") for probe in probes)


def test_sealing_failure_never_publishes_or_activates_and_keeps_a_diagnostic(tmp_path: Path):
    profile, bundle, _ = _fixture(tmp_path)

    class RejectedSealTransport(RuntimeCheckedTransport):
        def run(self, command, **kwargs):
            if any("payload.update(publication_contract=contract" in item for item in command):
                return CommandResult(1, stderr="fixture readiness write failed")
            return super().run(command, **kwargs)

    with pytest.raises(RuntimeError, match="publication failed.*readiness write failed"):
        CondaFixtureProvider().prepare(
            profile, RejectedSealTransport(), bundle, remote_bundle_dir=bundle.directory
        )
    prefix = tmp_path / "cache/environments" / str(bundle.environment_id)
    failure = json.loads((prefix / ".lambdaforge-publication.json").read_text())
    assert "readiness write failed" in failure["error"]
    assert not (prefix / ".lambdaforge-environment.json").exists()
    assert not (tmp_path / "state/active-environment").exists()


def test_corrupt_frozen_legacy_resolution_blocks_migration_without_repair(tmp_path: Path):
    profile, bundle, _ = _fixture(tmp_path)
    selected = bundle.directory / "packages" / bundle.package_names[0]
    layer = DependencyLayer.policy([selected], {})
    receipt = tmp_path / "state/dependency-resolutions" / f"{layer['legacy_identity']}.json"
    receipt.parent.mkdir()
    receipt.write_text('{"version":1,"identity":"wrong"}')
    broken = receipt.read_bytes()
    modern = replace(bundle, environment_policy={"dependency_layer": layer})
    with pytest.raises(RuntimeError, match="frozen dependency resolution"):
        CondaFixtureProvider().prepare(
            profile, RuntimeCheckedTransport(), modern, remote_bundle_dir=bundle.directory
        )
    assert receipt.read_bytes() == broken
    assert not (tmp_path / "cache/environments" / layer["identity"]).exists()


def test_new_identity_reuses_frozen_legacy_resolution_without_touching_old_prefix(tmp_path: Path):
    profile, bundle, _ = _fixture(tmp_path)
    selected = bundle.directory / "packages" / bundle.package_names[0]
    layer = DependencyLayer.policy([selected], {})
    legacy_id = DependencyLayer.previous_identity(layer)
    legacy = {**layer, "identity": legacy_id}
    legacy.pop("legacy_identity")
    provider = TinyBaseProvider()
    old_bundle = ExecutionBundle(
        "bundle-old",
        bundle.directory,
        selected,
        selected,
        1,
        environment_id="env-old-code",
        package_names=(selected.name,),
        environment_policy={"dependency_layer": legacy},
    )
    provider.prepare(
        profile, RuntimeCheckedTransport(), old_bundle, remote_bundle_dir=bundle.directory
    )
    old_prefix = tmp_path / "cache/environments" / legacy_id
    before = _tree(old_prefix)
    receipt_path = tmp_path / "state/dependency-resolutions" / f"{legacy_id}.json"
    original_receipt = receipt_path.read_bytes()
    modern = ExecutionBundle(
        "bundle-new",
        bundle.directory,
        selected,
        selected,
        1,
        environment_id="env-new-code",
        package_names=(selected.name,),
        environment_policy={"dependency_layer": layer},
    )
    provider.prepare(profile, RuntimeCheckedTransport(), modern, remote_bundle_dir=bundle.directory)
    assert set(provider.last_pins) == {
        f"{v['name']}=={v['version']}" for v in json.loads(original_receipt)["packages"]
    }
    assert _tree(old_prefix) == before
    assert receipt_path.read_bytes() == original_receipt
    assert layer["identity"] != legacy_id
    unsafe: dict[str, Any] = {**layer, "legacy_identity": "../../not-a-prefix"}
    with pytest.raises(ValueError, match="previous dependency"):
        DependencyLayer.previous_identity(unsafe)
    # Monolithic identities also change, independent of the framework's release version.
    identity = EnvironmentIdentity.create([], python_requirement=">=3.10", offline=False)
    historical = {
        "identity_version": 2,
        "wheels": [],
        "python_requirement": ">=3.10",
        "offline": False,
        "dependency_policy": {},
    }
    old_digest = hashlib.sha256(
        json.dumps(historical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert identity.environment_id != f"env-{old_digest[:24]}"
