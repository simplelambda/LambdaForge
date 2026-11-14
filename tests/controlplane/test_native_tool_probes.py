"""Execute managed native version probes against disposable, package-owned binaries."""

from __future__ import annotations

import json
import sys
import venv
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CommandResult import CommandResult
from lambdaforge.controlplane.EnvironmentPublication import PUBLICATION_CONTRACT
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.NativeEnvironment import NativeEnvironmentPlan
from lambdaforge.controlplane.NativeLibraryPolicy import LIBRARY_CONTRACT, REPORT_MARKER


@pytest.fixture
def prefix(tmp_path: Path) -> Path:
    """A real Python prefix; no package solve, dependency installation or GPU needed."""
    root = tmp_path / "managed prefix"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    (root / "conda-meta").mkdir()
    return root


def _tool(
    prefix: Path,
    *,
    name: str = "apbs",
    package: str = "apbs",
    version: str = "3.4.1",
    stdout: str = "APBS 3.4.1\n",
    stderr: str = "",
    status: int = 13,
    fallback_stdout: str = "invalid option: version",
    fallback_status: int = 2,
) -> NativeEnvironmentPlan:
    # Use the test interpreter for the fake executable, not PATH or an installed APBS.
    path = prefix / "bin" / name
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        f" sys.stdout.write({stdout!r}); sys.stderr.write({stderr!r}); sys.exit({status})\n"
        f"sys.stdout.write({fallback_stdout!r}); sys.exit({fallback_status})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    provenance = {
        "name": package,
        "version": version,
        "build": "test_0",
        "channel": "conda-forge",
        "subdir": "linux-64",
    }
    record = {**provenance, "files": [f"bin/{name}"]}
    (prefix / "conda-meta" / "package.json").write_text(json.dumps(record), encoding="utf-8")
    return NativeEnvironmentPlan(
        specification_sha256=f"sha256:{'a' * 64}",
        specification_name="environment.yml",
        specification_kind="environment-file",
        platform="linux-64",
        manager_version="2.8.1-0",
        channels=("conda-forge",),
        requested_packages=(package,),
        exact_packages=(f"conda-forge::{package}={version}=test_0",),
        package_inventory=(provenance,),
        required_executables=(name,),
        offline=False,
    )


def _probe(prefix: Path, native: NativeEnvironmentPlan) -> CommandResult:
    return ManagedEnvironmentProvider._native_tools(
        LocalTransport(), str(prefix / "bin" / "python"), native
    )


@pytest.mark.parametrize("stream", ["stdout", "stderr", "both"])
def test_apbs_341_version_exit_13_is_verified_and_recorded(prefix: Path, stream: str) -> None:
    native = _tool(
        prefix,
        stdout={"stdout": "APBS 3.4.1\n", "stderr": "", "both": "startup banner\n"}[stream],
        stderr="APBS 3.4.1\n" if stream != "stdout" else "",
    )
    identity = native.to_dict()
    result = _probe(prefix, native)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)[0]
    assert "APBS 3.4.1" in evidence["version"]
    assert evidence["path"] == str(prefix / "bin" / "apbs")
    for key in ("package", "package_version", "build", "channel", "subdir"):
        assert (
            evidence[key]
            == {
                "package": "apbs",
                "package_version": "3.4.1",
                "build": "test_0",
                "channel": "conda-forge",
                "subdir": "linux-64",
            }[key]
        )
    assert evidence["probe"]["argv"] == [str(prefix / "bin" / "apbs"), "--version"]
    assert evidence["probe"]["exit_code"] == 13
    assert evidence["probe"]["accepted"] is True
    assert evidence["probe"]["expected_banner"] == "APBS 3.4.1"
    assert native.to_dict() == identity


@pytest.mark.parametrize(
    "changes",
    [
        {"status": 1},
        {"status": 12},
        {"status": 14},
        {"status": -15},
        {"stdout": ""},
        {"stdout": "APBS not available: version 3.4.1"},
        {"stdout": "error while loading APBS 3.4.1"},
        {"stdout": "APBS 3.4.10\n"},
        {"stdout": "APBS 3.4.1 broken\n"},
        {"version": "3.5.0", "stdout": "APBS 3.5.0\n"},
        {"version": "3.4.0"},
        {"package": "other-package"},
        {"package": "other-package", "status": 0},
        {"name": "other-tool"},
        {"stdout": "APBS 3.4.0\n", "status": 0},
    ],
)
def test_nonzero_exception_is_not_a_general_bypass(prefix: Path, changes: dict[str, Any]) -> None:
    result = _probe(prefix, _tool(prefix, **changes))
    assert result.returncode != 0
    assert "could not query executable version:" in result.stderr
    diagnostics = json.loads(result.stderr.split("; probes=", 1)[1])
    assert len(diagnostics) == 2
    assert diagnostics[0]["argv"][-1] == "--version"
    assert diagnostics[1]["argv"][-1] == "version"
    assert all(probe["accepted"] is False for probe in diagnostics)


def test_apbs_nonzero_exception_does_not_apply_to_fallback_arguments(prefix: Path) -> None:
    native = _tool(
        prefix, stdout="bad option", status=1, fallback_stdout="APBS 3.4.1\n", fallback_status=13
    )
    result = _probe(prefix, native)
    assert result.returncode != 0
    probes = json.loads(result.stderr.split("; probes=", 1)[1])
    assert probes[1]["exit_code"] == 13
    assert probes[1]["accepted"] is False


@pytest.mark.parametrize(
    "name,package,version",
    [("apbs", "apbs", "3.4.1"), ("apbs", "apbs", "3.5.0"), ("mmseqs", "mmseqs2", "18.8cc5c")],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_ordinary_success_and_fallback_remain_supported(
    prefix: Path, name: str, package: str, version: str, fallback: bool
) -> None:
    banner = f"APBS {version}\n" if name == "apbs" else f"{version}\n"
    result = _probe(
        prefix,
        _tool(
            prefix,
            name=name,
            package=package,
            version=version,
            stdout="" if fallback else banner,
            status=0,
            fallback_stdout=banner,
            fallback_status=0,
        ),
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)[0]
    assert banner.strip() == record["version"]
    assert record["probe"]["exit_code"] == 0
    assert record["probe"]["argv"][-1] == ("version" if fallback else "--version")


def test_failure_reports_arguments_exit_code_and_both_streams(prefix: Path) -> None:
    native = _tool(prefix, stdout="startup", stderr="missing library", status=127)
    result = _probe(prefix, native)
    assert result.returncode != 0
    probes = json.loads(result.stderr.split("; probes=", 1)[1])
    assert probes[0]["exit_code"] == 127
    assert probes[0]["stdout"] == "startup"
    assert probes[0]["stderr"] == "missing library"
    assert probes[0]["expected_banner"] == "APBS 3.4.1"
    assert probes[1]["stdout"] == "invalid option: version"


def test_whitespace_only_response_is_not_version_evidence(prefix: Path) -> None:
    native = _tool(prefix, name="other-tool", stdout=" \n", stderr="\t\n", status=0)
    result = _probe(prefix, native)
    assert result.returncode != 0
    probes = json.loads(result.stderr.split("; probes=", 1)[1])
    assert probes[0]["exit_code"] == 0
    assert probes[0]["accepted"] is False


def test_os_error_is_not_silenced(prefix: Path) -> None:
    native = _tool(prefix)
    (prefix / "bin" / "apbs").write_text("#!/missing/interpreter\n", encoding="utf-8")
    result = _probe(prefix, native)
    assert result.returncode != 0
    probes = json.loads(result.stderr.split("; probes=", 1)[1])
    assert all("FileNotFoundError" in item["error"] for item in probes)
    assert all(item["exit_code"] is None for item in probes)


@pytest.mark.parametrize("invalid", ["missing", "not-executable", "unowned", "outside"])
def test_apbs_policy_preserves_executable_integrity_checks(prefix: Path, invalid: str) -> None:
    native = _tool(prefix)
    executable = prefix / "bin" / "apbs"
    if invalid == "missing":
        executable.unlink()
    elif invalid == "not-executable":
        executable.chmod(0o644)
    elif invalid == "unowned":
        (prefix / "conda-meta" / "package.json").unlink()
    else:
        outside = prefix.parent / "apbs"
        executable.rename(outside)
        executable.symlink_to(outside)
    result = _probe(prefix, native)
    assert result.returncode != 0
    assert result.stdout == ""


def mocked_library_report(command: Sequence[str]) -> str:
    """Fixture substitute for scientific imports, never used for actual loader regressions."""
    for value in command:
        if not value.startswith('{"contract":'):
            continue
        policy = json.loads(value)
        if policy.get("contract") == LIBRARY_CONTRACT:
            return REPORT_MARKER + json.dumps(
                {"policy": policy, "loaded": {}, "validated_libraries": []}
            )
    raise AssertionError("Library report requested without a selected loader policy")


class RuntimeCheckedTransport(LocalTransport):
    """Fake Torch/manager availability, not executable or inventory verification."""

    def run(
        self, command: Sequence[str], *, cwd: str | Path | None = None, timeout: float | None = None
    ) -> CommandResult:
        # The runtime import script also contains assertions and CUDA checks.
        if any(REPORT_MARKER in item for item in command):
            return CommandResult(0, mocked_library_report(command))
        if any(item.startswith("import lambdaforge,ssl,sys,torch\n") for item in command):
            return CommandResult(0)
        if tuple(command[1:3]) == ("list", "--json"):
            prefix = Path(command[command.index("--prefix") + 1])
            record = json.loads((prefix / "conda-meta" / "package.json").read_text())
            return CommandResult(0, json.dumps([record]))
        return super().run(command, cwd=cwd, timeout=timeout)


def test_bootstrap_and_cache_reuse_share_probe_without_skipping_inventory(prefix: Path) -> None:
    native = _tool(prefix)
    profile = ClusterProfile("local")
    python = str(prefix / "bin" / "python")
    transport = RuntimeCheckedTransport()
    # Fresh preparation invokes the same check used by reusable-environment validation.
    verified = ManagedEnvironmentProvider._verify(profile, transport, python, None, None, native)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)[0]["probe"]["exit_code"] == 13
    (prefix / ".lambdaforge-environment.json").write_text(
        json.dumps(
            {"publication_contract": PUBLICATION_CONTRACT, "installation_prefix": str(prefix)}
        ),
        encoding="utf-8",
    )
    reused = ManagedEnvironmentProvider._verify_reusable(
        profile, transport, PurePosixPath(prefix), python, None, None, native
    )
    assert reused.returncode == 0, reused.stderr
    # Same banner remains valid, but drift in the owning Conda package still fails reuse.
    metadata = prefix / "conda-meta" / "package.json"
    record = json.loads(metadata.read_text())
    record["build"] = "tampered_1"
    metadata.write_text(json.dumps(record), encoding="utf-8")
    changed = ManagedEnvironmentProvider._verify_reusable(
        profile, transport, PurePosixPath(prefix), python, None, None, native
    )
    assert changed.returncode != 0
    assert "build" in changed.stderr
