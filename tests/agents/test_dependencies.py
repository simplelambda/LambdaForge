"""Declared extra consent/isolation/offline/timeout behavior without user-env installation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lambdaforge.agents.dependencies import DependencyService
from lambdaforge.agents.scope import ProjectScope


@pytest.fixture
def dependencies(tmp_path: Path, monkeypatch):
    import lambdaforge.agents.dependencies as module

    # Unit installers are mocked. Explicitly model an isolated interpreter instead of
    # accidentally relying on pytest running inside a venv (CI may use setup-python).
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(
            executable=sys.executable,
            prefix=str(tmp_path / "virtualenv"),
            base_prefix=str(tmp_path / "system-python"),
        ),
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname="deps"\nversion="0.1"\n')
    return DependencyService(ProjectScope(tmp_path))


def test_preview_consent_known_group_and_no_system_environment_mutation(dependencies, monkeypatch):
    import lambdaforge.agents.dependencies as module

    monkeypatch.setattr(module, "version", lambda name: "2.0.0")
    plan = dependencies.preview()
    assert plan["available"] is False and plan["installation_performed"] is False
    assert plan["argv"][-1] == "mcp>=2.3,<2.4"
    assert plan["argv"][plan["argv"].index("--prefix") + 1] == module.sys.prefix
    assert plan["installer_environment"] == {"PIP_CONFIG_FILE": os.devnull}
    assert not (dependencies.scope.root / ".lambdaforge").exists()
    with pytest.raises(PermissionError, match="explicit"):
        dependencies.install(plan_hash=plan["plan_hash"])
    with pytest.raises(ValueError, match="Unknown"):
        dependencies.preview("arbitrary-package")
    monkeypatch.setattr(module.sys, "prefix", module.sys.base_prefix)
    with pytest.raises(PermissionError, match="system Python"):
        dependencies.preview()


def test_install_existing_extra_is_idempotent_and_stale_plan_rejected(dependencies, monkeypatch):
    import lambdaforge.agents.dependencies as module

    monkeypatch.setattr(module, "version", lambda name: "2.3.0")
    plan = dependencies.preview()
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not reinstall")
    )
    result = dependencies.install(plan_hash=plan["plan_hash"], consent=True)
    assert result["status"] == "already_available"
    with pytest.raises(ValueError, match="changed"):
        dependencies.install(plan_hash="not-the-plan", consent=True)


def test_offline_failed_install_preserves_files_and_reports_bounded_error(
    dependencies, monkeypatch
):
    import lambdaforge.agents.dependencies as module

    monkeypatch.setattr(module, "version", lambda name: "2.0.0")
    wheels = dependencies.scope.root / "wheels"
    wheels.mkdir()
    plan = dependencies.preview(wheelhouse="wheels")
    assert "--no-index" in plan["argv"] and plan["network_access"] is False

    class FailedPip:
        returncode = 1

        def __init__(self, argv, **kwargs):
            assert argv == plan["argv"]
            assert kwargs["stdin"] is subprocess.DEVNULL
            assert kwargs["env"]["PIP_CONFIG_FILE"] == os.devnull
            kwargs["stdout"].write(b"x" * 15000 + b"offline wheels missing")

        def wait(self, **kwargs):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(subprocess, "Popen", FailedPip)
    result = dependencies.install(plan_hash=plan["plan_hash"], wheelhouse="wheels", consent=True)
    assert result["status"] == "error" and result["diagnostic"].endswith("offline wheels missing")
    assert len(result["diagnostic"]) <= 12000
    assert (dependencies.scope.root / "pyproject.toml").read_text().startswith("[project]")
    with pytest.raises((ValueError, PermissionError)):
        dependencies.preview(wheelhouse="../external")


def test_install_timeout_kills_only_owned_installer_process(dependencies, monkeypatch):
    import lambdaforge.agents.dependencies as module

    monkeypatch.setattr(module, "version", lambda name: "2.0.0")
    calls = []

    class SlowPip:
        pid = 99999999
        returncode = -9

        def __init__(self, argv, **kwargs):
            self.waited = False

        def wait(self, **kwargs):
            if not self.waited:
                self.waited = True
                raise subprocess.TimeoutExpired("pip", 1)
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            calls.append((self.pid, 9))

    monkeypatch.setattr(subprocess, "Popen", SlowPip)
    monkeypatch.setattr(
        module.os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False
    )
    plan = dependencies.preview()
    result = dependencies.install(plan_hash=plan["plan_hash"], consent=True, timeout_seconds=1)
    assert result["timed_out"] and result["status"] == "error"
    assert calls == [(99999999, 9)]


def test_cli_dependency_failure_returns_nonzero_and_setup_requires_consent(
    dependencies, monkeypatch, capsys
):
    from lambdaforge.cli.ai import run_ai_command
    from lambdaforge.cli.parser import build_parser

    monkeypatch.setattr(
        DependencyService,
        "install",
        lambda self, **kwargs: {"status": "error", "diagnostic": "offline dependencies absent"},
    )
    root = str(dependencies.scope.root)
    command = build_parser().parse_args(
        ["ai", "--project", root, "dependencies", "install", "--apply"]
    )
    assert run_ai_command(command) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    unconsented = build_parser().parse_args(["ai", "--project", root, "init", "--install-deps"])
    with pytest.raises(ValueError, match="requires --apply"):
        run_ai_command(unconsented)
    assert not (dependencies.scope.root / "AGENTS.md").exists()


def test_real_offline_installer_failure_is_isolated_and_preserves_project(tmp_path):
    """Actually invoke pip in a fresh disposable venv, never install into the user's environment."""
    environment = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True, timeout=60)
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="offline"\nversion="0.1"\n')
    (project / "wheels").mkdir()
    (project / "research.md").write_text("Keep this researcher file unchanged.\n")
    checkout = Path(__file__).resolve().parents[2]
    # Base dev dependencies are reused read-only; the protocol SDK's temporary path is excluded.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(checkout / "src"),
                str(
                    Path(sys.prefix) / "Lib" / "site-packages"
                    if os.name == "nt"
                    else Path(sys.prefix)
                    / "lib"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}"
                    / "site-packages"
                ),
            )
        ),
    }
    child = subprocess.run(
        [
            str(environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
            "-c",
            "import json,sys; "
            "from lambdaforge.agents.dependencies import DependencyService,EXTRAS; "
            "EXTRAS['agents']=('lambdaforge-offline-fixture-missing==0',); "
            "from lambdaforge.agents.scope import ProjectScope; "
            "d=DependencyService(ProjectScope(sys.argv[1])); p=d.preview(wheelhouse='wheels'); "
            "print(json.dumps(d.install(plan_hash=p['plan_hash'], "
            "wheelhouse='wheels', consent=True)))",
            str(project),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    outcome = json.loads(child.stdout)
    assert outcome["isolated_environment"] is True
    assert outcome["status"] == "error" and outcome["installation_performed"] is True
    assert outcome["return_code"] != 0 and outcome["diagnostic"]
    assert outcome["environment"] == str(environment)
    assert (project / "research.md").read_text() == "Keep this researcher file unchanged.\n"
    assert not list(project.rglob("execution.json"))
