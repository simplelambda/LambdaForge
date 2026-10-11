"""Optional desktop assistance never becomes a dependency of canonical Markdown access."""

import os
from urllib.parse import unquote

import pytest

from lambdaforge.agents.knowledge import KnowledgeService
from lambdaforge.agents.obsidian import ObsidianService
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.agents.services import AgentServices


@pytest.fixture
def desktop(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="desktop"\nversion="0.1"\n')
    vault = KnowledgeService(ProjectScope(tmp_path))
    vault.initialize(obsidian=True)
    return ObsidianService(vault)


def test_missing_and_headless_obsidian_do_not_block_notes_or_launch_commands(desktop, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("lambdaforge.agents.obsidian.sys.platform", "linux")
    monkeypatch.setattr("lambdaforge.agents.obsidian.shutil.which", lambda _: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("Passive diagnosis/Markdown must not launch desktop commands")

    monkeypatch.setattr("lambdaforge.agents.obsidian.subprocess.Popen", forbidden)
    assert desktop.status()["cli_on_path"] is False
    assert desktop.probe_cli(consent=True)["reason"] == "cli_missing"
    assert desktop.open(apply=True)["reason"] == "headless_environment"
    note = desktop.vault.create("finding", "Offline", body="Still accessible")
    assert (
        desktop.vault.search("accessible")["items"][0]["metadata"]["id"] == note["metadata"]["id"]
    )
    agent = AgentServices(desktop.vault.scope.root)
    assert agent.knowledge_obsidian_status().data["markdown_usable_without_app"]


def test_probe_is_explicit_and_compatibility_is_observed_not_assumed(desktop, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":fixture")
    monkeypatch.setattr("lambdaforge.agents.obsidian.shutil.which", lambda _: "/test/obsidian")
    assert desktop.status()["compatible_version"] == "not_probed"
    with pytest.raises(PermissionError, match="start Obsidian"):
        desktop.probe_cli()
    calls = []

    def run(argv):
        calls.append(argv)
        return {"return_code": 0, "timed_out": False, "diagnostic": "Obsidian 1.12.7\n"}

    monkeypatch.setattr(desktop, "_run", run)
    assert desktop.probe_cli(consent=True)["compatible_version"] is True
    assert calls == [["/test/obsidian", "version"]]
    monkeypatch.setattr(
        desktop,
        "_run",
        lambda _: {
            "return_code": 1,
            "timed_out": False,
            "diagnostic": "CLI disabled or app unavailable",
        },
    )
    result = desktop.probe_cli(consent=True)
    assert result["status"] == "error" and result["cli_usable"] is False
    assert result["compatible_version"] is None


def test_open_uses_only_encoded_owned_readme_and_does_not_edit_preferences(desktop, monkeypatch):
    preferences = desktop.vault.root / ".obsidian/config.json"
    preferences.write_text('{"researcher":"preferences"}')
    monkeypatch.setenv("DISPLAY", ":fixture")
    monkeypatch.setattr("lambdaforge.agents.obsidian.sys.platform", "linux")
    monkeypatch.setattr("lambdaforge.agents.obsidian.shutil.which", lambda _: "/test/xdg-open")
    preview = desktop.open()
    assert preview["status"] == "preview" and preview["application_started"] is False
    assert unquote(preview["uri"]).endswith(str(desktop.vault.root / "README.md"))
    assert "%2F" in preview["uri"]
    calls = []

    def run(argv):
        calls.append(argv)
        return {"return_code": 0, "timed_out": False, "diagnostic": ""}

    monkeypatch.setattr(desktop, "_run", run)
    opened = desktop.open(apply=True)
    assert calls == [["/test/xdg-open", preview["uri"]]]
    assert opened["status"] == "open_requested" and opened["opening_verified"] is False
    assert preferences.read_text() == '{"researcher":"preferences"}'
    (desktop.vault.root / "README.md").unlink()
    (desktop.vault.root / "README.md").symlink_to(preferences)
    with pytest.raises(PermissionError, match="Symbolic"):
        desktop.open(apply=True)


@pytest.mark.skipif(os.name != "posix", reason="Owned process group timeout check")
def test_helper_timeout_kills_only_its_own_process_and_bounds_output():
    import sys

    result = ObsidianService._run(
        [sys.executable, "-c", 'print("x" * 10000, flush=True); import time; time.sleep(30)'],
        timeout=1,
    )
    assert result["timed_out"] and result["return_code"] < 0
    assert len(result["diagnostic"]) <= 2048
