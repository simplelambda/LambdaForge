"""Version, entry-point and durable CLI behavior checks independent of release names."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lambdaforge.cli import CommandLineInterface
from lambdaforge.controlplane import ClusterCatalog, ClusterProfile, JobService, JobState, JobStore
from lambdaforge.execution import ResourceRequest
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_and_entry_points_are_consistent() -> None:
    """Use runtime metadata as the source instead of hard-coding each future release."""
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
    assert version is not None
    assert version.group(1) == LambdaForgeVersion.CURRENT
    scripts = dict(re.findall(r'^(lambdaforge|lf) = "([^"]+)"$', project, re.MULTILINE))
    assert scripts["lf"] == scripts["lambdaforge"]
    assert f"`{LambdaForgeVersion.CURRENT}`" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{LambdaForgeVersion.CURRENT}`" in (ROOT / "docs" / "MANUAL.md").read_text(
        encoding="utf-8"
    )


def test_cli_reports_the_runtime_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Keep the installation check in the root README executable."""
    with pytest.raises(SystemExit) as exit_info:
        CommandLineInterface._parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"lambdaforge {LambdaForgeVersion.CURRENT}"


def test_documented_cli_forms_share_the_canonical_parser() -> None:
    """Protect current aliases and resource/action grammar."""
    parser = CommandLineInterface._parser()
    forms = (
        ["datasets", "plan", "example-records", "--on", "atlas"],
        ["datasets", "members", "example-records@1", "--partition", "split=train"],
        ["datasets", "diff", "example-records@1", "example-records@2", "--json"],
        ["completion", "bash"],
    )
    for form in forms:
        assert parser.parse_args(form).command == form[0]
    assert CommandLineInterface._normalize_arguments(["ds", "ls"]) == ["datasets", "list"]
    assert CommandLineInterface._normalize_arguments(["plan", "study.yaml"]) == [
        "run",
        "study.yaml",
        "--dry-run",
    ]


def test_dry_run_is_terminal_planned_and_latest_is_deterministic(tmp_path: Path) -> None:
    """Keep a preview visible without leaving a fake active job."""
    catalog = ClusterCatalog({"local": ClusterProfile("local")})
    jobs = JobService(catalog, JobStore(tmp_path / "jobs"))
    handle = jobs.submit(
        ("python", "-c", "print('not executed')"),
        cluster="local",
        resources=ResourceRequest(),
        work_dir=tmp_path,
        dry_run=True,
        metadata={"name": "preview"},
    )
    assert handle.state is JobState.PLANNED
    assert jobs.get(handle.job_id, refresh=False).state.terminal
    assert jobs.resolve_selector("latest") == handle.job_id
    assert jobs.resolve_selector("preview") == handle.job_id


def test_default_cluster_has_visible_project_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit project default wins over the user catalog and remains auditable."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="consumer"\nversion="1"\n', encoding="utf-8"
    )
    (project / "lambdaforge.yaml").write_text("default_cluster: project-gpu\n", encoding="utf-8")
    user = tmp_path / "config" / "lambdaforge"
    user.mkdir(parents=True)
    (user / "clusters.yaml").write_text("default_cluster: user-gpu\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.chdir(project)
    assert CommandLineInterface._default_cluster() == ("project-gpu", "project default")
    assert CommandLineInterface._normalize_arguments(["plan", "study"]) == [
        "run",
        "study",
        "--dry-run",
        "--on",
        "project-gpu",
    ]
