"""Release metadata, public API and current command grammar remain synchronized."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import lambdaforge
from lambdaforge._version import VERSION
from lambdaforge.cli.parser import build_parser
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_minimal_public_api_are_consistent() -> None:
    source = (ROOT / "src/lambdaforge/_version.py").read_text(encoding="utf-8")
    version = re.search(r'^VERSION = "([^"]+)"$', source, re.MULTILINE)
    assert version is not None
    assert version.group(1) == LambdaForgeVersion.CURRENT == VERSION
    assert lambdaforge.__version__ == VERSION
    assert lambdaforge.__all__ == ["Work", "__version__", "clustering"]


def test_cli_reports_version_and_accepts_only_current_execution_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"lf {VERSION}"
    assert parser.parse_args(["run", "study.yaml"]).command == "run"
    clear = parser.parse_args(["jobs", "clear", "--apply"])
    assert clear.job_command == "clear" and clear.apply is True
    run_detail = parser.parse_args(
        ["show", "training", "--run", "trial-00001-seed-4", "--curve-points", "40"]
    )
    assert run_detail.study_run == "trial-00001-seed-4"
    assert run_detail.curve_points == 40
    assert parser.parse_args(["datasets", "members", "data@1"]).command == "datasets"
    export = parser.parse_args(["export", "study", "--output", "exports"])
    assert export.command == "export" and export.selector == "study"
    with pytest.raises(ValueError):
        parser.parse_args(["datasets", "build", "old.yaml"])


@pytest.mark.parametrize("suffix", ["", ".es"])
def test_current_release_documentation_is_consistent(suffix: str) -> None:
    readme = (ROOT / f"README{suffix}.md").read_text(encoding="utf-8")
    pins = re.findall(r"lambdaforge(?:\[[^\]]+\])?==([\d.]+)", readme)
    assert pins and all(pin == VERSION for pin in pins)
    assert f"LambdaForge {VERSION}" in (ROOT / f"AGENTS{suffix}.md").read_text(encoding="utf-8")
    minor = ".".join(VERSION.split(".")[:2])
    manual_title = (ROOT / f"docs/MANUAL{suffix}.md").read_text(encoding="utf-8").splitlines()[0]
    assert f"LambdaForge {minor}" in manual_title
    assert f"## [{VERSION}] - " in (ROOT / f"CHANGELOG{suffix}.md").read_text(encoding="utf-8")
