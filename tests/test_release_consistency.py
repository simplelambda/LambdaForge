"""Release metadata, public API and current command grammar remain synchronized."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import lambdaforge
from lambdaforge.cli.parser import build_parser
from lambdaforge.LambdaForgeVersion import LambdaForgeVersion

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_minimal_public_api_are_consistent() -> None:
    source = (ROOT / "src/lambdaforge/_version.py").read_text(encoding="utf-8")
    version = re.search(r'^VERSION = "([^"]+)"$', source, re.MULTILINE)
    assert version is not None
    assert version.group(1) == LambdaForgeVersion.CURRENT == "0.14.0"
    assert lambdaforge.__version__ == "0.14.0"
    assert lambdaforge.__all__ == ["Work", "__version__", "clustering"]


def test_cli_reports_version_and_accepts_only_current_execution_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "lf 0.14.0"
    assert parser.parse_args(["run", "study.yaml"]).command == "run"
    clear = parser.parse_args(["jobs", "clear", "--apply"])
    assert clear.job_command == "clear" and clear.apply is True
    run_detail = parser.parse_args(
        ["show", "training", "--run", "trial-00001-seed-4", "--curve-points", "40"]
    )
    assert run_detail.study_run == "trial-00001-seed-4"
    assert run_detail.curve_points == 40
    assert parser.parse_args(["datasets", "members", "data@1"]).command == "datasets"
    with pytest.raises(ValueError):
        parser.parse_args(["datasets", "build", "old.yaml"])
