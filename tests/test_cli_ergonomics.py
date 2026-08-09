"""Focused tests for scaffold and discoverability commands."""

import json
from pathlib import Path
from typing import Any

from lambdaforge.cli import CommandLineInterface


def test_init_creates_complete_non_overwriting_consumer(tmp_path: Path, capsys: Any) -> None:
    project = tmp_path / "consumer"
    assert CommandLineInterface.main(["init", str(project)]) == 0
    capsys.readouterr()
    assert (project / "src/my_project/tasks.py").is_file()
    assert (project / "schemas/lambdaforge-task.schema.json").is_file()
    assert json.loads((project / ".vscode/settings.json").read_text(encoding="utf-8"))
    ignored = (project / ".gitignore").read_text(encoding="utf-8")
    assert ".lambdaforge/" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored
    assert "*.whl" in ignored
    assert "slurm-*.out" in ignored
    assert CommandLineInterface.main(["init", str(project)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_explain_and_target_make_public_contracts_discoverable(capsys: Any) -> None:
    assert CommandLineInterface.main(["explain", "workflow", "nodes"]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "object"
    assert CommandLineInterface.main(["target", "lambdaforge.hpo.RandomSearch"]) == 0
    assert "RandomSearch(" in capsys.readouterr().out
