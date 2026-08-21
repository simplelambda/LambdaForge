"""Focused tests for scaffold and discoverability commands."""

import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from packaging.version import Version

from lambdaforge import LambdaForge
from lambdaforge._version import VERSION
from lambdaforge.cli import CommandLineInterface


def test_control_plane_reference_commands_parse() -> None:
    parser = CommandLineInterface._parser()
    commands = (
        ["overview", "--json"],
        ["resources", "--all", "--json"],
        ["storage", "status", "--all", "--json"],
        ["clusters", "resources", "atlas", "--json"],
        ["clusters", "storage", "atlas", "--json"],
        ["clusters", "bootstrap", "atlas", "--dry-run"],
        [
            "clusters",
            "add",
            "atlas",
            "--host",
            "atlas-login",
            "--workspace",
            "/scratch/me/lf",
            "--environment",
            "managed",
            "--python-strategy",
            "auto",
            "--python-version",
            "3.13",
        ],
        ["jobs", "list", "--all", "--json"],
        ["jobs", "group", "show", "group-123"],
        ["datasets", "materialize", "corpus", "--on", "atlas"],
        ["experiments", "status", "baseline", "--json"],
        ["experiments", "history", "baseline", "--json"],
        ["tasks", "run", "prepare", "--on", "atlas"],
    )
    for command in commands:
        assert parser.parse_args(command).command == command[0]


def test_init_creates_complete_non_overwriting_consumer(tmp_path: Path, capsys: Any) -> None:
    project = tmp_path / "consumer"
    assert CommandLineInterface.main(["init", str(project)]) == 0
    capsys.readouterr()
    assert (project / "src/my_project/tasks.py").is_file()
    assert (project / "schemas/lambdaforge-task.schema.json").is_file()
    current = Version(VERSION)
    requirement = (
        f"lambdaforge>={current.major}.{current.minor},"
        f"<{current.major}.{current.minor + 1}"
    )
    assert requirement in (project / "pyproject.toml").read_text(encoding="utf-8")
    assert json.loads((project / ".vscode/settings.json").read_text(encoding="utf-8"))
    ignored = (project / ".gitignore").read_text(encoding="utf-8")
    assert ".lambdaforge/" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored
    assert "*.whl" in ignored
    assert "slurm-*.out" in ignored
    assert CommandLineInterface.main(["init", str(project)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().err.lower()


def test_every_scaffold_template_has_valid_python_and_yaml(tmp_path: Path, capsys: Any) -> None:
    """Keep generated consumer projects executable rather than merely present."""
    for template in ("minimal", "preprocessing", "training", "full"):
        project = tmp_path / template
        assert CommandLineInterface.main(["init", str(project), "--template", template]) == 0
        capsys.readouterr()
        for source in project.joinpath("src").rglob("*.py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

        source_root = str(project / "src")
        sys.path.insert(0, source_root)
        importlib.invalidate_caches()
        try:
            for config in project.joinpath("experiments").glob("*.yaml"):
                report = LambdaForge.validate(config, check_imports=True)
                assert report.is_valid, f"{template}/{config.name}: {report.summary()}"
        finally:
            sys.path.remove(source_root)
            for name in tuple(sys.modules):
                if name == "my_project" or name.startswith("my_project."):
                    del sys.modules[name]


def test_explain_and_target_make_public_contracts_discoverable(capsys: Any) -> None:
    assert CommandLineInterface.main(["explain", "workflow", "nodes"]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "object"
    assert CommandLineInterface.main(["target", "lambdaforge.hpo.RandomSearch"]) == 0
    assert "RandomSearch(" in capsys.readouterr().out
