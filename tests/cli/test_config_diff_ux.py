"""Semantic authoring diffs remain descriptive, not a recovery/extension authority."""

import pytest
import yaml

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.cli.parser import build_parser
from lambdaforge.configuration.ConfigurationGuidance import ConfigurationGuidance


def test_native_diff_classifies_parameters_resources_seeds_without_claiming_recovery(
    tmp_path, monkeypatch
):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="diff-fixture"\nversion="1"\n')
    package = tmp_path / "diff_example"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from lambdaforge import Work\n"
        "class Model(Work):\n"
        "    def run(self, strength: float=0.1): return {'score': 1.0}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    previous = tmp_path / "old.yaml"
    proposed = tmp_path / "new.yaml"
    raw = {"name": "same", "run": "diff_example.Model", "with": {"strength": 0.1}, "seeds": [1, 2]}
    previous.write_text(yaml.safe_dump(raw))
    proposed.write_text(
        yaml.safe_dump(
            {**raw, "with": {"strength": 0.3}, "resources": {"cpu": 3}, "seeds": [1, 2, 3]}
        )
    )
    report = ConfigurationGuidance.diff(previous, proposed)
    assert {row["category"] for row in report["changes"]} >= {"scientific", "operational"}
    assert any(
        row["path"] == "with" and row["proposed"]["strength"] == 0.3 for row in report["changes"]
    )
    assert report["compatible_with_recovery"] == "not evaluated"
    assert not list(tmp_path.rglob("execution.json"))


def test_misspelled_command_or_flag_suggests_only_real_grammar_choices(capsys):
    with pytest.raises(ValueError, match="Did you mean 'validate'"):
        build_parser().parse_args(["valdiate"])
    with pytest.raises(ValueError, match="--format"):
        build_parser().parse_args(["config", "resolve", "file.yaml", "--fomat", "json"])
    assert CommandLineInterface.main(["_complete", "--", "help", "--se"]) == 0
    assert "--search" in capsys.readouterr().out
    assert "==SUPPRESS==" not in build_parser().format_help()
