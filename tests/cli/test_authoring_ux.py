"""Native authoring/completion/scaffold UX with disposable files and no provider mutation."""

from __future__ import annotations

import importlib
import json
import keyword
import shutil
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.cli.discovery import command_catalog, completion_candidates, completion_script
from lambdaforge.cli.parser import build_parser
from lambdaforge.cli.scaffold import initialize, project_names
from lambdaforge.configuration.ConfigurationGuidance import ConfigurationGuidance


@pytest.mark.parametrize(
    "name", ["wisdom", "my-research", "2026 experiment", "class", "torch", "Ciencia ágil", "你好"]
)
def test_generated_names_are_installable_import_safe_and_actual_yaml_matches(
    tmp_path, monkeypatch, name
):
    root = tmp_path / name
    distribution, package = project_names(name)
    assert package.isidentifier() and not keyword.iskeyword(package)
    assert package != "torch" and package not in sys.stdlib_module_names
    assert initialize(root, force=False) == 0
    pyproject = (root / "pyproject.toml").read_text()
    assert f'name = "{distribution}"' in pyproject
    monkeypatch.syspath_prepend(str(root / "src"))
    importlib.invalidate_caches()
    from lambdaforge.work.config import WorkConfig

    value = WorkConfig.from_yaml(root / "experiments/work.yaml")
    assert value.levels[0].runs[0].work_class == package + ".work.Example"
    assert (root / "tests/test_work_signature.py").exists()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        initialize(root, force=False)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("template", ["preprocessing", "training", "full"])
def test_generated_templates_use_one_native_work_entry(tmp_path, monkeypatch, template):
    root = tmp_path / (template + "-fixture")
    initialize(root, force=False, template=template)
    monkeypatch.syspath_prepend(str(root / "src"))
    from lambdaforge.work.config import WorkConfig

    for path in (root / "experiments").glob("*.yaml"):
        config = WorkConfig.from_yaml(path)
        assert WorkConfig.validate_file(path).valid
        Draft202012Validator(ConfigurationGuidance.schema()).validate(config.to_dict())


@pytest.mark.parametrize(
    "words, expected",
    [
        (["config", ""], "schema"),
        (["clusters", ""], "add"),
        (["init", "--template", "tr"], "training"),
        (["init", "--template=pr"], "--template=preprocessing"),
        (["ai", "mcp", "serve", "--trans"], "--transport"),
    ],
)
def test_completion_walks_native_flags_commands_and_choices(words, expected):
    assert expected in completion_candidates(build_parser(), words)
    assert "_complete" not in completion_candidates(build_parser(), [""])


def test_cli_completion_preserves_debug_json_words_as_candidates(capsys):
    assert CommandLineInterface.main(["_complete", "--", "run", "--j"]) == 0
    assert "--json" in capsys.readouterr().out.splitlines()


def test_task_help_returns_real_recovery_route_and_normal_help_stays_working(capsys):
    assert CommandLineInterface.main(["help", "--search", "recuperar estudio", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["items"][0]["identifier"] == "lf retry"
    catalog = {row["identifier"] for row in command_catalog(build_parser())}
    assert all(row["identifier"] in catalog for row in value["items"])
    for args in (["help"], ["--help"], ["config", "help"]):
        assert CommandLineInterface.main(args) == 0
        assert "usage:" in capsys.readouterr().out


def test_schema_export_is_explicit_non_overwriting_and_json_valid(tmp_path, capsys):
    path = tmp_path / "work.schema.json"
    assert CommandLineInterface.main(["config", "schema", "--output", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["path"] == str(path)
    Draft202012Validator.check_schema(json.loads(path.read_text()))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        # Direct dispatcher preserves technical cause; public main preserves diagnostic JSON.
        CommandLineInterface._dispatch(
            ["config", "schema", "--output", str(path)], json_output=True
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("limit", [None, "auto", "AUTO", 3])
def test_execution_schema_matches_native_policy_and_static_field_defaults(tmp_path, limit):
    from lambdaforge.work.design import ExecutionPolicy

    schema = ConfigurationGuidance.schema()
    policy = {"max_parallel": limit, "runs_per_gpu": limit}
    Draft202012Validator(schema["$defs"]["execution"]).validate(policy)
    assert ExecutionPolicy.from_mapping(policy).max_parallel == (3 if limit == 3 else None)
    path = tmp_path / "scientific.yaml"
    path.write_text("run: deliberately.not_importable\nexecution:\n  max_parallel: 3\n")
    help_value = ConfigurationGuidance.explain_field(path, "execution.max_parallel")
    assert help_value["configured"] == help_value["normalized"] == 3
    assert help_value["default"] == "auto" and not help_value["project_code_imported"]


def test_light_commands_import_no_torch_or_consumer_code_in_a_fresh_process(tmp_path):
    path = tmp_path / "work.yaml"
    path.write_text("run: would_fail.Import\nexecution: {max_time: 2h}\n")
    code = (
        "import sys; from lambdaforge.cli.CommandLineInterface import CommandLineInterface as C; "
        f"C.main(['config','explain',{str(path)!r},'--field','execution.max_time']); "
        "C.main(['config','schema']); C.main(['_complete','--','config','']); "
        "assert 'torch' not in sys.modules; assert 'would_fail' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=10)


def test_bash_completion_script_is_syntactically_valid_and_has_no_install_writes():
    if shutil.which("bash") is None:
        pytest.skip("Bash is not installed on this platform.")
    subprocess.run(["bash", "-n"], input=completion_script("bash"), text=True, check=True)
    for shell in ("bash", "zsh", "fish"):
        assert "_complete" in completion_script(shell)
        assert "Activate:" in completion_script(shell)
