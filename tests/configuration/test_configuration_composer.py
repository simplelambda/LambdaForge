"""Focused tests for composition, secrets, provenance and semantic diff."""

from pathlib import Path

import pytest

from lambdaforge.configuration import ConfigurationComposer, ConfigurationDiff, SecretValue
from lambdaforge.experiments import ExperimentConfig, ExperimentValidator
from lambdaforge.tasks import TaskRun, TaskValidator


def test_composition_order_delete_interpolation_and_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("model: {width: 8, obsolete: true}\nroot: /data\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "model: {width: 16, obsolete: {$delete: true}}\n"
        'path: "${config:root}/train"\n'
        'token: "${secret:API_TOKEN}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("API_TOKEN", "not-for-output")

    resolved = ConfigurationComposer().resolve(child)

    assert resolved.values["model"] == {"width": 16}
    assert resolved.values["path"] == "/data/train"
    assert isinstance(resolved.values["token"], SecretValue)
    assert resolved.materialized()["token"] == "***"
    assert resolved.materialized(reveal_secrets=True)["token"] == "not-for-output"
    assert resolved.provenance["model.width"] == str(child)
    assert ConfigurationDiff().compare({"a": 1}, {"a": 2, "b": 3}) == {
        "a": {"before": 1, "after": 2, "change": "changed"},
        "b": {"before": None, "after": 3, "change": "added"},
    }


def test_composition_rejects_cycles(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\na: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("include: a.yaml\nb: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        ConfigurationComposer().resolve(tmp_path / "a.yaml")


def test_composed_task_uses_secret_without_persisting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TASK_MESSAGE", "runtime-secret")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        'kind: task\nschema_version: "1.0"\nname: secret-task\n'
        f"output_root: {tmp_path / 'runs'}\n"
        "task:\n  target: tests.fixtures.UserTask.UserTask\n"
        '  params: {message: "${secret:TASK_MESSAGE}"}\n',
        encoding="utf-8",
    )

    result = TaskRun.from_yaml(task_path).run()

    assert TaskValidator().validate_file(task_path).is_valid
    assert result.outputs["message"] == "runtime-secret"
    persisted = (Path(result.run_dir) / "config.yaml").read_text(encoding="utf-8")
    assert "runtime-secret" not in persisted
    assert "'***'" in persisted


def test_experiment_loader_composes_before_schema_migration(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        'schema_version: "1.1"\nexperiment: {name: composed, seeds: [7]}\n'
        "data: {}\n"
        "model: {target: builtins.dict, params: {width: 8}}\n"
        "losses: [{target: builtins.dict}]\n",
        encoding="utf-8",
    )
    child = tmp_path / "study.yaml"
    child.write_text(
        "extends: base.yaml\nmodel: {params: {width: 16}}\n",
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(child)

    assert ExperimentValidator().validate_file(child, check_imports=False).is_valid
    assert config.value("experiment.name") == "composed"
    assert config.value("model.params.width") == 16
