"""Frozen JSON key sorting must not renumber native historical Trial identities."""

import json

import pytest
import yaml

from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.extension_evidence import (
    inspect_extension_execution,
    read_extension_configuration,
)
from lambdaforge.work.recovery import restore_seed_context, validate_recovery_configuration
from tests.controlplane.test_study_extension_preview import execute, service


@pytest.fixture(name="project")
def fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    value = project.__wrapped__(tmp_path, monkeypatch)
    value["code"].write_text(value["code"].read_text().replace(
        "source: Path | None = None", "repeat: int = 0, source: Path | None = None"
    ))
    value["raw"]["seeds"] = [11, 23]
    value["raw"]["sweep"]["space"] = {
        "strength": [0.0, 0.1],
        "repeat": [0, 1],
        "fail": {"values": [False], "when": {"strength": 0.0}},
    }
    value["path"].write_text(yaml.safe_dump(value["raw"], sort_keys=False))
    return value


def test_frozen_order_uses_owned_snapshot_not_edited_source_and_survives_wire(project):
    origin = execute(project)
    root = origin.execution_dir
    manifest = json.loads((root / "execution.json").read_text())
    raw = json.loads((root / "configuration.json").read_text())
    reordered = restore_seed_context(WorkConfig.from_mapping(raw, source=project["path"]), manifest)
    assert WorkRunner._study_identity(reordered, code_identity=manifest["code_identity"]) != (
        manifest["scientific_fingerprint"]
    )
    original_bytes = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    project["raw"]["sweep"]["include"] = [{"strength": 0.2, "repeat": 1}]
    project["path"].write_text(yaml.safe_dump(project["raw"], sort_keys=False))
    snapshot = inspect_extension_execution(str(root))
    assert len(snapshot["evidence"]) == 8
    # Metadata paging canonically sorts dictionary keys: explicit ordered JSON must survive.
    transported = json.loads(json.dumps(snapshot, sort_keys=True))
    ordered = json.loads(transported["configuration_ordered_json"])
    restored = restore_seed_context(
        WorkConfig.from_mapping(ordered, source=project["path"]), manifest
    )
    assert validate_recovery_configuration(restored, manifest) == manifest["scientific_fingerprint"]
    assert list(ordered["sweep"]["space"]) == ["strength", "repeat", "fail"]
    plan = service(project).preview(origin.execution_id, project["path"])
    assert plan["comparison"]["counts"] == {"reused": 8, "new": 2, "total": 10}
    assert original_bytes == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for ref in plan["reused_evidence"]:
        prior = next(r for r in origin.runs if r.run_id == ref["run_id"])
        assert ref["original_trial"] == prior.trial["index"]


@pytest.mark.parametrize("failure", ["changed", "symlink", "corrupt", "missing"])
def test_snapshot_does_not_waive_identity_or_path_checks(project, tmp_path, failure):
    origin = execute(project)
    root = origin.execution_dir
    snapshot = root / "authored.yaml"
    if failure == "changed":
        raw = yaml.safe_load(snapshot.read_text())
        raw["seeds"] = [11, 99]
        snapshot.write_text(yaml.safe_dump(raw))
    elif failure == "corrupt":
        snapshot.write_text("sweep: [unterminated")
    else:
        snapshot.unlink()
        if failure == "symlink":
            snapshot.symlink_to(project["path"])
    with pytest.raises(ValueError):
        inspect_extension_execution(str(root))


def test_semantic_equality_never_coerces_integer_and_float_snapshot_values(tmp_path):
    (tmp_path / "configuration.json").write_text(json.dumps({"with": {"x": 1}}))
    (tmp_path / "authored.yaml").write_text("with: {x: 1.0}\n")
    with pytest.raises(ValueError, match="differs from frozen"):
        read_extension_configuration(tmp_path)


def test_remote_source_shipped_reader_and_paging_preserve_native_trial_order(project, monkeypatch):
    from tests.controlplane.test_study_extension_remote import remote_fixture

    svc, job_id, _, work = remote_fixture(project, monkeypatch)
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    project["raw"]["sweep"]["include"] = [{"strength": 0.2, "repeat": 1}]
    project["path"].write_text(yaml.safe_dump(project["raw"], sort_keys=False))
    plan = svc.preview(job_id, project["path"])
    assert plan["comparison"]["counts"] == {"reused": 8, "new": 2, "total": 10}
    assert len({row["original_trial"] for row in plan["reused_evidence"]}) == 4
    assert before == {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
