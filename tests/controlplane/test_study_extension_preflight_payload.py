"""Large scientific metadata must not become an oversized remote command argument."""

import copy
import json
import os
import shlex
import site
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.StudyExtensionDispatch import (
    delta_contract_digests,
    native_delta_contract,
    prepare_native_delta,
)
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.work import WorkConfig


@pytest.fixture(name="project")
def fixture(tmp_path, monkeypatch):
    from tests.controlplane.test_study_extension_preview import project

    return project.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("changed", [None, "description", "parameters", "seeds"])
def test_large_native_contract_is_checked_exactly_with_small_read_only_command(
    project, monkeypatch, changed
):
    # More metadata than the real V6a failure, authored through native AnalysisProfile.
    raw = copy.deepcopy(project["raw"])
    marker = "full-sized-scientific-description-"
    raw["analysis"] = {"metrics": {
        f"metric_{index}": {"description": marker + "x" * 3500, "direction": "max"}
        for index in range(180)
    }}
    project["path"].write_text(yaml.safe_dump(raw))
    expected = native_delta_contract(WorkConfig.from_yaml(project["path"]))
    assert len(json.dumps(expected).encode()) > 600_000
    work = project["root"] / "owned-job" / "work"
    work.mkdir(parents=True)
    config = work / "config.yaml"
    staged = copy.deepcopy(raw)
    if changed == "description":
        staged["analysis"]["metrics"]["metric_0"]["description"] += "modified"
    elif changed == "parameters":
        staged["with"] = {"fail": True}
    elif changed == "seeds":
        staged["seeds"] = [999]
    config.write_text(yaml.safe_dump(staged))
    # A minimal native embedded identity; no Work/Execution is executed or invented.
    (work / "code-identity.json").write_text(json.dumps({
        "provider": "distribution", "revision": "disposable-fixture",
    }))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([
        str(project["root"] / "src"), str(Path(__file__).resolve().parents[2] / "src"),
        *site.getsitepackages(),
    ]))
    commands = []

    class LimitedControlChannel(LocalTransport):
        def run(self, command, **kwargs):
            # Catch regression on either SSH exec limits or OS single-argument limits.
            assert len(shlex.join(command).encode()) < 16 * 1024
            assert marker not in shlex.join(command)
            commands.append(tuple(command))
            return super().run(command, **kwargs)

    context = ProjectContext.discover(project["root"])
    profile = ClusterCatalog({"local": ClusterProfile("local")}, project=context).get("local")
    prepared = SimpleNamespace(
        profile=profile, transport=LimitedControlChannel(), python=sys.executable,
        config=str(config),
    )
    original = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    if changed is None:
        result = prepare_native_delta(prepared, expected)
        assert result["scientific_identity"].startswith("sha256:")
    else:
        with pytest.raises(ValueError, match="Prepared extension differs from reviewed"):
            prepare_native_delta(prepared, expected)
    assert len(commands) == 1
    assert original == {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    assert not list(work.rglob("execution.json"))


def test_contract_commitments_keep_exact_types_and_all_fields():
    original = {"parameters": {"x": True}, "analysis_semantics": {"label": "á"}}
    digests = delta_contract_digests(original)
    assert digests == delta_contract_digests(dict(reversed(list(original.items()))))
    for value in (1, 1.0, "1", False):
        changed = {**original, "parameters": {"x": value}}
        assert delta_contract_digests(changed)["parameters"] != digests["parameters"]
    assert delta_contract_digests({**original, "analysis_semantics": {"label": "a"}}) != digests
    assert set(delta_contract_digests({**original, "extra": None})) == {*original, "extra"}
    with pytest.raises(ValueError):
        delta_contract_digests({"parameters": {"x": float("nan")}})
