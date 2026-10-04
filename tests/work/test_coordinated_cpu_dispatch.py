"""Real detached ProcessScheduler Jobs supply one central fixed Study's evidence."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.Fleet import Fleet, FleetMember
from lambdaforge.controlplane.FleetPlacement import ExecutionEquivalence
from lambdaforge.controlplane.PreparedShardExecutor import PreparedShardExecutor
from lambdaforge.controlplane.StudyCoordinator import StudyCoordinator
from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.coordinated_dispatch import CoordinatedDispatcher


@pytest.mark.skipif(os.name != "posix", reason="Detached process provider requires POSIX")
@pytest.mark.parametrize("automatic_blocks", [False, True])
def test_fixed_study_across_two_real_direct_cpu_targets(
    tmp_path: Path, automatic_blocks: bool
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="fleet-cpu-test"\nversion="1"\n'
        '[tool.lambdaforge]\nproject_id="fleet-cpu-test"\n'
    )
    source = tmp_path / "study.yaml"
    authored: dict[str, Any] = {
        "name": "distributed-fixed",
        "run": "tests.work_cases.AdaptiveScoreWork",
        "sweep": {"space": {"quality": {"values": [0.2, 0.8]}}},
        "resources": {"cpu": 2},
        "execution": {"max_parallel": 2},
        "objective": {"metric": "score", "mode": "max", "range": [0, 1]},
    }
    if automatic_blocks:
        authored["execution"]["max_runs"] = 6
    else:
        authored["seeds"] = [4, 7]
    source.write_text(yaml.safe_dump(authored))
    config = WorkConfig.from_yaml(source)
    plan = WorkRunner().plan(config)
    control = StudyCoordinator(tmp_path / "coordinator")
    members = (FleetMember("A", max_runs=1), FleetMember("B", max_runs=1))
    control.initialize(plan.scientific_fingerprint, plan.execution_id, Fleet("cpu", members))
    eq = ExecutionEquivalence("test-code", "test-env", "no-inputs", "fp32", "local-cpu")
    # Tests' consumer package is deliberately not installed; inherit only its import root.
    import_root = str(Path(__file__).resolve().parents[2])
    executors: dict[str, PreparedShardExecutor] = {}
    dispatcher: CoordinatedDispatcher

    def invocation(key: str) -> dict[str, Any]:
        return dict(dispatcher.invocation(key))

    for member in members:
        profile = ClusterProfile(
            member.cluster,
            python=sys.executable,
            workspace=str(tmp_path / member.cluster),
            command_prefix=("env", "PYTHONPATH=" + import_root),
        )
        executors[member.cluster] = PreparedShardExecutor(
            profile,
            member,
            root=tmp_path / member.cluster / "shards",
            resources=ResourceRequest(cpu_cores=1),
            equivalence=eq,
            invocation=invocation,
        )
    dispatcher = CoordinatedDispatcher(control, executors)
    started = time.monotonic()
    result = WorkRunner(dispatcher=dispatcher).run(config)
    assert time.monotonic() - started < 60
    assert result.status == "succeeded"
    expected_runs = 6 if automatic_blocks else 4
    assert len(result.runs) == expected_runs
    assert result.scientific_fingerprint == plan.scientific_fingerprint
    assert Counter(item.metrics["score"] for item in result.runs) == {
        0.2: expected_runs // 2,
        0.8: expected_runs // 2,
    }
    records = control.run_records()
    assert len({item["run_key"] for item in records}) == expected_runs
    assert {item["attempts"][-1]["cluster"] for item in records} == {"A", "B"}
    assert all(len(item["attempts"]) == 1 for item in records)
    assert all(
        "study_design" not in dispatcher.invocation(item["run_key"])["definition"]
        for item in records
    )
    assert (result.execution_dir / "analysis.json").is_file()
    assert len(list(tmp_path.rglob("analysis.json"))) == 1
    assert not list(tmp_path.glob("*/shards/**/decisions.jsonl"))
    if automatic_blocks:
        blocks = json.loads(
            (result.execution_dir / "hpo-control" / "sweep-blocks.json").read_text()
        )
        assert len(blocks["blocks"]) == 3
        assert all(block["completed"] == 2 for block in blocks["blocks"])
        assert len({item.seed for item in result.runs}) == 3
    # Reconnect/re-read accepted evidence, without another provider launch.
    acknowledged = list(tmp_path.glob("*/shards/*/submission.json"))
    assert WorkRunner(dispatcher=dispatcher).run(config).execution_id == result.execution_id
    assert list(tmp_path.glob("*/shards/*/submission.json")) == acknowledged
