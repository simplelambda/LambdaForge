"""Resource declarations must become the scheduler allocation for every YAML family."""

from __future__ import annotations

from pathlib import Path

import yaml

from lambdaforge.execution.ConfigurationResourceResolver import ConfigurationResourceResolver


def _task(path: Path, name: str, resources: dict[str, object]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "kind": "task",
                "schema_version": "1.0",
                "name": name,
                "task": {"target": "builtins.dict"},
                "resources": resources,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_dataset_stage_resources_become_one_safe_serial_reservation(tmp_path: Path) -> None:
    _task(tmp_path / "first.yaml", "first", {"cpus": 4, "memory": "8GiB", "time": "2h"})
    _task(
        tmp_path / "second.yaml",
        "second",
        {"cpus": 8, "memory": "4GiB", "gpus": 1, "processes": 2, "time": "3h"},
    )
    recipe = tmp_path / "dataset.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "kind": "dataset",
                "schema_version": "1.0",
                "dataset": {"name": "records", "version": "1"},
                "max_parallel": 1,
                "stages": {
                    "first": {"task": "first.yaml"},
                    "second": {"task": "second.yaml", "needs": ["first"]},
                },
                "publish": {"from": "second", "index": "members.jsonl"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    request = ConfigurationResourceResolver.resolve(recipe)

    assert request.cpu_cores == 8
    assert request.ram_bytes == 8 * 1024**3
    assert request.gpu_count == 1
    assert request.processes == 2
    assert request.runtime_seconds == 5 * 3600

    value = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    value["resources"] = {"cpus": 12, "memory": "32GiB", "gpus": 2, "time": "8h"}
    recipe.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    explicit = ConfigurationResourceResolver.resolve(recipe)
    assert explicit.cpu_cores == 12
    assert explicit.ram_bytes == 32 * 1024**3
    assert explicit.gpu_count == 2
    assert explicit.runtime_seconds == 8 * 3600


def test_parallel_workflow_sums_concurrent_capacity_and_accepts_exact_override(
    tmp_path: Path,
) -> None:
    first = _task(tmp_path / "first.yaml", "first", {})
    second = _task(tmp_path / "second.yaml", "second", {})
    workflow = tmp_path / "workflow.yaml"
    value = {
        "kind": "workflow",
        "schema_version": "1.0",
        "name": "parallel",
        "max_parallel": 2,
        "nodes": {
            "first": {"config": str(first), "resources": {"cpus": 4, "memory": "8GiB"}},
            "second": {
                "config": str(second),
                "resources": {"cpus": 6, "memory": "12GiB", "gpus": 1},
            },
        },
    }
    workflow.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    derived = ConfigurationResourceResolver.resolve(workflow)
    assert derived.cpu_cores == 10
    assert derived.ram_bytes == 20 * 1024**3
    assert derived.gpu_count == 1

    value["resources"] = {
        "cpus": 16,
        "memory": "64GiB",
        "gpus": 2,
        "processes": 4,
        "time": "12h",
    }
    workflow.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    explicit = ConfigurationResourceResolver.resolve(workflow)
    assert explicit.cpu_cores == 16
    assert explicit.ram_bytes == 64 * 1024**3
    assert explicit.gpu_count == 2
    assert explicit.processes == 4
    assert explicit.runtime_seconds == 12 * 3600
