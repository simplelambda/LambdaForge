"""Focused CPU scheduling, packing, SLURM and reliability tests."""

from pathlib import Path

import pytest

from lambdaforge.execution import (
    FailureCategory,
    FailureClassifier,
    ResourcePlanner,
    ResourceRequest,
    RetryPolicy,
    SlurmExecutionBackend,
)
from lambdaforge.experiments import ExecutionConfig, ExecutionMode


def test_cpu_parallel_slots_and_oversubscription() -> None:
    config = ExecutionConfig.from_mapping(
        {"execution": {"mode": "parallel", "cpu_jobs": 2, "cpu_cores_per_job": 1}}
    )
    assert config.mode is ExecutionMode.PARALLEL
    assert config.slots() == [[], []]
    patched = config.patch_run({"trainer": {}})
    assert patched["trainer"]["accelerator"] == "cpu"


def test_resource_plan_and_slurm_preview(tmp_path: Path) -> None:
    requests = {
        "gpu": ResourceRequest(cpu_cores=2, ram_bytes=4, gpu_count=1, runtime_seconds=20),
        "cpu": ResourceRequest(cpu_cores=2, ram_bytes=4, runtime_seconds=10),
    }
    plan = ResourcePlanner().plan(
        requests, capacity=ResourceRequest(cpu_cores=4, ram_bytes=8, gpu_count=1)
    )
    assert plan.waves == (("gpu", "cpu"),)
    submission = SlurmExecutionBackend(
        partition="research", array="0-2", dependency="afterok:123"
    ).submit(
        ("python", "-m", "lambdaforge", "run", "study.yaml"), requests["gpu"], work_dir=tmp_path
    )
    text = submission.artifact.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in text
    assert "exec python -m lambdaforge run study.yaml" in text
    with pytest.raises(ValueError, match="Unsafe"):
        SlurmExecutionBackend(partition="ok\nmalicious").submit(
            ("true",), requests["cpu"], work_dir=tmp_path
        )


def test_failure_classification_and_bounded_retry() -> None:
    assert (
        FailureClassifier().classify(RuntimeError("CUDA out of memory")) is FailureCategory.GPU_OOM
    )
    attempts: list[int] = []

    def operation(attempt: int, parent: str | None) -> str:
        attempts.append(attempt)
        if attempt == 1:
            raise TimeoutError("temporary")
        assert parent == "attempt-1"
        return "ok"

    assert RetryPolicy(max_attempts=2).execute(operation) == "ok"
    assert attempts == [1, 2]
