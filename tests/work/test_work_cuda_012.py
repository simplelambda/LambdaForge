"""Minimal accelerator integration for the canonical Work execution path."""

from pathlib import Path

import pytest
import torch
import yaml

from lambdaforge.work import WorkConfig, WorkRunner


@pytest.mark.cuda
def test_work_executes_and_records_a_cuda_metric(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable to this Python/PyTorch environment.")
    source = tmp_path / "cuda.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "name": "cuda-work",
                "run": "tests.work_cases.CudaWork",
                "resources": {"gpu": 1},
            }
        ),
        encoding="utf-8",
    )

    result = WorkRunner().run(WorkConfig.from_yaml(source))

    assert result.status == "succeeded"
    assert result.runs[0].metrics == {"cuda_score": 4.0}
