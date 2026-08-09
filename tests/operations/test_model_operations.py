"""Focused inference, evaluation and export task tests."""

from pathlib import Path

import torch

from lambdaforge.operations import EvaluationTask, ExportTask, InferenceTask
from lambdaforge.tasks import TaskContext
from tests.fixtures.UserModel import UserModel


def context(tmp_path: Path, checkpoint: Path) -> TaskContext:
    return TaskContext(
        name="operation",
        run_dir=tmp_path / "run",
        source_dir=tmp_path,
        attempt_id="a",
        config_fingerprint="sha256:x",
        resume=True,
        inputs=({"resolved_path": str(checkpoint)},),
    )


def test_inference_evaluation_and_torchscript_export(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    torch.save(UserModel().state_dict(), checkpoint)
    shared = {
        "model": {"target": "tests.fixtures.UserModel.UserModel"},
        "checkpoints": "model.pt",
        "data": {
            "target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset",
            "params": {"size": 5},
        },
        "batch_size": 2,
    }
    task_context = context(tmp_path, checkpoint)

    inference = InferenceTask(**shared).run(task_context)
    predictions = torch.load(tmp_path / "run" / "predictions.pt", weights_only=True)
    assert predictions["user_logits"].shape == (5, 1)
    assert inference.outputs["sample_count"] == 5

    evaluation = EvaluationTask(
        **shared, metrics=({"target": "tests.fixtures.UserMetric.UserMetric"},)
    ).run(task_context)
    assert 0 <= evaluation.metrics["user_accuracy"] <= 1

    export = ExportTask(
        model=shared["model"], checkpoints="model.pt", example_inputs=([[0.0, 0.0, 0.0, 0.0]],)
    ).run(task_context)
    assert (tmp_path / "run" / "model.pt").is_file()
    assert export.outputs["format"] == "torchscript"
