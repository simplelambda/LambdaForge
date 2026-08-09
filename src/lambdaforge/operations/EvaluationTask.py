"""Checkpoint evaluation task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.operations.ModelOperation import ModelOperation
from lambdaforge.tasks.Task import Task
from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskOutput import TaskOutput


class EvaluationTask(ModelOperation, Task):
    """Evaluate a checkpoint or ensemble on a new dataset with streaming metrics."""

    def __init__(self, *, metrics: Sequence[Mapping[str, Any]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.metric_specs = tuple(dict(value) for value in metrics)
        if not self.metric_specs:
            raise ValueError("EvaluationTask requires at least one metric.")

    def run(self, context: TaskContext) -> TaskOutput:
        """Update metrics per batch and return scalar terminal values."""
        models = self.models(context)
        metrics = [ObjectFactory.build(spec) for spec in self.metric_specs]
        for metric in metrics:
            reset = getattr(metric, "reset", None)
            if not callable(reset):
                raise TypeError("Evaluation metrics must expose reset/update/compute.")
            reset()
        with torch.inference_mode():
            for batch in self.loader():
                outputs = self.predict_batch(models, batch)
                metric_batch = self._move(batch, self.resolved_device())
                for metric in metrics:
                    metric.update(outputs, metric_batch)
        values: dict[str, float] = {}
        for index, metric in enumerate(metrics):
            name = str(getattr(metric, "name", f"metric_{index}"))
            value = metric.compute()
            scalar = (
                float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            )
            values[name] = scalar
        return TaskOutput(metrics=values, outputs={"evaluated_batches": len(self.loader())})
