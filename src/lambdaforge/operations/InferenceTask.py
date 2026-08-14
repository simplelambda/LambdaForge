"""Batched checkpoint inference task."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from lambdaforge.operations.ModelOperation import ModelOperation
from lambdaforge.tasks.artifacts import ArtifactDeclaration
from lambdaforge.tasks.Task import Task
from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskOutput import TaskOutput


class InferenceTask(ModelOperation, Task):
    """Generate deterministic batched predictions from one model or ensemble."""

    def run(self, context: TaskContext) -> TaskOutput:
        """Persist CPU prediction tensors and return their sample count."""
        models = self.models(context)
        collected: dict[str, list[torch.Tensor]] = defaultdict(list)
        with torch.inference_mode():
            for batch in self.loader():
                for key, value in self.predict_batch(models, batch).items():
                    collected[key].append(value.detach().cpu())
        predictions: dict[str, Any] = {
            key: torch.cat(values, dim=0) if values else torch.empty(0)
            for key, values in collected.items()
        }
        path = context.output_path("predictions.pt", create_parent=True)
        torch.save(predictions, path)
        sample_count = next(iter(predictions.values())).shape[0] if predictions else 0
        return TaskOutput(
            outputs={"prediction_path": str(path), "sample_count": sample_count},
            artifacts=(
                ArtifactDeclaration("predictions.pt", media_type="application/vnd.pytorch"),
            ),
        )
