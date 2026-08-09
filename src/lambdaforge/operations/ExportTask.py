"""Pluggable model export task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.operations.ModelOperation import ModelOperation
from lambdaforge.tasks.ArtifactDeclaration import ArtifactDeclaration
from lambdaforge.tasks.Task import Task
from lambdaforge.tasks.TaskContext import TaskContext
from lambdaforge.tasks.TaskOutput import TaskOutput


class ExportTask(ModelOperation, Task):
    """Export weights through TorchScript, torch.export, ONNX or an injected exporter."""

    def __init__(
        self,
        *,
        example_inputs: Sequence[Any],
        format: str = "torchscript",
        exporter: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.example_inputs = tuple(example_inputs)
        self.format = format
        self.exporter_spec = dict(exporter) if exporter is not None else None
        if len(self.checkpoints) != 1:
            raise ValueError("ExportTask exports exactly one checkpoint.")

    def run(self, context: TaskContext) -> TaskOutput:
        """Create one export artifact under the task run directory."""
        model = self.models(context)[0]
        args = tuple(
            torch.as_tensor(value, device=self.resolved_device()) for value in self.example_inputs
        )
        suffix = {"torchscript": "pt", "torch_export": "pt2", "onnx": "onnx"}.get(
            self.format, "bin"
        )
        path = context.output_path(f"model.{suffix}", create_parent=True)
        if self.exporter_spec is not None:
            exporter = ObjectFactory.build(self.exporter_spec)
            method = getattr(exporter, "export", None)
            if not callable(method):
                raise TypeError("Custom exporters must expose export(model, args, path).")
            method(model, args, path)
        elif self.format == "torchscript":
            torch.jit.trace(model, args, strict=False).save(str(path))
        elif self.format == "torch_export":
            torch.export.save(torch.export.export(model, args), path)
        elif self.format == "onnx":
            torch.onnx.export(model, args, path)
        else:
            raise ValueError(f"Unsupported export format: {self.format!r}.")
        return TaskOutput(
            outputs={"export_path": str(path), "format": self.format},
            artifacts=(ArtifactDeclaration(path.name),),
        )
