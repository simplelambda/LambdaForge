"""Shared model-operation mechanics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.tasks.TaskContext import TaskContext


class ModelOperation:
    """Load model/checkpoint inputs and route batches for operational tasks."""

    def __init__(
        self,
        *,
        model: Mapping[str, Any],
        checkpoints: str | Sequence[str],
        data: Mapping[str, Any] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        model_input_key: str = "x",
        model_output_key: str = "logits",
        device: str = "auto",
    ) -> None:
        if batch_size < 1 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative.")
        self.model_spec = dict(model)
        self.data_spec = dict(data) if data is not None else None
        self.checkpoints = (checkpoints,) if isinstance(checkpoints, str) else tuple(checkpoints)
        if not self.checkpoints:
            raise ValueError("At least one checkpoint is required.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_input_key = model_input_key
        self.model_output_key = model_output_key
        self.device = device

    def models(self, context: TaskContext) -> tuple[nn.Module, ...]:
        """Construct identical models and load every declared weights checkpoint."""
        device = self.resolved_device()
        models: list[nn.Module] = []
        for raw_checkpoint in self.checkpoints:
            model = ObjectFactory.build(self.model_spec)
            if not isinstance(model, nn.Module):
                raise TypeError("Model operation targets must construct torch.nn.Module.")
            checkpoint = context.declared_input_path(raw_checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
            if not isinstance(state, Mapping):
                raise TypeError(f"Checkpoint does not contain a state mapping: {checkpoint}")
            try:
                model.load_state_dict(state)
            except RuntimeError:
                stripped = {str(key).removeprefix("model."): value for key, value in state.items()}
                model.load_state_dict(stripped)
            model.to(device).eval()
            models.append(model)
        return tuple(models)

    def loader(self) -> DataLoader[Any]:
        """Build the declared dataset and deterministic non-shuffled loader."""
        if self.data_spec is None:
            raise ValueError("This model operation requires a data specification.")
        dataset = ObjectFactory.build(self.data_spec)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def predict_batch(self, models: Sequence[nn.Module], batch: Any) -> Mapping[str, Tensor]:
        """Run an ensemble and average matching tensor outputs."""
        device = self.resolved_device()
        moved = self._move(batch, device)
        argument = moved[self.model_input_key] if isinstance(moved, Mapping) else moved
        predictions: list[Mapping[str, Tensor]] = []
        for model in models:
            raw = model(argument)
            output = raw if isinstance(raw, Mapping) else {self.model_output_key: raw}
            if any(not isinstance(value, Tensor) for value in output.values()):
                raise TypeError("Model operation outputs must be tensors or tensor mappings.")
            predictions.append(output)
        keys = tuple(predictions[0])
        if any(tuple(output) != keys for output in predictions):
            raise ValueError("Every ensemble model must return identical output keys.")
        return {
            key: torch.stack([output[key] for output in predictions]).mean(dim=0) for key in keys
        }

    def resolved_device(self) -> torch.device:
        """Resolve auto without ever requiring CUDA."""
        return torch.device(
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.device == "auto"
            else self.device
        )

    @classmethod
    def _move(cls, value: Any, device: torch.device) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: cls._move(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._move(item, device) for item in value)
        return value
