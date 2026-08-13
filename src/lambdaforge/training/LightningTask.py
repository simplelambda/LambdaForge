"""Implementation of the LightningTask object."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from lambdaforge.integrations.Lightning import LightningModuleBase
from lambdaforge.metrics.Metric import Metric
from lambdaforge.nn.losses.Loss import Loss
from lambdaforge.training.TaskLoggingConfig import TaskLoggingConfig


class LightningTask(LightningModuleBase):
    r"""Lightning adapter for project models, losses and metrics.

    This class wraps a standard PyTorch model and connects it to the project's
    ``Loss`` and ``Metric`` interfaces.

    Expected conventions
    --------------------
    If ``model_input_key`` is not ``None``:

        outputs = model(batch[model_input_key])

    If ``model_input_keys`` is a sequence:

        outputs = model(*(batch[key] for key in model_input_keys))

    If ``model_input_keys`` is a mapping:

        outputs = model(**{argument: batch[key] for argument, key in model_input_keys.items()})

    If both routing options are ``None``:

        outputs = model(batch)

    If the model returns a tensor, it is wrapped as:

        {model_output_key: outputs}

    Losses must follow:

        loss(outputs, batch, context) -> torch.Tensor

    Metrics must follow:

        metric.reset()
        metric.update(outputs, batch, context)
        metric.compute() -> float

    Parameters
    ----------
    model : nn.Module
        Model to train.
    losses : Loss | Sequence[Loss]
        One or more losses. They are summed.
    metrics : Sequence[Metric] | None
        Metrics used for train and validation. Validation metrics are deep
        copied by default so train and validation states are independent.
    val_metrics : Sequence[Metric] | None
        Optional explicit validation metrics.
    test_metrics : Sequence[Metric] | None
        Optional explicit test metrics.
    optimizer_cls : type[torch.optim.Optimizer]
        Optimizer class.
    optimizer_kwargs : dict[str, Any] | None
        Keyword arguments passed to the optimizer.
    optimizer_group_kwargs : Mapping[str, Mapping[str, Any]] | None
        Per-group optimizer overrides keyed by names returned from
        ``model.parameter_groups()``. Remaining trainable task parameters form
        a ``task`` group.
    scheduler_cls : type | None
        Optional scheduler class.
    scheduler_kwargs : dict[str, Any] | None
        Keyword arguments passed to the scheduler.
    scheduler_config : dict[str, Any] | None
        Optional Lightning scheduler config. For example:
        ``{"monitor": "val_loss", "interval": "epoch"}``.
    model_input_key : str | None
        Single batch key passed to the model. Mutually exclusive with
        ``model_input_keys``. If both options are ``None``, the full batch is
        passed.
    model_input_keys : Sequence[str] | Mapping[str, str] | None
        Multiple model inputs. A sequence routes positional arguments in the
        given order; a mapping routes ``model argument -> batch key`` keyword
        arguments. This keeps graph, multimodal and conditional models usable
        from YAML without a custom training task.
    model_output_key : str
        Key used when the model returns a raw tensor.
    logging : TaskLoggingConfig | Mapping[str, Any] | None
        Policy controlling loss/metric publication, progress-bar visibility
        and distributed logging. YAML mappings are accepted directly.
    """

    def __init__(
        self,
        model: nn.Module,
        losses: Loss | Sequence[Loss],
        metrics: Sequence[Metric] | None = None,
        val_metrics: Sequence[Metric] | None = None,
        test_metrics: Sequence[Metric] | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        optimizer_group_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_cls: type | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
        scheduler_config: dict[str, Any] | None = None,
        model_input_key: str | None = "x",
        model_input_keys: Sequence[str] | Mapping[str, str] | None = None,
        model_output_key: str = "logits",
        logging: TaskLoggingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.model = model
        loss_list = list(losses) if isinstance(losses, Sequence) else [losses]
        if any(not isinstance(loss, Loss) for loss in loss_list):
            raise TypeError("Every loss must subclass lambdaforge.nn.losses.Loss.")
        self._validate_unique_names(loss_list, "losses")
        self.losses = nn.ModuleList(loss_list)
        self.train_metrics = list(metrics or [])
        try:
            self.val_metrics = (
                list(val_metrics) if val_metrics is not None else copy.deepcopy(self.train_metrics)
            )
            self.test_metrics = (
                list(test_metrics) if test_metrics is not None else copy.deepcopy(self.val_metrics)
            )
        except Exception as error:
            raise TypeError(
                "Metrics reused across stages must support deepcopy; alternatively "
                "configure explicit train_metrics, val_metrics and test_metrics."
            ) from error
        for stage, stage_metrics in (
            ("train_metrics", self.train_metrics),
            ("val_metrics", self.val_metrics),
            ("test_metrics", self.test_metrics),
        ):
            if any(not isinstance(metric, Metric) for metric in stage_metrics):
                raise TypeError(f"Every object in {stage} must subclass Metric.")
            self._validate_unique_names(stage_metrics, stage)

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer_group_kwargs = {
            str(name): dict(group_kwargs)
            for name, group_kwargs in (optimizer_group_kwargs or {}).items()
        }
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_config = scheduler_config

        if model_input_keys is not None and model_input_key not in (None, "x"):
            raise ValueError("model_input_key and model_input_keys are mutually exclusive.")
        if isinstance(model_input_keys, Mapping):
            if not model_input_keys or any(
                not isinstance(argument, str)
                or not argument.strip()
                or not isinstance(key, str)
                or not key.strip()
                for argument, key in model_input_keys.items()
            ):
                raise ValueError(
                    "model_input_keys mappings require non-empty string keys and values."
                )
            self.model_input_keys: tuple[str, ...] | dict[str, str] | None = dict(model_input_keys)
            self.model_input_key = None
        elif model_input_keys is not None:
            if isinstance(model_input_keys, str):
                raise TypeError("model_input_keys must be a sequence of keys, not one string.")
            routed_keys = tuple(model_input_keys)
            if not routed_keys or any(
                not isinstance(key, str) or not key.strip() for key in routed_keys
            ):
                raise ValueError("model_input_keys sequences require non-empty string values.")
            self.model_input_keys = routed_keys
            self.model_input_key = None
        else:
            self.model_input_keys = None
            self.model_input_key = model_input_key
        self.model_output_key = model_output_key
        self.logging = TaskLoggingConfig.from_value(logging)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.forward_model(batch)
        loss, loss_values = self.compute_loss_breakdown(outputs, batch)
        self.log_losses("train", loss, loss_values, batch)

        detached_outputs = self.detach_tree(outputs)
        detached_batch = self.detach_tree(batch)

        for metric in self.train_metrics:
            metric.update(detached_outputs, detached_batch, self)

        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Mapping[str, Any]:
        """Evaluate once and expose detached model outputs to project callbacks."""
        outputs = self.forward_model(batch)
        loss, loss_values = self.compute_loss_breakdown(outputs, batch)
        self.log_losses("val", loss, loss_values, batch)

        detached_outputs = self.detach_tree(outputs)
        detached_batch = self.detach_tree(batch)

        for metric in self.val_metrics:
            metric.update(detached_outputs, detached_batch, self)

        return {"loss": loss.detach(), "model_outputs": detached_outputs}

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.forward_model(batch)
        loss, loss_values = self.compute_loss_breakdown(outputs, batch)
        self.log_losses("test", loss, loss_values, batch)

        detached_outputs = self.detach_tree(outputs)
        detached_batch = self.detach_tree(batch)

        for metric in self.test_metrics:
            metric.update(detached_outputs, detached_batch, self)

        return loss

    def on_train_epoch_start(self) -> None:
        for metric in self.train_metrics:
            metric.reset()

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_metrics:
            metric.reset()

    def on_test_epoch_start(self) -> None:
        for metric in self.test_metrics:
            metric.reset()

    def on_train_epoch_end(self) -> None:
        for metric in self.train_metrics:
            metric.synchronize()
            self.log(
                f"train_{metric.name}",
                float(metric.compute()),
                prog_bar=self.logging.metric_prog_bar,
                logger=self.logging.logger,
                sync_dist=self.logging.sync_dist,
            )

    def on_validation_epoch_end(self) -> None:
        for metric in self.val_metrics:
            metric.synchronize()
            self.log(
                f"val_{metric.name}",
                float(metric.compute()),
                prog_bar=self.logging.metric_prog_bar,
                logger=self.logging.logger,
                sync_dist=self.logging.sync_dist,
            )

    def on_test_epoch_end(self) -> None:
        for metric in self.test_metrics:
            metric.synchronize()
            self.log(
                f"test_{metric.name}",
                float(metric.compute()),
                prog_bar=self.logging.metric_prog_bar,
                logger=self.logging.logger,
                sync_dist=self.logging.sync_dist,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self._optimizer_parameters(),
            **self.optimizer_kwargs,
        )

        if self.scheduler_cls is None:
            return optimizer

        scheduler = self.scheduler_cls(
            optimizer,
            **self.scheduler_kwargs,
        )

        if self.scheduler_config is None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        config = dict(self.scheduler_config)
        config["scheduler"] = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": config,
        }

    def _optimizer_parameters(self) -> Any:
        """Build validated parameter groups only when overrides are requested."""
        if not self.optimizer_group_kwargs:
            return self.parameters()

        provider = getattr(self.model, "parameter_groups", None)
        if provider is None or not callable(provider):
            raise TypeError(
                "optimizer_group_kwargs requires model.parameter_groups() to return a mapping."
            )
        provided = provider()
        if not isinstance(provided, Mapping) or not provided:
            raise TypeError("model.parameter_groups() must return a non-empty mapping.")

        groups: list[dict[str, Any]] = []
        assigned: set[int] = set()
        available: set[str] = set()
        for raw_name, raw_parameters in provided.items():
            name = str(raw_name)
            if not name.strip() or name == "task":
                raise ValueError("Model parameter group names must be non-empty and not 'task'.")
            parameters = tuple(raw_parameters)
            if any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
                raise TypeError(f"Parameter group {name!r} contains a non-Parameter value.")
            if any(id(parameter) in assigned for parameter in parameters):
                raise ValueError(f"Parameter group {name!r} repeats parameters from another group.")
            assigned.update(id(parameter) for parameter in parameters)
            available.add(name)
            if parameters:
                group: dict[str, Any] = {"params": parameters}
                group.update(self.optimizer_group_kwargs.get(name, {}))
                groups.append(group)

        remaining = tuple(
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in assigned
        )
        available.add("task")
        if remaining:
            task_group: dict[str, Any] = {"params": remaining}
            task_group.update(self.optimizer_group_kwargs.get("task", {}))
            groups.append(task_group)

        unknown = set(self.optimizer_group_kwargs) - available
        if unknown:
            raise ValueError(f"Unknown optimizer parameter groups: {sorted(unknown)}.")
        if not groups:
            raise ValueError("No trainable parameters remain for the optimizer.")
        return groups

    def forward_model(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(self.model_input_keys, dict):
            outputs = self.model(
                **{argument: batch[key] for argument, key in self.model_input_keys.items()}
            )
        elif self.model_input_keys is not None:
            outputs = self.model(*(batch[key] for key in self.model_input_keys))
        elif self.model_input_key is None:
            outputs = self.model(batch)
        else:
            outputs = self.model(batch[self.model_input_key])

        if isinstance(outputs, Mapping):
            return outputs

        if torch.is_tensor(outputs):
            return {
                self.model_output_key: outputs,
            }

        raise TypeError("Model output must be a mapping or a tensor.")

    def compute_loss(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> torch.Tensor:
        """Return the summed loss while preserving the established public API."""
        return self.compute_loss_breakdown(outputs, batch)[0]

    def compute_loss_breakdown(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute every loss once and return both total and named values."""
        total: torch.Tensor | None = None
        values: dict[str, torch.Tensor] = {}

        for loss_module in self.losses:
            loss = cast(Loss, loss_module)
            value = loss(outputs, batch, self)
            total = value if total is None else total + value
            values[loss.name] = value

        if total is None:
            raise RuntimeError("LightningTask requires at least one loss.")

        return total, values

    def log_losses(
        self,
        stage: str,
        total: torch.Tensor,
        values: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
    ) -> None:
        """Publish total and individual losses according to the logging policy."""
        batch_size = self.batch_size(batch)
        if self.logging.log_total_loss:
            self.log(
                f"{stage}_loss",
                total.detach(),
                prog_bar=self.logging.loss_prog_bar,
                on_step=self.logging.loss_on_step,
                on_epoch=self.logging.loss_on_epoch,
                logger=self.logging.logger,
                sync_dist=self.logging.sync_dist,
                batch_size=batch_size,
            )
        if self.logging.log_individual_losses:
            for name, value in values.items():
                self.log(
                    f"{stage}_loss_{name}",
                    value.detach(),
                    prog_bar=self.logging.individual_loss_prog_bar,
                    on_step=self.logging.loss_on_step,
                    on_epoch=self.logging.loss_on_epoch,
                    logger=self.logging.logger,
                    sync_dist=self.logging.sync_dist,
                    batch_size=batch_size,
                )

    @staticmethod
    def _validate_unique_names(objects: Sequence[Any], label: str) -> None:
        """Require stable, non-empty and unique names within one logging stage."""
        names: list[str] = []
        for value in objects:
            name = getattr(value, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Every object in {label} must have a non-empty name.")
            names.append(name)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate names in {label}: {duplicates}. Use distinct names or MetricAlias."
            )

    @staticmethod
    def batch_size(batch: Mapping[str, Any]) -> int:
        for value in batch.values():
            if torch.is_tensor(value):
                return int(value.shape[0])

        return 1

    @staticmethod
    def detach_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach()

        if isinstance(value, dict):
            return {key: LightningTask.detach_tree(item) for key, item in value.items()}

        if isinstance(value, list):
            return [LightningTask.detach_tree(item) for item in value]

        if isinstance(value, tuple):
            return tuple(LightningTask.detach_tree(item) for item in value)

        return value
