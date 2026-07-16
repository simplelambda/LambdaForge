"""Implementation of the LightningTrainConfig object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lambdaforge.training.CheckpointPolicy import CheckpointPolicy
from lambdaforge.training.LoggerMode import LoggerMode
from lambdaforge.training.MatmulPrecision import MatmulPrecision
from lambdaforge.training.MonitorMode import MonitorMode


@dataclass
class LightningTrainConfig:
    r"""Configuration for a Lightning training run.

    Parameters
    ----------
    max_epochs : int
        Maximum number of training epochs.
    accelerator : str
        Lightning accelerator. Common values are ``"auto"``, ``"cpu"``,
        ``"gpu"``.
    devices : Any
        Devices passed to Lightning. Examples: ``"auto"``, ``1``, ``[0, 1]``.
    strategy : str
        Lightning strategy. Examples: ``"auto"``, ``"ddp"``.
    precision : str
        Precision setting. Examples: ``"32-true"``, ``"16-mixed"``,
        ``"bf16-mixed"``. Reduced precision is safe for the loss layer:
        numerically fragile losses are evaluated in ``float32`` regardless (see
        :class:`lambdaforge.nn.losses.Loss.Loss`). On Tensor-Core GPUs
        ``"bf16-mixed"`` is the recommended trade-off (about half the memory and
        a large speedup at negligible metric cost).
    matmul_precision : str
        Precision of ``float32`` matmuls on Tensor Cores, forwarded to
        ``torch.set_float32_matmul_precision``. This only changes how matmuls
        *accumulate* internally; tensor values, geometry and reductions stay
        ``float32``. ``"highest"`` keeps true ``float32`` (slowest); ``"high"``
        uses TF32; ``"medium"`` (default) uses a ``bfloat16`` accumulate (fastest,
        negligible metric impact in practice). The chosen value is printed to the
        run log. Kept identical across a sweep so runs stay comparable; set
        ``"highest"`` only to reproduce a previous pure-``float32`` baseline.
    accumulate_grad_batches : int
        Gradient accumulation steps.
    gradient_clip_val : float | None
        Optional gradient clipping value.
    check_val_every_n_epoch : int
        Validation frequency in epochs.
    default_root_dir : str | Path
        Root directory for logs and checkpoints.
    log_every_n_steps : int
        Logging frequency.
    checkpoint_policy : CheckpointPolicy
        Checkpoint saving policy.
    checkpoint_monitor, checkpoint_mode
        Optional explicit Lightning key and optimization direction used for
        best-checkpoint selection. By default the first validation metric is
        used, falling back to ``val_loss``.
    early_stopping_patience : int | None
        If not ``None``, enable early stopping with this patience.
    early_stopping_min_delta : float
        Minimum improvement required by early stopping.
    early_stopping_monitor, early_stopping_mode
        Optional explicit key and direction for early stopping.
    num_sanity_val_steps : int
        Number of validation sanity-check steps before training.
    enable_progress_bar : bool
        Whether to show Lightning progress bar.
    deterministic : bool | None
        Deterministic mode passed to Lightning.
    logger : str
        Optional Lightning logger. ``"csv"`` uses only LambdaForge's dense
        epoch CSV, ``"lightning_csv"`` enables Lightning's native CSV logger,
        and ``"none"`` disables the Lightning logger. A compatible logger
        object is also accepted.
    write_epoch_metrics_csv : bool
        Preserve the canonical dense ``metrics.csv`` artifact independently
        of the selected Lightning logger. Disable explicitly only when the
        experiment does not require LambdaForge aggregation.
    track_epoch_stats : bool
        If true, attach :class:`~lambdaforge.training.callbacks.EpochStats.EpochStats` so
        per-epoch wall time and peak GPU memory land in ``metrics.csv``.
    print_epoch_table : bool
        If true, mirror epoch-level losses, metrics and runtime stats to
        stdout as a compact table. Since experiment runners capture stdout,
        these rows also appear in each run's ``train.log``.
    trainer_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to ``lightning.Trainer``. Keys
        managed explicitly by this class cannot be overridden here.
    epoch_metrics_include, epoch_metrics_exclude
        Optional shell-style key patterns controlling CSV columns.
    epoch_console_include, epoch_console_exclude
        Optional shell-style key patterns controlling the epoch table.
    """

    max_epochs: int = 10

    accelerator: str = "auto"
    devices: Any = "auto"
    strategy: str = "auto"
    precision: str = "32-true"

    matmul_precision: str = MatmulPrecision.MEDIUM.value

    accumulate_grad_batches: int = 1
    gradient_clip_val: float | None = None

    check_val_every_n_epoch: int = 1

    default_root_dir: str | Path = "runs"
    log_every_n_steps: int = 50

    checkpoint_policy: str = CheckpointPolicy.LAST_AND_BEST.value
    checkpoint_monitor: str | None = None
    checkpoint_mode: str | None = None

    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_monitor: str | None = None
    early_stopping_mode: str | None = None

    num_sanity_val_steps: int = 2
    enable_progress_bar: bool = True
    deterministic: bool | None = None

    logger: Any = LoggerMode.CSV.value
    write_epoch_metrics_csv: bool = True
    track_epoch_stats: bool = True
    print_epoch_table: bool = True
    epoch_metrics_include: list[str] | None = None
    epoch_metrics_exclude: list[str] | None = None
    epoch_console_include: list[str] | None = None
    epoch_console_exclude: list[str] | None = None
    trainer_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate framework-owned settings before creating a Trainer."""
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be positive.")
        if self.check_val_every_n_epoch < 1:
            raise ValueError("check_val_every_n_epoch must be positive.")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be positive.")
        if self.num_sanity_val_steps < 0:
            raise ValueError("num_sanity_val_steps cannot be negative.")
        if self.checkpoint_policy not in {item.value for item in CheckpointPolicy}:
            raise ValueError(f"Unknown checkpoint_policy: {self.checkpoint_policy!r}.")
        if self.matmul_precision not in {item.value for item in MatmulPrecision}:
            raise ValueError(f"Unknown matmul_precision: {self.matmul_precision!r}.")
        if isinstance(self.logger, str) and self.logger not in {item.value for item in LoggerMode}:
            raise ValueError(f"Unknown logger mode: {self.logger!r}.")
        valid_monitor_modes = {item.value for item in MonitorMode}
        for field_name in ("checkpoint_mode", "early_stopping_mode"):
            mode = getattr(self, field_name)
            if mode is not None and mode not in valid_monitor_modes:
                raise ValueError(f"Unknown {field_name}: {mode!r}.")
        for field_name in (
            "epoch_metrics_include",
            "epoch_metrics_exclude",
            "epoch_console_include",
            "epoch_console_exclude",
        ):
            patterns = getattr(self, field_name)
            if isinstance(patterns, str):
                raise TypeError(f"{field_name} must be a list of patterns, not a string.")
            if patterns is not None:
                setattr(self, field_name, [str(pattern) for pattern in patterns])
        self.trainer_kwargs = dict(self.trainer_kwargs or {})
