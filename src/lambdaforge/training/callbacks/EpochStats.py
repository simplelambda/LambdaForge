"""Callback that records epoch resource statistics."""

from __future__ import annotations

import os
import sys
import time

import torch

from lambdaforge.integrations.Lightning import (
    CallbackBase,
    LightningModuleBase,
    TrainerType,
)


class EpochStats(CallbackBase):
    r"""Log per-epoch wall time and peak GPU memory.

    These land in the same ``metrics.csv`` row as the losses/metrics logged by
    :class:`~lambdaforge.training.LightningTask.LightningTask`, so the CSV is
    directly plottable against epoch (training curve, time per epoch, memory
    footprint) without post-processing.

    Logged keys
    -----------
    ``epoch_time_s``
        Wall-clock seconds of the training epoch.
    ``gpu_mem_mb``
        Peak CUDA memory allocated during the epoch, in MiB. ``0`` on CPU.
    ``cpu_rss_mb``
        Process resident memory in MiB when available. Uses ``psutil`` if
        installed, otherwise falls back to ``resource`` on Unix-like systems.

    The peak-memory counter is reset at the start of every training epoch so
    the value reflects that epoch alone, not the run-to-date maximum.
    """

    def __init__(self) -> None:
        super().__init__()
        self._epoch_start: float | None = None

    def on_train_epoch_start(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        self._epoch_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleBase) -> None:
        if self._epoch_start is not None:
            pl_module.log(
                "epoch_time_s",
                time.perf_counter() - self._epoch_start,
                on_epoch=True,
                sync_dist=True,
            )

        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        pl_module.log(
            "gpu_mem_mb",
            gpu_mem_mb,
            on_epoch=True,
            sync_dist=True,
        )

        cpu_rss_mb = self._cpu_rss_mb()
        if cpu_rss_mb is not None:
            pl_module.log(
                "cpu_rss_mb",
                cpu_rss_mb,
                on_epoch=True,
                sync_dist=True,
            )

    @staticmethod
    def _cpu_rss_mb() -> float | None:
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        except Exception:
            pass

        try:
            import resource

            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                return rss / (1024**2)
            return rss / 1024
        except Exception:
            return None
