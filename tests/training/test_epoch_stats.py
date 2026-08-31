"""Runtime metric semantics for training dashboards."""

from types import SimpleNamespace

import torch

from lambdaforge.training.callbacks.EpochStats import EpochStats
from lambdaforge.work.runner import _release_unused_cuda_cache


class RecordingModule:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def log(self, name: str, value: float, **_: object) -> None:
        self.values[name] = value


def test_epoch_stats_distinguish_live_allocations_from_allocator_cache(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2 * 1024**2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 6 * 1024**2)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 8 * 1024**2)
    callback = EpochStats()
    module = RecordingModule()

    callback.on_train_epoch_start(SimpleNamespace(), module)  # type: ignore[arg-type]
    callback.on_train_epoch_end(SimpleNamespace(), module)  # type: ignore[arg-type]

    assert module.values["gpu_mem_mb"] == 2
    assert module.values["gpu_reserved_mb"] == 6
    assert module.values["gpu_peak_reserved_mb"] == 8
    assert module.values["epoch_time_s"] >= 0


def test_finished_packed_run_releases_only_unused_cuda_cache(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("lambdaforge.work.runner.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    _release_unused_cuda_cache()

    assert calls == ["gc", "empty_cache"]
