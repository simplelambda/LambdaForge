"""Real CUDA memory protection checks."""

from __future__ import annotations

import pytest
import torch

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.CudaMemoryLimiter import CudaMemoryLimiter
from lambdaforge.hpo.TorchMemoryPreflight import TorchMemoryPreflight
from tests.hpo.test_adaptive_optimizer import _base


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_allocator_cap_and_isolated_representative_preflight(tmp_path) -> None:
    assert torch.cuda.is_available(), "CUDA marker must run only on a CUDA-enabled host."
    total = int(torch.cuda.get_device_properties(0).total_memory)
    budget = min(512 * 1024**2, total // 2)
    fraction = CudaMemoryLimiter().apply(budget)
    try:
        value = torch.ones(1024, device="cuda")
        torch.cuda.synchronize()
        assert value.is_cuda
        assert 0 < fraction < 1
        assert torch.cuda.max_memory_allocated(0) > 0
    finally:
        del value
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(1.0, device=0)

    results = TorchMemoryPreflight().run(
        {"target": "tests.fixtures.CudaMemoryProbe.CudaMemoryProbe", "params": {}},
        devices=[0],
        budget_bytes=budget,
        output_dir=tmp_path,
        grace_seconds=10,
    )
    assert results[0]["peak_allocated_bytes"] > 0
    assert results[0]["peak_reserved_bytes"] <= budget


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_adaptive_optimizer_runs_a_real_gpu_action_with_telemetry(tmp_path) -> None:
    base = _base(tmp_path)
    base["trainer"]["accelerator"] = "gpu"  # type: ignore[index]
    base["execution"] = {"mode": "sequential", "gpus": [0]}
    hpo = dict(base["hpo"])  # type: ignore[arg-type]
    hpo["initialization"] = {"strategy": "sobol", "trials": 1}
    hpo["fidelity"] = {"min": 1, "max": 1, "step": 1}
    hpo["seeds"] = {"values": [7], "confirmation_values": []}
    hpo["memory"] = {"per_job_budget": "512MiB", "allocator_cap": True}
    hpo["budget"] = {"max_actions": 1, "max_total_epochs": 1}
    base["hpo"] = hpo

    result = Experiment(base).run()
    assert result.summary["status"] == "ok"  # type: ignore[union-attr]
    state = AdaptiveOptimizerState.load(result.state_path)  # type: ignore[union-attr]
    assert state.observations[0].peak_allocated_bytes > 0
    assert state.observations[0].peak_reserved_bytes <= 512 * 1024**2
