"""Real CUDA memory protection checks."""

from __future__ import annotations

import csv

import pytest
import torch

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveActionKind import AdaptiveActionKind
from lambdaforge.hpo.AdaptiveOptimizerConfig import AdaptiveOptimizerConfig
from lambdaforge.hpo.AdaptiveOptimizerState import AdaptiveOptimizerState
from lambdaforge.hpo.AdaptiveRunMaterializer import AdaptiveRunMaterializer
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
def test_candidate_aware_preflight_and_isolated_oom_censoring(tmp_path) -> None:
    total = int(torch.cuda.get_device_properties(0).total_memory)
    budget = min(256 * 1024**2, total // 4)
    preflight = TorchMemoryPreflight()
    result = preflight.run(
        {
            "target": "tests.fixtures.CandidateAwareCudaMemoryProbe.CandidateAwareCudaMemoryProbe",
            "params": {},
        },
        devices=[0],
        budget_bytes=budget,
        output_dir=tmp_path / "candidate",
        configuration={"probe_width": 64},
        resource_context={
            "memory_budget_bytes": budget,
            "resource_features": {"batch_size": 8},
        },
        label="candidate",
        grace_seconds=15,
    )[0]
    assert result["status"] == "ok"
    assert result["candidate_aware"] is True

    oom = preflight.run(
        {"target": "tests.fixtures.CudaOOMMemoryProbe.CudaOOMMemoryProbe", "params": {}},
        devices=[0],
        budget_bytes=budget,
        output_dir=tmp_path / "oom",
        configuration={"candidate": "large"},
        resource_context={"memory_budget_bytes": budget},
        label="oom",
        grace_seconds=15,
    )[0]
    assert oom["status"] == "oom"
    assert oom["lower_bound_bytes"] == budget
    assert torch.cuda.is_available()


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checkpoint_resume_executes_only_incremental_epochs(tmp_path) -> None:
    base = _base(tmp_path)
    base["trainer"]["accelerator"] = "gpu"  # type: ignore[index]
    base["execution"] = {"mode": "sequential", "gpus": [0]}
    optimizer = AdaptiveOptimizerConfig.from_experiment(base)
    parameters = {"optimizer.params.lr": 0.001}
    actions = (
        AdaptiveAction(
            "cuda-resume-1",
            AdaptiveActionKind.START_NEW,
            "cuda-resume",
            parameters,
            7,
            0,
            1,
        ),
        AdaptiveAction(
            "cuda-resume-2",
            AdaptiveActionKind.RESUME,
            "cuda-resume",
            parameters,
            7,
            1,
            3,
        ),
    )
    materializer = AdaptiveRunMaterializer()
    runner = ExperimentRunner()

    for action in actions:
        result = runner.run_single_experiment(materializer.materialize(base, action, optimizer))
        assert result.status.value == "ok"

    final_config = materializer.materialize(base, actions[-1], optimizer)
    metrics = runner.experiment_run_dir(final_config) / "metrics.csv"
    with open(metrics, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(float(row["epoch"])) for row in rows] == [1, 2, 3]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_single_gpu_runs_multiple_isolated_adaptive_trials(tmp_path) -> None:
    base = _base(tmp_path)
    base["trainer"]["accelerator"] = "gpu"  # type: ignore[index]
    base["execution"] = {"mode": "parallel", "gpus": [0], "jobs_per_gpu": 2}
    hpo = dict(base["hpo"])  # type: ignore[arg-type]
    hpo["initialization"] = {"strategy": "sobol", "trials": 2}
    hpo["fidelity"] = {"min": 1, "max": 1, "step": 1}
    hpo["seeds"] = {"values": [7], "confirmation_values": []}
    hpo["memory"] = {"per_job_budget": "512MiB", "allocator_cap": True}
    hpo["budget"] = {"max_actions": 2, "max_total_epochs": 2}
    hpo["max_concurrency"] = 2
    base["hpo"] = hpo

    result = Experiment(base).run()

    state = AdaptiveOptimizerState.load(result.state_path)  # type: ignore[union-attr]
    assert result.summary["status"] == "ok"  # type: ignore[union-attr]
    assert len(state.observations) == 2
    assert all(item.peak_reserved_bytes > 0 for item in state.observations)


@pytest.mark.cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two visible GPUs are required")
def test_multi_gpu_preflight_uses_logical_visible_devices(tmp_path) -> None:
    capacities = [int(torch.cuda.get_device_properties(index).total_memory) for index in (0, 1)]
    budget = min(256 * 1024**2, *(capacity // 4 for capacity in capacities))

    results = TorchMemoryPreflight().run(
        {"target": "tests.fixtures.CudaMemoryProbe.CudaMemoryProbe", "params": {}},
        devices=[0, 1],
        budget_bytes=budget,
        output_dir=tmp_path,
        label="multi-gpu",
        grace_seconds=15,
    )

    assert set(results) == {0, 1}
    assert all(result["status"] == "ok" for result in results.values())
