"""Experiment configuration and object-factory contracts."""

from types import SimpleNamespace

import pytest

from lambdaforge.experiments import ExecutionConfig, ExecutionMode, ExperimentConfig, ObjectFactory


class TestExperimentConfig:
    """Verify deterministic YAML-style expansion and resource validation."""

    def test_expands_grid_ablations_and_seeds_without_mutation(self) -> None:
        source = {
            "experiment": {"name": "demo", "seeds": [1, 2]},
            "model": {"params": {"width": 8, "dropout": 0.2}},
            "sweep": {
                "grid": {"model.params.width": [8, 16]},
                "ablations": [{"name": "no_dropout", "set": {"model.params.dropout": 0.0}}],
            },
        }
        runs = ExperimentConfig(source).expand()
        assert len(runs) == 8
        assert source["model"]["params"]["dropout"] == 0.2
        assert {run["experiment"]["seed"] for run in runs} == {1, 2}
        assert {run["experiment"]["variant"] for run in runs} == {
            "width=8",
            "width=8__no_dropout",
            "width=16",
            "width=16__no_dropout",
        }

    def test_rejects_duplicate_materialized_runs(self) -> None:
        config = {"experiment": {"name": "demo", "seeds": [1, 1]}}
        with pytest.raises(ValueError, match="duplicate"):
            ExperimentConfig(config).expand()

    def test_object_factory_resolves_nested_target_and_ref(self) -> None:
        value = ObjectFactory.build(
            {
                "target": "types.SimpleNamespace",
                "params": {"factory": {"ref": "types.SimpleNamespace"}, "value": 3},
            }
        )
        assert isinstance(value, SimpleNamespace)
        assert value.factory is SimpleNamespace
        assert value.value == 3

    def test_parallel_execution_builds_fixed_gpu_slots(self) -> None:
        config = ExecutionConfig.from_mapping(
            {
                "experiment": {"name": "demo"},
                "execution": {"mode": "parallel", "gpus": [0, 1], "jobs_per_gpu": 2},
            }
        )
        assert config.mode is ExecutionMode.PARALLEL
        assert config.slots() == [[0], [0], [1], [1]]

    def test_rejects_duplicate_gpus(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            ExecutionConfig.from_mapping(
                {"experiment": {"name": "demo"}, "execution": {"mode": "parallel", "gpus": [0, 0]}}
            )
