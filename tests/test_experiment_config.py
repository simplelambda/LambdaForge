"""Experiment configuration and object-factory contracts."""

from types import SimpleNamespace
from typing import Any

import pytest

from lambdaforge.experiments import ExecutionConfig, ExecutionMode, ExperimentConfig, ObjectFactory


class TestExperimentConfig:
    """Verify deterministic YAML-style expansion and resource validation."""

    def test_expands_grid_ablations_and_seeds_without_mutation(self) -> None:
        source: dict[str, Any] = {
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

    def test_execution_config_preserves_supported_integer_strings(self) -> None:
        config = ExecutionConfig.from_mapping(
            {
                "execution": {
                    "mode": "parallel",
                    "gpus": "0,1",
                    "jobs_per_gpu": "2",
                    "devices_per_job": "1",
                    "grace_seconds": 3,
                    "cpu_threads_per_job": "2",
                    "cpu_interop_threads_per_job": "1",
                    "cpu_cores_per_job": "2",
                    "dataloader_num_workers_per_job": "0",
                }
            }
        )

        assert config.gpus == [0, 1]
        assert config.jobs_per_gpu == 2
        assert config.devices_per_job == 1
        assert config.grace_seconds == 3.0
        assert config.cpu_threads_per_job == 2
        assert config.dataloader_num_workers_per_job == 0

    @pytest.mark.parametrize(
        "execution, expected_error",
        [
            ({"gpus": [True]}, TypeError),
            ({"gpus": [0.5]}, TypeError),
            ({"gpus": 0}, TypeError),
            ({"grace_seconds": True}, TypeError),
            ({"grace_seconds": "1"}, TypeError),
            ({"grace_seconds": float("nan")}, ValueError),
            ({"grace_seconds": float("inf")}, ValueError),
            ({"jobs_per_gpu": True}, TypeError),
            ({"jobs_per_gpu": 1.5}, ValueError),
            ({"devices_per_job": float("inf")}, ValueError),
            ({"cpu_threads_per_job": True}, TypeError),
            ({"cpu_interop_threads_per_job": 1.5}, ValueError),
            ({"cpu_cores_per_job": [1]}, TypeError),
            ({"dataloader_num_workers_per_job": float("nan")}, ValueError),
            ({"cpu_threads_per_job": 0}, ValueError),
        ],
    )
    def test_execution_mapping_rejects_unsafe_numeric_coercions(
        self,
        execution,
        expected_error,
    ) -> None:
        with pytest.raises(expected_error):
            ExecutionConfig.from_mapping({"execution": execution})

    @pytest.mark.parametrize(
        "field, value, expected_error",
        [
            ("grace_seconds", float("-inf"), ValueError),
            ("jobs_per_gpu", True, TypeError),
            ("devices_per_job", 1.5, ValueError),
            ("cpu_threads_per_job", 1.25, ValueError),
            ("gpus", [True], TypeError),
        ],
    )
    def test_direct_execution_config_validation_is_equally_strict(
        self,
        field,
        value,
        expected_error,
    ) -> None:
        config = ExecutionConfig()
        setattr(config, field, value)
        with pytest.raises(expected_error):
            config.validate()
