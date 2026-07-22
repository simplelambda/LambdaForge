"""Opt-in end-to-end CUDA training smoke test."""

import json

import pytest
import torch

from lambdaforge.experiments import ExperimentRunner


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
class TestCudaSmoke:
    """Exercise the public YAML runner on one real CUDA training epoch."""

    def test_one_gpu_epoch_through_experiment_runner(self, tmp_path) -> None:
        config = {
            "experiment": {
                "name": "cuda_smoke",
                "output_root": str(tmp_path),
                "seed": 13,
                "variant": "base",
            },
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "datamodule": {
                    "params": {
                        "batch_size": 8,
                        "num_workers": 0,
                        "pin_memory": True,
                    }
                },
            },
            "model": {
                "target": "lambdaforge.nn.models.MLP",
                "params": {"in_features": 4, "out_features": 1, "hidden": [8]},
            },
            "losses": [{"target": "lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss"}],
            "task": {"params": {"model_input_key": "x", "model_output_key": "logits"}},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "gpu",
                "devices": 1,
                "precision": "32-true",
                "checkpoint_policy": "none",
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "print_epoch_table": False,
                "trainer_kwargs": {
                    "enable_model_summary": False,
                    "limit_train_batches": 1,
                    "limit_val_batches": 1,
                },
            },
        }

        try:
            result = ExperimentRunner().run_single_experiment(config)
            torch.cuda.synchronize()
        finally:
            torch.cuda.empty_cache()

        run_dir = tmp_path / "cuda_smoke" / "base" / "seed=13"
        environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))

        assert result["status"] == "ok"
        assert (run_dir / "metrics.csv").is_file()
        assert environment["torch"]["cuda_available"] is True
        assert environment["torch"]["device_count"] >= 1
        assert environment["torch"]["devices"][0]["total_memory_bytes"] > 0
