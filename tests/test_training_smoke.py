"""End-to-end CPU Lightning smoke test."""

import pytest

from lambdaforge.experiments import ExperimentRunner
from lambdaforge.nn import MLP, BinaryCrossEntropyWithLogitsLoss
from lambdaforge.training import LightningRunner, LightningTask, LightningTrainConfig
from lambdaforge.training.data import LightningDataModule
from tests.fixtures.TinyMappingDataset import TinyMappingDataset


class TestTrainingSmoke:
    """Verify the default model-loss-task-runner integration for one epoch."""

    def test_one_cpu_epoch(self, tmp_path) -> None:
        dataset = TinyMappingDataset()
        datamodule = LightningDataModule(dataset, dataset, batch_size=4, num_workers=0)
        task = LightningTask(
            model=MLP(4, 1, hidden=[8]),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_kwargs={"lr": 0.01},
        )
        runner = LightningRunner(
            LightningTrainConfig(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                checkpoint_policy="none",
                default_root_dir=tmp_path,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                print_epoch_table=False,
            )
        )
        trainer = runner.fit(task, datamodule)
        assert trainer.current_epoch == 1
        assert (tmp_path / "metrics.csv").exists()

    def test_advanced_kwargs_are_forwarded_without_overriding_owned_fields(self, tmp_path) -> None:
        config = LightningTrainConfig(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            checkpoint_policy="none",
            default_root_dir=tmp_path,
            trainer_kwargs={"limit_train_batches": 1},
        )
        trainer = LightningRunner(config).build_trainer()
        assert trainer.limit_train_batches == 1

        config.trainer_kwargs = {"max_epochs": 99}
        with pytest.raises(ValueError, match="max_epochs"):
            LightningRunner(config).build_trainer()

        dataset = TinyMappingDataset()
        datamodule = LightningDataModule(
            dataset,
            batch_size=4,
            dataloader_kwargs={"timeout": 0},
        )
        assert datamodule.train_dataloader().timeout == 0
        with pytest.raises(ValueError, match="batch_size"):
            LightningDataModule(dataset, dataloader_kwargs={"batch_size": 99})

    def test_experiment_test_after_fit_uses_current_weights_without_checkpoint(
        self, tmp_path
    ) -> None:
        config = {
            "experiment": {
                "name": "test_after_fit",
                "output_root": str(tmp_path),
                "seed": 7,
                "variant": "base",
                "test_after_fit": True,
            },
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "test": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "datamodule": {"params": {"batch_size": 8, "num_workers": 0}},
            },
            "model": {
                "target": "lambdaforge.nn.models.MLP",
                "params": {"in_features": 4, "out_features": 1, "hidden": [4]},
            },
            "losses": [{"target": "lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss"}],
            "task": {"params": {"model_input_key": "x", "model_output_key": "logits"}},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "checkpoint_policy": "none",
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "print_epoch_table": False,
                "trainer_kwargs": {"enable_model_summary": False},
            },
        }
        result = ExperimentRunner().run_single_experiment(config)
        assert result["status"] == "ok"
        assert result["best_model_path"] is None
