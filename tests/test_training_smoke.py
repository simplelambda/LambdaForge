"""End-to-end CPU Lightning smoke test."""


import pytest
import torch

from lambdaforge.nn import MLP, BinaryCrossEntropyWithLogitsLoss
from lambdaforge.training import LightningModule, LightningRunner, LightningTrainConfig
from lambdaforge.training.data import LightningDataModule
from tests.fixtures.TinyMappingDataset import TinyMappingDataset


class TestTrainingSmoke:
    """Verify the default model-loss-task-runner integration for one epoch."""

    def test_one_cpu_epoch(self, tmp_path) -> None:
        dataset = TinyMappingDataset()
        datamodule = LightningDataModule(dataset, dataset, batch_size=4, num_workers=0)
        task = LightningModule(
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

    def test_model_input_routing_supports_positional_and_named_arguments(self) -> None:
        batch = {
            "left": torch.randn(3, 2),
            "right": torch.randn(3, 2),
            "y": torch.zeros(3, 1),
        }

        positional = LightningModule(
            model=torch.nn.Bilinear(2, 2, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            model_input_keys=["left", "right"],
        )
        assert positional.forward_model(batch)["logits"].shape == (3, 1)

        named = LightningModule(
            model=torch.nn.Bilinear(2, 2, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            model_input_keys={"input1": "left", "input2": "right"},
        )
        assert named.forward_model(batch)["logits"].shape == (3, 1)

        with pytest.raises(ValueError, match="mutually exclusive"):
            LightningModule(
                model=torch.nn.Bilinear(2, 2, 1),
                losses=BinaryCrossEntropyWithLogitsLoss(),
                model_input_key="features",
                model_input_keys=["left", "right"],
            )

    def test_named_optimizer_group_overrides_are_applied(self) -> None:
        task = LightningModule(
            model=MLP(4, 1, hidden=[8]),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_kwargs={"lr": 0.01},
            optimizer_group_kwargs={"default": {"lr": 0.02}},
        )
        optimizer = task.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.02

        invalid = LightningModule(
            model=MLP(4, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_group_kwargs={"missing": {"lr": 0.02}},
        )
        with pytest.raises(ValueError, match="Unknown optimizer"):
            invalid.configure_optimizers()
