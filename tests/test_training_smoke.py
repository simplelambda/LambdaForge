"""End-to-end CPU Lightning smoke test."""

import threading
from types import SimpleNamespace

import pytest
import torch

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

    def test_model_input_routing_supports_positional_and_named_arguments(self) -> None:
        batch = {
            "left": torch.randn(3, 2),
            "right": torch.randn(3, 2),
            "y": torch.zeros(3, 1),
        }

        positional = LightningTask(
            model=torch.nn.Bilinear(2, 2, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            model_input_keys=["left", "right"],
        )
        assert positional.forward_model(batch)["logits"].shape == (3, 1)

        named = LightningTask(
            model=torch.nn.Bilinear(2, 2, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            model_input_keys={"input1": "left", "input2": "right"},
        )
        assert named.forward_model(batch)["logits"].shape == (3, 1)

        with pytest.raises(ValueError, match="mutually exclusive"):
            LightningTask(
                model=torch.nn.Bilinear(2, 2, 1),
                losses=BinaryCrossEntropyWithLogitsLoss(),
                model_input_key="features",
                model_input_keys=["left", "right"],
            )

    def test_named_optimizer_group_overrides_are_applied(self) -> None:
        task = LightningTask(
            model=MLP(4, 1, hidden=[8]),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_kwargs={"lr": 0.01},
            optimizer_group_kwargs={"default": {"lr": 0.02}},
        )
        optimizer = task.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.02

        invalid = LightningTask(
            model=MLP(4, 1),
            losses=BinaryCrossEntropyWithLogitsLoss(),
            optimizer_group_kwargs={"missing": {"lr": 0.02}},
        )
        with pytest.raises(ValueError, match="Unknown optimizer"):
            invalid.configure_optimizers()

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

    def test_cooperative_stop_is_never_persisted_as_success(self, tmp_path) -> None:
        stop_event = threading.Event()
        stop_event.set()
        config = {
            "experiment": {
                "name": "interrupted_fit",
                "output_root": str(tmp_path),
                "seed": 7,
                "variant": "base",
            },
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "datamodule": {"params": {"batch_size": 8, "num_workers": 0}},
            },
            "model": {
                "target": "lambdaforge.nn.models.MLP",
                "params": {"in_features": 4, "out_features": 1, "hidden": [4]},
            },
            "losses": [{"target": "lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss"}],
            "task": {"params": {"model_input_key": "x", "model_output_key": "logits"}},
            "trainer": {
                "max_epochs": 3,
                "accelerator": "cpu",
                "devices": 1,
                "checkpoint_policy": "none",
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "print_epoch_table": False,
                "trainer_kwargs": {"enable_model_summary": False},
            },
        }

        result = ExperimentRunner().run_single_experiment(config, stop_event=stop_event)

        assert result["status"] == "interrupted"
        assert result["error"] == "Cooperative stop requested."
        persisted = (tmp_path / "interrupted_fit" / "base" / "seed=7" / "result.json").read_text(
            encoding="utf-8"
        )
        assert '"status": "interrupted"' in persisted

    def test_stop_requested_during_test_after_fit_is_persisted_as_interrupted(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stop_event = threading.Event()
        trainer = SimpleNamespace(checkpoint_callback=None, callback_metrics={})
        fake_runner = SimpleNamespace(
            fit=lambda **_kwargs: trainer,
            test=lambda **_kwargs: stop_event.set() or trainer,
        )
        task = SimpleNamespace(train_metrics=[], val_metrics=[], test_metrics=[])
        runner = ExperimentRunner()
        monkeypatch.setattr(runner, "_build_datamodule", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(runner, "_build_task", lambda *_args, **_kwargs: (task, []))
        monkeypatch.setattr(runner, "_build_runner", lambda *_args, **_kwargs: fake_runner)
        config = {
            "experiment": {
                "name": "interrupted_test",
                "output_root": str(tmp_path),
                "seed": 7,
                "variant": "base",
                "test_after_fit": True,
            },
            "model": {"target": "torch.nn.Identity", "params": {}},
        }

        result = runner.run_single_experiment(config, stop_event=stop_event)

        assert stop_event.is_set()
        assert result.status.value == "interrupted"
        assert result.error == "Cooperative stop requested."
        persisted = (tmp_path / "interrupted_test" / "base" / "seed=7" / "result.json").read_text(
            encoding="utf-8"
        )
        assert '"status": "interrupted"' in persisted
