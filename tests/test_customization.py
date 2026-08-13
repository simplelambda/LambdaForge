"""End-to-end YAML customization and public extension contract tests."""

import csv
import importlib.metadata as metadata
import json
from types import SimpleNamespace

from lambdaforge.experiments import ExperimentRunner, ObjectFactory
from lambdaforge.hpo.AdaptiveObservationReader import AdaptiveObservationReader
from lambdaforge.metrics import MetricAlias
from lambdaforge.plugins import PluginKind, PluginRegistry
from lambdaforge.training.callbacks import LogKeyFilter
from tests.fixtures.UserCallback import UserCallback


class TestCustomization:
    """Verify that external objects and logging choices require no framework edits."""

    def test_metric_alias_supports_duplicate_metric_types(self) -> None:
        first = ObjectFactory.build(
            {
                "target": "lambdaforge.metrics.MetricAlias",
                "params": {
                    "name": "accuracy_at_08",
                    "metric": {
                        "target": "lambdaforge.metrics.classification.BinaryAccuracy",
                        "params": {"threshold": 0.8},
                    },
                },
            }
        )
        assert isinstance(first, MetricAlias)
        assert first.name == "accuracy_at_08"

    def test_log_key_filter_uses_include_and_exclude_patterns(self) -> None:
        selector = LogKeyFilter(include=["val_*", "epoch_time_s"], exclude=["*_loss_*"])
        assert selector.accepts("val_accuracy")
        assert selector.accepts("epoch_time_s")
        assert not selector.accepts("val_loss_aux")
        assert not selector.accepts("train_accuracy")

    def test_project_callback_artifact_is_rank_zero_only(self, tmp_path) -> None:
        marker = tmp_path / "rank-zero.txt"
        callback = UserCallback(str(marker))
        callback.on_fit_end(SimpleNamespace(is_global_zero=False), object())
        assert not marker.exists()
        callback.on_fit_end(SimpleNamespace(is_global_zero=True), object())
        assert marker.read_text(encoding="utf-8") == "callback invoked"

    def test_validation_callback_reuses_outputs_and_publishes_hpo_metric(self, tmp_path) -> None:
        marker = tmp_path / "auxiliary.txt"
        config = {
            "experiment": {
                "name": "callback_outputs",
                "output_root": str(tmp_path / "runs"),
                "seed": 5,
                "variant": "base",
            },
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "datamodule": {"params": {"batch_size": 8, "num_workers": 0}},
            },
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "task": {"params": {"model_input_key": "x"}},
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "checkpoint_policy": "none",
                "print_epoch_table": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "trainer_kwargs": {"enable_model_summary": False},
            },
            "callbacks": [
                {
                    "target": (
                        "tests.fixtures.ValidationAuxiliaryCallback.ValidationAuxiliaryCallback"
                    ),
                    "params": {"artifact_path": str(marker)},
                }
            ],
        }

        result = ExperimentRunner().run_single_experiment(config)
        metrics_path = tmp_path / "runs" / "callback_outputs" / "base" / "seed=5" / "metrics.csv"
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))

        assert result.status.value == "ok"
        assert marker.read_text(encoding="utf-8") == "rank-zero"
        assert rows and rows[-1]["val_auxiliary_score"]
        assert AdaptiveObservationReader._curve(metrics_path, "val_auxiliary_score")

    def test_external_model_loss_metric_logger_and_callback_from_yaml(self, tmp_path) -> None:
        marker = tmp_path / "callback.txt"
        logger_output = tmp_path / "custom_metrics.jsonl"
        config = {
            "experiment": {
                "name": "custom_objects",
                "output_root": str(tmp_path / "runs"),
                "seed": 11,
                "variant": "base",
            },
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "datamodule": {"params": {"batch_size": 8, "num_workers": 0}},
            },
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "train_metrics": [
                {
                    "target": "tests.fixtures.UserMetric.UserMetric",
                    "params": {"name": "train_score"},
                }
            ],
            "val_metrics": [
                {
                    "target": "tests.fixtures.UserMetric.UserMetric",
                    "params": {"name": "validation_score"},
                }
            ],
            "task": {
                "params": {
                    "model_input_key": "x",
                    "logging": {
                        "loss_prog_bar": False,
                        "metric_prog_bar": False,
                        "log_individual_losses": True,
                        "logger": True,
                    },
                }
            },
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "checkpoint_policy": "best",
                "early_stopping_patience": None,
                "logger": {
                    "target": "tests.fixtures.UserLogger.UserLogger",
                    "params": {"output_path": str(logger_output)},
                },
                "track_epoch_stats": False,
                "print_epoch_table": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "trainer_kwargs": {"enable_model_summary": False},
            },
            "callbacks": [
                {
                    "target": "tests.fixtures.UserCallback.UserCallback",
                    "params": {"marker_path": str(marker)},
                }
            ],
        }

        result = ExperimentRunner().run_single_experiment(config)
        assert result["status"] == "ok"
        assert result["best_metric"]["monitor"] == "val_validation_score"
        assert result["best_model_path"]
        assert (tmp_path / "runs" / "custom_objects" / "base" / "seed=11" / "metrics.csv").exists()
        assert (
            tmp_path / "runs" / "custom_objects" / "base" / "seed=11" / "environment.json"
        ).exists()
        assert marker.read_text(encoding="utf-8") == "callback invoked"
        records = [
            json.loads(line) for line in logger_output.read_text(encoding="utf-8").splitlines()
        ]
        logged_keys = {key for record in records for key in record.get("metrics", {}).keys()}
        assert "train_train_score" in logged_keys
        assert "val_validation_score" in logged_keys
        assert "train_loss_user_bce" in logged_keys

    def test_dataset_callback_and_logger_plugins_run_end_to_end(
        self, tmp_path, monkeypatch
    ) -> None:
        marker = tmp_path / "plugin_callback.txt"
        logger_output = tmp_path / "plugin_metrics.jsonl"
        distribution = SimpleNamespace(
            metadata={"Name": "lambda-research-extensions"},
            version="1.7.0",
        )
        entries = metadata.EntryPoints(
            [
                metadata.EntryPoint(
                    name="tiny_dataset",
                    value="tests.fixtures.TinyMappingDataset:TinyMappingDataset",
                    group=PluginKind.DATASET.entry_point_group,
                )._for(distribution),
                metadata.EntryPoint(
                    name="marker_callback",
                    value="tests.fixtures.UserCallback:UserCallback",
                    group=PluginKind.CALLBACK.entry_point_group,
                )._for(distribution),
                metadata.EntryPoint(
                    name="json_logger",
                    value="tests.fixtures.UserLogger:UserLogger",
                    group=PluginKind.LOGGER.entry_point_group,
                )._for(distribution),
            ]
        )

        def selected_entry_points(**selection):
            return entries.select(**selection)

        monkeypatch.setattr(metadata, "entry_points", selected_entry_points)
        monkeypatch.setattr(PluginRegistry, "_default", PluginRegistry())
        config = {
            "experiment": {
                "name": "non_neural_plugins",
                "output_root": str(tmp_path / "runs"),
                "seed": 13,
                "variant": "base",
            },
            "data": {
                "train": {
                    "plugin": {"kind": "dataset", "name": "tiny_dataset"},
                    "params": {"size": 8},
                },
                "val": {
                    "plugin": {"kind": "dataset", "name": "tiny_dataset"},
                    "params": {"size": 4},
                },
                "datamodule": {"params": {"batch_size": 4, "num_workers": 0}},
            },
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "task": {
                "params": {
                    "model_input_key": "x",
                    "logging": {"logger": True},
                }
            },
            "trainer": {
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "checkpoint_policy": "none",
                "logger": {
                    "plugin": {"kind": "logger", "name": "json_logger"},
                    "params": {"output_path": str(logger_output)},
                },
                "track_epoch_stats": False,
                "print_epoch_table": False,
                "enable_progress_bar": False,
                "num_sanity_val_steps": 0,
                "trainer_kwargs": {"enable_model_summary": False},
            },
            "callbacks": [
                {
                    "plugin": {"kind": "callback", "name": "marker_callback"},
                    "params": {"marker_path": str(marker)},
                }
            ],
        }

        result = ExperimentRunner().run_single_experiment(config)
        run_dir = tmp_path / "runs" / "non_neural_plugins" / "base" / "seed=13"
        plugin_names = [
            item["name"]
            for item in json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))[
                "plugins"
            ]
        ]

        assert result.status.value == "ok"
        assert marker.read_text(encoding="utf-8") == "callback invoked"
        assert logger_output.exists()
        assert plugin_names == ["marker_callback", "tiny_dataset", "json_logger"]
