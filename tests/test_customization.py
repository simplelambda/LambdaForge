"""End-to-end YAML customization and public extension contract tests."""

import json

from lambdaforge.experiments import ExperimentRunner, ObjectFactory
from lambdaforge.metrics import MetricAlias
from lambdaforge.training.callbacks import LogKeyFilter


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
        assert marker.read_text(encoding="utf-8") == "callback invoked"
        records = [
            json.loads(line) for line in logger_output.read_text(encoding="utf-8").splitlines()
        ]
        logged_keys = {key for record in records for key in record.get("metrics", {}).keys()}
        assert "train_train_score" in logged_keys
        assert "val_validation_score" in logged_keys
        assert "train_loss_user_bce" in logged_keys
