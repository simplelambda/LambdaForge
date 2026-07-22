"""Schema, semantic and CLI validation without experiment side effects."""

import copy

import yaml
from jsonschema import Draft202012Validator

from lambdaforge import LambdaForge
from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments import ExperimentValidator
from lambdaforge.plugins import PluginKind


class TestExperimentValidation:
    """Verify useful validation reports and non-mutating CLI behavior."""

    @staticmethod
    def valid_config() -> dict:
        """Return a minimal runnable configuration using test-owned extensions."""
        return {
            "schema_version": "1.0",
            "experiment": {"name": "validated", "seeds": [3, 5]},
            "data": {
                "train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
                "val": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"},
            },
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "val_metrics": [
                {
                    "target": "tests.fixtures.UserMetric.UserMetric",
                    "params": {"name": "score"},
                }
            ],
            "optimizer": {"ref": "torch.optim.AdamW", "params": {"lr": 0.001}},
            "trainer": {"max_epochs": 1, "checkpoint_policy": "last"},
            "execution": {"mode": "sequential"},
        }

    def test_packaged_schema_is_valid_and_accepts_a_real_config(self) -> None:
        validator = ExperimentValidator()
        Draft202012Validator.check_schema(validator.schema())
        report = validator.validate(self.valid_config())
        assert report.is_valid
        assert report.expanded_runs == 2
        assert report.imports_checked

    def test_plugin_kind_enum_cannot_drift_from_the_packaged_schema(self) -> None:
        schema = ExperimentValidator().schema()
        schema_kinds = set(schema["$defs"]["pluginReference"]["properties"]["kind"]["enum"])

        assert schema_kinds == {kind.value for kind in PluginKind}
        assert PluginKind.DATASET.entry_point_group == "lambdaforge.datasets"
        assert PluginKind.CALLBACK.entry_point_group == "lambdaforge.callbacks"
        assert PluginKind.LOGGER.entry_point_group == "lambdaforge.loggers"

    def test_logger_schema_preserves_modes_references_and_sequences(self) -> None:
        logger_target = {
            "target": "tests.fixtures.UserLogger.UserLogger",
            "params": {"output_path": "metrics.jsonl"},
        }
        logger_reference = {"ref": "tests.fixtures.UserLogger.UserLogger"}

        for logger in (
            "none",
            False,
            None,
            logger_target,
            logger_reference,
            [logger_target, logger_reference],
        ):
            config = self.valid_config()
            config["trainer"]["logger"] = logger
            assert ExperimentValidator().validate(config).is_valid

    def test_unknown_keys_and_bad_imports_are_reported_together(self) -> None:
        config = copy.deepcopy(self.valid_config())
        config["trianer"] = {}
        config["model"]["target"] = "missing_package.models.UnknownModel"
        report = ExperimentValidator().validate(config)
        assert not report.is_valid
        assert any("trianer" in error for error in report.errors)
        assert any("missing_package" in error for error in report.errors)

    def test_import_checks_can_be_disabled_for_templates(self) -> None:
        config = copy.deepcopy(self.valid_config())
        config["model"]["target"] = "your_project.models.ProjectModel"
        report = ExperimentValidator().validate(config, check_imports=False)
        assert report.is_valid
        assert not report.imports_checked
        assert report.warnings

    def test_schema_accepts_recursive_hardened_cache_objects(self, tmp_path) -> None:
        config = self.valid_config()
        config["data"]["train"] = {
            "target": "lambdaforge.data.DatasetCache",
            "params": {
                "dataset": {
                    "target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset",
                    "params": {"size": 4},
                },
                "max_memory_bytes_per_process": 0,
                "backend": {
                    "target": "lambdaforge.data.MemoryMappedCacheBackend",
                    "params": {
                        "root": str(tmp_path),
                        "namespace": "schema-cache-v1",
                        "max_bytes": 1_000_000,
                        "record_codec": {
                            "target": "lambdaforge.data.CacheRecordCodec",
                            "params": {"integrity": "checksum_sha256"},
                        },
                    },
                },
                "serializer": {
                    "target": "lambdaforge.data.NumpyDatasetSerializer",
                    "params": {"compressed": False},
                },
                "fingerprint": {
                    "target": "lambdaforge.data.DatasetFingerprint",
                    "params": {
                        "content": "sha256:fixture",
                        "transform": "identity-v1",
                        "configuration": {"size": 4},
                    },
                },
            },
        }

        report = ExperimentValidator().validate(config)
        assert report.is_valid
        assert report.imports_checked

    def test_materialized_sweep_keys_and_resource_rules_are_validated(self) -> None:
        config = self.valid_config()
        config["sweep"] = {"grid": {"trainer.max_epocs": [2]}}
        config["execution"] = {"mode": "ddp", "gpus": [], "devices_per_job": 2}
        report = ExperimentValidator().validate(config)
        assert not report.is_valid
        assert any("max_epocs" in error for error in report.errors)
        assert any("requires a non-empty gpus" in error for error in report.errors)

    def test_cli_validate_does_not_create_run_directories(self, tmp_path, capsys) -> None:
        config = self.valid_config()
        output_root = tmp_path / "runs"
        config["experiment"]["output_root"] = str(output_root)
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        assert LambdaForge.validate(path).is_valid

        exit_code = CommandLineInterface.main(["validate", str(path), "--json"])
        output = capsys.readouterr().out
        assert exit_code == 0
        assert '"valid": true' in output
        assert not output_root.exists()
