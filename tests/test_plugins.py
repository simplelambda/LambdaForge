"""Entry-point plugin discovery, safety and YAML integration tests."""

import importlib
import importlib.metadata as metadata
import json
import multiprocessing
from types import SimpleNamespace

import pytest
from torch import nn
from torch.utils.data import Dataset

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments import ExperimentValidator, ObjectFactory
from lambdaforge.integrations.Lightning import CallbackBase, LoggerType
from lambdaforge.nn import ComponentRegistry
from lambdaforge.nn.activations import GELU, ReLU
from lambdaforge.plugins import (
    PluginKind,
    PluginReference,
    PluginRegistry,
    PluginResolutionError,
)
from tests.fixtures.PluginUsageSpawnJob import PluginUsageSpawnJob


class TestPluginDiscovery:
    """Verify lazy, unambiguous and contract-checked plugin resolution."""

    @staticmethod
    def _entry_point(
        name: str,
        value: str,
        kind: PluginKind,
        *,
        distribution: str | None = None,
        version: str | None = None,
    ) -> metadata.EntryPoint:
        """Build standard-library metadata without installing a test distribution."""
        entry = metadata.EntryPoint(name=name, value=value, group=kind.entry_point_group)
        if distribution is None:
            return entry
        fake_distribution = SimpleNamespace(
            metadata={"Name": distribution},
            version=version,
        )
        return entry._for(fake_distribution)

    @staticmethod
    def _publish(monkeypatch, entries: list[metadata.EntryPoint]) -> list[str]:
        """Expose deterministic entry points and return the queried groups."""
        calls: list[str] = []

        def selected_entry_points(**selection):
            group = str(selection["group"])
            calls.append(group)
            return metadata.EntryPoints(entries).select(**selection)

        monkeypatch.setattr(metadata, "entry_points", selected_entry_points)
        return calls

    def test_discovery_reads_metadata_without_loading_modules(self, monkeypatch) -> None:
        entry = self._entry_point(
            "unimportable",
            "package_that_must_not_be_imported.models:Missing",
            PluginKind.MODEL,
        )
        calls = self._publish(monkeypatch, [entry])
        registry = PluginRegistry()

        descriptors = registry.discover(PluginKind.MODEL)

        assert len(descriptors) == 1
        assert descriptors[0].name == "unimportable"
        assert descriptors[0].value == entry.value
        assert calls == ["lambdaforge.models"]
        assert registry.resolved_plugins() == ()

    def test_usage_session_records_exact_successful_provenance_once(self, monkeypatch) -> None:
        entry = self._entry_point(
            "user_model",
            "tests.fixtures.UserModel:UserModel",
            PluginKind.MODEL,
            distribution="lambda-acme",
            version="2.4.1",
        )
        self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        reference = PluginReference(PluginKind.MODEL, "user_model")

        with registry.usage_session() as usage:
            assert registry.resolve(reference) is registry.resolve(reference)

        assert [descriptor.to_dict() for descriptor in usage.descriptors()] == [
            {
                "kind": "model",
                "name": "user_model",
                "group": "lambdaforge.models",
                "value": "tests.fixtures.UserModel:UserModel",
                "distribution": "lambda-acme",
                "version": "2.4.1",
            }
        ]
        assert registry.resolved_plugins() == usage.descriptors()

    def test_usage_sessions_are_isolated_from_validation_and_prior_runs(self, monkeypatch) -> None:
        entries = [
            self._entry_point(
                "first",
                "tests.fixtures.UserModel:UserModel",
                PluginKind.MODEL,
            ),
            self._entry_point(
                "second",
                "tests.fixtures.UserModel:UserModel",
                PluginKind.MODEL,
            ),
        ]
        self._publish(monkeypatch, entries)
        registry = PluginRegistry()
        first = PluginReference(PluginKind.MODEL, "first")
        second = PluginReference(PluginKind.MODEL, "second")

        registry.resolve(first)
        with registry.usage_session() as validation_usage:
            registry.resolve(second, record_usage=False)
        with registry.usage_session() as first_run:
            registry.resolve(second)
            registry.resolve(second)
        with registry.usage_session() as second_run:
            pass

        assert validation_usage.descriptors() == ()
        assert [item.name for item in first_run.descriptors()] == ["second"]
        assert second_run.descriptors() == ()
        assert [item.name for item in registry.resolved_plugins()] == ["first", "second"]

    def test_usage_session_captures_component_alias_fallback(self, monkeypatch) -> None:
        entry = self._entry_point(
            "externalgelu",
            "lambdaforge.nn.activations.GELU:GELU",
            PluginKind.ACTIVATION,
        )
        self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        monkeypatch.setattr(PluginRegistry, "_default", registry)

        with registry.usage_session() as usage:
            assert ComponentRegistry.resolve_activation("external-gelu") is GELU

        assert [(item.kind, item.name) for item in usage.descriptors()] == [
            (PluginKind.ACTIVATION, "externalgelu")
        ]

    def test_cli_lists_filtered_metadata_as_json_without_loading(self, monkeypatch, capsys) -> None:
        entry = self._entry_point(
            "cli_model",
            "package_that_must_not_be_imported.models:Missing",
            PluginKind.MODEL,
        )
        calls = self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        monkeypatch.setattr(PluginRegistry, "_default", registry)

        exit_code = CommandLineInterface.main(["plugins", "--kind", "model", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload == [
            {
                "kind": "model",
                "name": "cli_model",
                "group": "lambdaforge.models",
                "value": entry.value,
                "distribution": None,
                "version": None,
            }
        ]
        assert calls == ["lambdaforge.models"]

    def test_cli_discovers_dataset_plugins_without_loading_them(self, monkeypatch, capsys) -> None:
        entry = self._entry_point(
            "remote_records",
            "package_that_must_not_be_imported.datasets:Records",
            PluginKind.DATASET,
        )
        calls = self._publish(monkeypatch, [entry])
        monkeypatch.setattr(PluginRegistry, "_default", PluginRegistry())

        exit_code = CommandLineInterface.main(["plugins", "--kind", "dataset", "--json"])
        payload = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert payload[0]["kind"] == "dataset"
        assert payload[0]["group"] == "lambdaforge.datasets"
        assert calls == ["lambdaforge.datasets"]

    def test_object_factory_builds_fresh_model_instances_from_plugin(self, monkeypatch) -> None:
        entry = self._entry_point(
            "user_model",
            "tests.fixtures.UserModel:UserModel",
            PluginKind.MODEL,
        )
        self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        spec = {
            "plugin": {"kind": "model", "name": "user_model"},
            "params": {"in_features": 7},
        }

        first = ObjectFactory.build(spec, plugins=registry)
        second = ObjectFactory.build(spec, plugins=registry)

        assert isinstance(first, nn.Module)
        assert first.projection.in_features == 7
        assert first is not second
        assert registry.resolve(PluginReference(PluginKind.MODEL, "user_model")) is type(first)

    def test_metric_plugin_is_built_inside_recursive_containers(self, monkeypatch) -> None:
        entry = self._entry_point(
            "user_metric",
            "tests.fixtures.UserMetric:UserMetric",
            PluginKind.METRIC,
        )
        self._publish(monkeypatch, [entry])
        registry = PluginRegistry()

        result = ObjectFactory.build(
            [
                {
                    "plugin": {"kind": "metric", "name": "user_metric"},
                    "params": {"name": "plugin_score"},
                }
            ],
            plugins=registry,
        )

        assert result[0].name == "plugin_score"

    def test_non_neural_plugins_build_datasets_callbacks_and_loggers(
        self, monkeypatch, tmp_path
    ) -> None:
        entries = [
            self._entry_point(
                "tiny_dataset",
                "tests.fixtures.TinyMappingDataset:TinyMappingDataset",
                PluginKind.DATASET,
            ),
            self._entry_point(
                "marker_callback",
                "tests.fixtures.UserCallback:UserCallback",
                PluginKind.CALLBACK,
            ),
            self._entry_point(
                "json_logger",
                "tests.fixtures.UserLogger:UserLogger",
                PluginKind.LOGGER,
            ),
        ]
        self._publish(monkeypatch, entries)
        registry = PluginRegistry()

        dataset = ObjectFactory.build(
            {
                "plugin": {"kind": "dataset", "name": "tiny_dataset"},
                "params": {"size": 3},
            },
            plugins=registry,
        )
        callback = ObjectFactory.build(
            {
                "plugin": {"kind": "callback", "name": "marker_callback"},
                "params": {"marker_path": str(tmp_path / "marker.txt")},
            },
            plugins=registry,
        )
        logger = ObjectFactory.build(
            {
                "plugin": {"kind": "logger", "name": "json_logger"},
                "params": {"output_path": str(tmp_path / "metrics.jsonl")},
            },
            plugins=registry,
        )

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3
        assert isinstance(callback, CallbackBase)
        assert isinstance(logger, LoggerType)

    def test_non_neural_plugin_contracts_and_schema_positions_are_strict(self, monkeypatch) -> None:
        entries = [
            self._entry_point(
                "tiny_dataset",
                "tests.fixtures.TinyMappingDataset:TinyMappingDataset",
                PluginKind.DATASET,
            ),
            self._entry_point(
                "marker_callback",
                "tests.fixtures.UserCallback:UserCallback",
                PluginKind.CALLBACK,
            ),
            self._entry_point(
                "json_logger",
                "tests.fixtures.UserLogger:UserLogger",
                PluginKind.LOGGER,
            ),
            self._entry_point(
                "wrong_dataset",
                "tests.fixtures.UserCallback:UserCallback",
                PluginKind.DATASET,
            ),
            self._entry_point(
                "wrong_callback",
                "tests.fixtures.UserModel:UserModel",
                PluginKind.CALLBACK,
            ),
            self._entry_point(
                "wrong_logger",
                "tests.fixtures.UserCallback:UserCallback",
                PluginKind.LOGGER,
            ),
        ]
        self._publish(monkeypatch, entries)
        registry = PluginRegistry()
        config = {
            "schema_version": "1.0",
            "experiment": {"name": "non_neural_plugins"},
            "data": {
                "train": {
                    "plugin": {"kind": "dataset", "name": "tiny_dataset"},
                    "params": {"size": 4},
                }
            },
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "trainer": {
                "logger": {
                    "plugin": {"kind": "logger", "name": "json_logger"},
                    "params": {"output_path": "metrics.jsonl"},
                }
            },
            "callbacks": [
                {
                    "plugin": {"kind": "callback", "name": "marker_callback"},
                    "params": {"marker_path": "marker.txt"},
                }
            ],
        }

        assert ExperimentValidator(plugins=registry).validate(config).is_valid
        with pytest.raises(PluginResolutionError, match="must subclass"):
            registry.resolve(PluginReference(PluginKind.DATASET, "wrong_dataset"))
        with pytest.raises(PluginResolutionError, match="must subclass"):
            registry.resolve(PluginReference(PluginKind.CALLBACK, "wrong_callback"))
        with pytest.raises(PluginResolutionError, match="must subclass"):
            registry.resolve(PluginReference(PluginKind.LOGGER, "wrong_logger"))

        wrong_position = {
            **config,
            "data": {
                "train": {
                    "plugin": {"kind": "callback", "name": "marker_callback"},
                }
            },
        }
        report = ExperimentValidator(plugins=registry).validate(
            wrong_position,
            check_imports=False,
        )
        assert not report.is_valid
        assert any("data.train" in error for error in report.errors)

        wrong_callback_position = {
            **config,
            "callbacks": [
                {
                    "plugin": {"kind": "logger", "name": "json_logger"},
                }
            ],
        }
        callback_report = ExperimentValidator(plugins=registry).validate(
            wrong_callback_position,
            check_imports=False,
        )
        assert not callback_report.is_valid
        assert any("callbacks" in error for error in callback_report.errors)

        wrong_logger_position = {
            **config,
            "trainer": {
                "logger": {
                    "plugin": {"kind": "callback", "name": "marker_callback"},
                }
            },
        }
        logger_report = ExperimentValidator(plugins=registry).validate(
            wrong_logger_position,
            check_imports=False,
        )
        assert not logger_report.is_valid
        assert any("trainer.logger" in error for error in logger_report.errors)

    def test_loss_plugins_are_contract_checked_and_valid_in_experiments(self, monkeypatch) -> None:
        entry = self._entry_point(
            "user_loss",
            "tests.fixtures.UserLoss:UserLoss",
            PluginKind.LOSS,
        )
        calls = self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        config = {
            "schema_version": "1.0",
            "experiment": {"name": "loss_plugin"},
            "data": {},
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [
                {
                    "plugin": {"kind": "loss", "name": "user_loss"},
                    "params": {"name": "external_loss"},
                }
            ],
        }

        report = ExperimentValidator(plugins=registry).validate(config)
        built = ObjectFactory.build(config["losses"][0], plugins=registry)

        assert report.is_valid
        assert built.name == "external_loss"
        assert calls == ["lambdaforge.losses"]

    def test_missing_conflicting_and_invalid_contracts_are_rejected(self, monkeypatch) -> None:
        entries = [
            self._entry_point("duplicate", "tests.fixtures.UserModel:UserModel", PluginKind.MODEL),
            self._entry_point("duplicate", "torch.nn:Linear", PluginKind.MODEL),
            self._entry_point(
                "wrong_metric", "tests.fixtures.UserModel:UserModel", PluginKind.METRIC
            ),
            self._entry_point("not_a_class", "tests.fixtures.UserModel:torch", PluginKind.MODEL),
        ]
        self._publish(monkeypatch, entries)
        registry = PluginRegistry()

        with pytest.raises(PluginResolutionError, match="No plugin 'missing'"):
            registry.resolve(PluginReference(PluginKind.MODEL, "missing"))
        with pytest.raises(PluginResolutionError, match="ambiguous"):
            registry.resolve(PluginReference(PluginKind.MODEL, "duplicate"))
        with pytest.raises(PluginResolutionError, match="must subclass"):
            registry.resolve(PluginReference(PluginKind.METRIC, "wrong_metric"))
        with pytest.raises(PluginResolutionError, match="must expose a class"):
            registry.resolve(PluginReference(PluginKind.MODEL, "not_a_class"))
        assert registry.resolved_plugins() == ()

    def test_load_errors_preserve_the_original_cause(self, monkeypatch) -> None:
        entry = self._entry_point(
            "broken", "package_that_does_not_exist.models:Missing", PluginKind.MODEL
        )
        self._publish(monkeypatch, [entry])

        with pytest.raises(PluginResolutionError, match="Could not load") as captured:
            PluginRegistry().resolve(PluginReference(PluginKind.MODEL, "broken"))

        assert isinstance(captured.value.__cause__, ModuleNotFoundError)

    def test_resolution_cache_and_explicit_refresh(self, monkeypatch) -> None:
        entry = self._entry_point(
            "user_model", "tests.fixtures.UserModel:UserModel", PluginKind.MODEL
        )
        calls = self._publish(monkeypatch, [entry])
        registry = PluginRegistry()
        reference = PluginReference(PluginKind.MODEL, "user_model")

        assert registry.resolve(reference) is registry.resolve(reference)
        assert calls == ["lambdaforge.models"]
        registry.refresh(PluginKind.MODEL)
        registry.resolve(reference)
        assert calls == ["lambdaforge.models", "lambdaforge.models"]

    def test_default_registry_is_recreated_after_process_id_change(self, monkeypatch) -> None:
        module = importlib.import_module("lambdaforge.plugins.PluginRegistry")
        first = PluginRegistry.default()
        process_id = first._process_id
        monkeypatch.setattr(module.os, "getpid", lambda: process_id + 1)

        second = PluginRegistry.default()

        assert second is not first
        assert second._process_id == process_id + 1

    def test_spawn_child_captures_only_its_process_local_plugin(
        self, tmp_path, monkeypatch
    ) -> None:
        dist_info = tmp_path / "spawn_lf_plugin-9.1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: spawn-lf-plugin\nVersion: 9.1.0\n",
            encoding="utf-8",
        )
        (dist_info / "entry_points.txt").write_text(
            "[lambdaforge.models]\nspawn_model = tests.fixtures.UserModel:UserModel\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        output_path = tmp_path / "spawn-environment.json"
        process = multiprocessing.get_context("spawn").Process(
            target=PluginUsageSpawnJob.run,
            args=(str(output_path),),
        )

        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

        assert process.exitcode == 0
        plugins = json.loads(output_path.read_text(encoding="utf-8"))["plugins"]
        assert plugins == [
            {
                "kind": "model",
                "name": "spawn_model",
                "group": "lambdaforge.models",
                "value": "tests.fixtures.UserModel:UserModel",
                "distribution": "spawn-lf-plugin",
                "version": "9.1.0",
            }
        ]

    def test_component_alias_plugins_are_fallbacks_and_cannot_shadow_builtins(
        self, monkeypatch
    ) -> None:
        entries = [
            self._entry_point(
                "externalgelu",
                "lambdaforge.nn.activations.GELU:GELU",
                PluginKind.ACTIVATION,
            ),
            self._entry_point(
                "relu", "package_that_must_not_be_imported:Missing", PluginKind.ACTIVATION
            ),
        ]
        calls = self._publish(monkeypatch, entries)
        registry = PluginRegistry()
        monkeypatch.setattr(PluginRegistry, "_default", registry)

        assert ComponentRegistry.resolve_activation("relu") is ReLU
        assert calls == []
        assert ComponentRegistry.resolve_activation("external-gelu") is GELU
        assert calls == ["lambdaforge.activations"]

    def test_validator_accepts_plugins_and_no_imports_mode_skips_resolution(
        self, monkeypatch
    ) -> None:
        entries = [
            self._entry_point("user_model", "tests.fixtures.UserModel:UserModel", PluginKind.MODEL),
            self._entry_point(
                "user_metric", "tests.fixtures.UserMetric:UserMetric", PluginKind.METRIC
            ),
        ]
        calls = self._publish(monkeypatch, entries)
        registry = PluginRegistry()
        config = {
            "schema_version": "1.0",
            "experiment": {"name": "plugins"},
            "data": {"train": {"target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset"}},
            "model": {"plugin": {"kind": "model", "name": "user_model"}},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "val_metrics": [{"plugin": {"kind": "metric", "name": "user_metric"}}],
        }

        report = ExperimentValidator(plugins=registry).validate(config)
        assert report.is_valid
        assert set(calls) == {"lambdaforge.models", "lambdaforge.metrics"}
        assert registry.resolved_plugins() == ()

        calls.clear()
        registry.refresh()
        template = {
            **config,
            "model": {"plugin": {"kind": "model", "name": "not_installed"}},
        }
        unchecked = ExperimentValidator(plugins=registry).validate(template, check_imports=False)
        assert unchecked.is_valid
        assert calls == []

        wrong_kind = {
            **config,
            "model": {"plugin": {"kind": "metric", "name": "user_metric"}},
        }
        invalid = ExperimentValidator(plugins=registry).validate(wrong_kind, check_imports=False)
        assert not invalid.is_valid
        assert any("model" in error for error in invalid.errors)

        metadata_config = {**config, "metadata": {"plugin": "experiment-tag"}}
        metadata_report = ExperimentValidator(plugins=registry).validate(metadata_config)
        assert metadata_report.is_valid

    def test_plugin_reference_rejects_magic_or_ambiguous_values(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            PluginReference.from_value("model:user_model")
        with pytest.raises(ValueError, match="Unexpected"):
            PluginReference.from_value({"kind": "model", "name": "x", "extra": True})
        with pytest.raises(ValueError, match="Unknown plugin kind"):
            PluginReference.from_value({"kind": "task", "name": "x"})
