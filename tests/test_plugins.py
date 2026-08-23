"""Entry-point plugin discovery, safety and YAML integration tests."""

import importlib
import importlib.metadata as metadata
from types import SimpleNamespace

import pytest

from lambdaforge.nn import ComponentRegistry
from lambdaforge.nn.activations import GELU, ReLU
from lambdaforge.plugins import (
    PluginKind,
    PluginReference,
    PluginRegistry,
    PluginResolutionError,
)


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
            "lambdaforge.nn.activations:GELU",
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


    def test_component_alias_plugins_are_fallbacks_and_cannot_shadow_builtins(
        self, monkeypatch
    ) -> None:
        entries = [
            self._entry_point(
                "externalgelu",
                "lambdaforge.nn.activations:GELU",
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


    def test_plugin_reference_rejects_magic_or_ambiguous_values(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            PluginReference.from_value("model:user_model")
        with pytest.raises(ValueError, match="Unexpected"):
            PluginReference.from_value({"kind": "model", "name": "x", "extra": True})
        with pytest.raises(ValueError, match="Unknown plugin kind"):
            PluginReference.from_value({"kind": "workflow", "name": "x"})
