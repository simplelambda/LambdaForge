"""Environment provenance capture and run-artifact integration."""

import importlib.metadata as metadata
import json
from types import SimpleNamespace

import pytest

from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.experiments import ExperimentRunner
from lambdaforge.plugins import PluginDescriptor, PluginKind, PluginReference, PluginRegistry


class TestEnvironmentManifest:
    """Verify that provenance is typed, serializable and emitted per run."""

    def test_capture_and_write_contains_reproducibility_fields(self, tmp_path) -> None:
        manifest = EnvironmentManifest.capture(tmp_path)
        path = manifest.write(tmp_path / "environment.json")
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["lambdaforge_version"]
        assert content["python"]["version"]
        assert content["platform"]["system"]
        assert content["torch"]["version"]
        assert "cuda_available" in content["torch"]
        assert "numpy" in content["packages"]
        assert "lightning" in content["packages"]
        assert "pytorch-lightning" in content["packages"]
        assert "mlflow" in content["packages"]
        assert "tensorboard" in content["packages"]
        assert "tensorboardX" in content["packages"]
        assert "wandb" in content["packages"]
        assert content["plugins"] == []

    def test_manifest_serializes_deduplicated_plugins_in_canonical_order(self, tmp_path) -> None:
        model = PluginDescriptor(
            kind=PluginKind.MODEL,
            name="z_model",
            value="acme.models:Z",
            distribution="acme-models",
            version="1.0",
        )
        metric = PluginDescriptor(
            kind=PluginKind.METRIC,
            name="a_metric",
            value="acme.metrics:A",
        )

        manifest = EnvironmentManifest.capture(
            tmp_path,
            plugins=(model, metric, model),
        )
        content = manifest.to_dict()

        assert [item["kind"] for item in content["plugins"]] == ["metric", "model"]
        assert content["plugins"][0] == {
            "kind": "metric",
            "name": "a_metric",
            "group": "lambdaforge.metrics",
            "value": "acme.metrics:A",
            "distribution": None,
            "version": None,
        }
        assert manifest.with_plugins(()).plugins == ()

    def test_dry_run_writes_manifest_beside_materialized_config(self, tmp_path) -> None:
        config = {
            "experiment": {
                "name": "manifest_demo",
                "base_name": "manifest_demo",
                "variant": "base",
                "seed": 7,
                "output_root": str(tmp_path),
            }
        }
        result = ExperimentRunner().run_single_experiment(config, dry_run=True)
        run_dir = tmp_path / "manifest_demo" / "base" / "seed=7"
        assert result["status"] == "dry_run"
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "environment.json").exists()
        content = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        assert content["plugins"] == []

    def test_failed_run_persists_only_plugins_resolved_in_that_run(
        self, tmp_path, monkeypatch
    ) -> None:
        distribution = SimpleNamespace(
            metadata={"Name": "lambda-test-plugins"},
            version="3.2.0",
        )
        entries = [
            metadata.EntryPoint(
                name=name,
                value="tests.fixtures.UserModel:UserModel",
                group=PluginKind.MODEL.entry_point_group,
            )._for(distribution)
            for name in ("previous_model", "used_model")
        ]

        def selected_entry_points(**selection):
            return metadata.EntryPoints(entries).select(**selection)

        monkeypatch.setattr(metadata, "entry_points", selected_entry_points)
        registry = PluginRegistry()
        monkeypatch.setattr(PluginRegistry, "_default", registry)
        registry.resolve(PluginReference(PluginKind.MODEL, "previous_model"))
        config = {
            "experiment": {
                "name": "plugin_failure",
                "base_name": "plugin_failure",
                "variant": "base",
                "seed": 5,
                "output_root": str(tmp_path),
            },
            "data": {
                "train": {
                    "target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset",
                }
            },
            "model": {"plugin": {"kind": "model", "name": "used_model"}},
            "losses": [],
        }

        with pytest.raises(RuntimeError, match="at least one loss"):
            ExperimentRunner().run_single_experiment(config)

        path = tmp_path / "plugin_failure" / "base" / "seed=5" / "environment.json"
        content = json.loads(path.read_text(encoding="utf-8"))
        assert [item["name"] for item in content["plugins"]] == ["used_model"]
        assert content["plugins"][0]["distribution"] == "lambda-test-plugins"
        assert content["plugins"][0]["version"] == "3.2.0"
