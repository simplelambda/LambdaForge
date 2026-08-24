"""Environment provenance capture and run-artifact integration."""

import json

from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.plugins import PluginDescriptor, PluginKind


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
        assert "scikit-learn" in content["packages"]
        assert "scipy" in content["packages"]
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
