"""Spawn-safe worker used to verify process-local plugin provenance."""

from pathlib import Path

from lambdaforge.EnvironmentManifest import EnvironmentManifest
from lambdaforge.plugins import PluginKind, PluginReference, PluginRegistry


class PluginUsageSpawnJob:
    """Resolve one installed entry point and write its child-process manifest."""

    @staticmethod
    def run(output_path: str) -> None:
        """Execute the provenance capture in a multiprocessing spawn child."""
        registry = PluginRegistry.default()
        with registry.usage_session() as usage:
            registry.resolve(PluginReference(PluginKind.MODEL, "spawn_model"))
        EnvironmentManifest.capture(plugins=usage.descriptors()).write(Path(output_path))
