"""Non-executing metadata descriptions of installed LambdaForge plugins."""

from __future__ import annotations

from dataclasses import dataclass

from lambdaforge.plugins.PluginKind import PluginKind


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    """Describe one installed or successfully resolved entry point."""

    kind: PluginKind
    name: str
    value: str
    distribution: str | None = None
    version: str | None = None

    @property
    def group(self) -> str:
        """Return the canonical entry-point group."""
        return self.kind.entry_point_group

    def to_dict(self) -> dict[str, str | None]:
        """Return stable JSON-compatible metadata for CLI and integrations."""
        return {
            "kind": self.kind.value,
            "name": self.name,
            "group": self.group,
            "value": self.value,
            "distribution": self.distribution,
            "version": self.version,
        }

    def sort_key(self) -> tuple[str, str, str, str, str, str]:
        """Return the canonical ordering key used by manifests and registries."""
        return (
            self.kind.value,
            self.name,
            self.group,
            self.distribution or "",
            self.version or "",
            self.value,
        )
