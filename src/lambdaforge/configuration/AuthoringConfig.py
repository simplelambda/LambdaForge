"""User-facing concise configuration document."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.configuration.AuthoringConfigNormalizer import AuthoringConfigNormalizer
from lambdaforge.configuration.MaterializedConfig import MaterializedConfig


class AuthoringConfig:
    """Load concise YAML and compile it to the strict internal representation."""

    def __init__(self, values: Mapping[str, Any], *, source: str | Path | None = None) -> None:
        self.values = dict(values)
        self.source = Path(source).resolve() if source is not None else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AuthoringConfig:
        """Load a composed authoring document without constructing user objects."""
        from lambdaforge.configuration.ConfigurationComposer import ConfigurationComposer

        source = Path(path).resolve()
        resolution = ConfigurationComposer().resolve(source)
        return cls(resolution.values, source=source)

    def materialize(self) -> MaterializedConfig:
        """Compile defaults and shorthand to a strict runnable document."""
        return AuthoringConfigNormalizer().normalize(self.values, source=self.source)
