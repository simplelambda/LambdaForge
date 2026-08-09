"""Safe YAML composition and interpolation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.configuration.ResolvedConfiguration import ResolvedConfiguration
from lambdaforge.configuration.SecretValue import SecretValue
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec


class ConfigurationComposer:
    """Resolve explicit YAML inheritance without expressions or implicit file discovery.

    ``extends`` files are merged first, then ``include`` files, then the current
    document. Mappings merge recursively, sequences replace, and ``{$delete: true}``
    removes an inherited key. Interpolation supports only ``${config:path}``,
    ``${env:NAME}``, and ``${secret:NAME}``.
    """

    _TOKEN = re.compile(r"\$\{(config|env|secret):([^{}]+)\}")
    _MISSING = object()

    def resolve(
        self,
        path: str | Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        override_source: str = "command-line override",
    ) -> ResolvedConfiguration:
        """Compose YAML and apply explicit dotted-path overrides last."""
        source = Path(path).resolve()
        values, provenance, sources = self._load(source, ())
        interpolated = self._interpolate(values, values, ())
        for dotted_path, value in (overrides or {}).items():
            ExperimentConfig.set_value(interpolated, str(dotted_path), value)
            provenance[str(dotted_path)] = override_source
        return ResolvedConfiguration(interpolated, provenance, tuple(dict.fromkeys(sources)))

    def _load(
        self, source: Path, stack: tuple[Path, ...]
    ) -> tuple[dict[str, Any], dict[str, str], list[Path]]:
        if source in stack:
            chain = " -> ".join(str(path) for path in (*stack, source))
            raise ValueError(f"Configuration composition cycle: {chain}")
        loaded, _ = RoundTripYamlCodec().load_file(source)
        current = RoundTripYamlCodec().to_plain_mapping(loaded)
        parents = self._paths(current.pop("extends", ()), source, "extends")
        includes = self._paths(current.pop("include", ()), source, "include")
        merged: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        sources: list[Path] = []
        for dependency in (*parents, *includes):
            child, child_origins, child_sources = self._load(dependency, (*stack, source))
            merged = self._merge(merged, child)
            provenance.update(child_origins)
            sources.extend(child_sources)
        merged = self._merge(merged, current)
        self._record_origins(current, source, provenance)
        sources.append(source)
        return merged, provenance, sources

    @classmethod
    def _merge(cls, base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
        result = {str(key): value for key, value in base.items()}
        for raw_key, value in overlay.items():
            key = str(raw_key)
            if isinstance(value, Mapping) and value == {"$delete": True}:
                result.pop(key, None)
            elif isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
                result[key] = cls._merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _paths(value: Any, source: Path, label: str) -> tuple[Path, ...]:
        if value in (None, ()):
            return ()
        raw: Sequence[Any] = (value,) if isinstance(value, (str, Path)) else value
        if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
            raise TypeError(f"{label} must be a path or sequence of paths in {source}.")
        paths: list[Path] = []
        for item in raw:
            if not isinstance(item, (str, Path)):
                raise TypeError(f"{label} entries must be paths in {source}.")
            candidate = Path(item)
            paths.append(
                (source.parent / candidate).resolve()
                if not candidate.is_absolute()
                else candidate.resolve()
            )
        return tuple(paths)

    def _interpolate(self, value: Any, root: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): self._interpolate(item, root, (*stack, str(key)))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._interpolate(item, root, (*stack, str(index)))
                for index, item in enumerate(value)
            ]
        if not isinstance(value, str):
            return value
        full = self._TOKEN.fullmatch(value)
        if full:
            return self._resolve_token(full.group(1), full.group(2), root, stack)

        def replace(match: re.Match[str]) -> str:
            resolved = self._resolve_token(match.group(1), match.group(2), root, stack)
            if isinstance(resolved, SecretValue):
                raise ValueError(
                    "Secret interpolation must occupy the complete value so redaction remains "
                    "structurally reliable."
                )
            return str(resolved)

        return self._TOKEN.sub(replace, value)

    def _resolve_token(
        self, kind: str, name: str, root: Mapping[str, Any], stack: tuple[str, ...]
    ) -> Any:
        if kind == "config":
            if name in stack:
                raise ValueError(f"Configuration interpolation cycle at {name!r}.")
            value = ExperimentConfig.get_value(root, name, self._MISSING)
            if value is self._MISSING:
                raise KeyError(f"Unknown configuration reference: {name!r}.")
            return self._interpolate(value, root, (*stack, name))
        if name not in os.environ:
            raise KeyError(f"Required environment variable {name!r} is not set.")
        return (
            SecretValue(os.environ[name], f"environment:{name}")
            if kind == "secret"
            else os.environ[name]
        )

    @classmethod
    def _record_origins(
        cls, value: Mapping[str, Any], source: Path, output: dict[str, str], prefix: str = ""
    ) -> None:
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output[path] = str(source)
            if isinstance(item, Mapping):
                cls._record_origins(item, source, output, path)
