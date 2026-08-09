"""Recursive object construction from task-agnostic YAML fragments."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from lambdaforge.plugins.PluginReference import PluginReference
from lambdaforge.plugins.PluginRegistry import PluginRegistry


class ObjectFactory:
    """Import and instantiate Python objects described by YAML mappings.

    ``{"ref": "package.object"}`` returns the referenced object unchanged.
    ``{"target": "package.Class", "params": {...}}`` recursively resolves
    parameters and instantiates the target. Ordinary containers are traversed
    recursively without applying domain-specific assumptions.
    """

    @staticmethod
    def import_object(path: str) -> Any:
        """Import an object from a fully qualified dotted path."""
        if "." not in path:
            raise ValueError(f"Expected a fully-qualified object path, got {path!r}.")
        module_name, object_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        try:
            return getattr(module, object_name)
        except AttributeError as error:
            raise ImportError(f"Module {module_name!r} has no object {object_name!r}.") from error

    @classmethod
    def build(cls, spec: Any, *, plugins: PluginRegistry | None = None) -> Any:
        """Recursively resolve one YAML-compatible object specification.

        Entry-point plugins use ``{"plugin": {"kind": ..., "name": ...}}``.
        Existing fully qualified ``target`` and ``ref`` specifications remain
        unchanged. Resolved plugin classes are instantiated exactly like
        ordinary targets; plugin instances are never shared by the registry.
        """
        from lambdaforge.configuration.SecretValue import SecretValue

        if isinstance(spec, SecretValue):
            return spec.value
        registry = plugins or PluginRegistry.default()
        if isinstance(spec, Mapping):
            keys = set(spec)
            if "ref" in spec:
                if keys != {"ref"}:
                    raise ValueError("A 'ref' object cannot contain additional keys.")
                return cls.import_object(str(spec["ref"]))
            if "plugin" in spec:
                unexpected = keys - {"plugin", "params"}
                if unexpected:
                    raise ValueError(f"Unexpected plugin keys: {sorted(unexpected)}.")
                target = registry.resolve(PluginReference.from_value(spec["plugin"]))
                params = spec.get("params", {})
                if not isinstance(params, Mapping):
                    raise TypeError("'params' must be a mapping when 'plugin' is used.")
                return target(
                    **{key: cls.build(value, plugins=registry) for key, value in params.items()}
                )
            if "target" in spec:
                unexpected = keys - {"target", "params"}
                if unexpected:
                    raise ValueError(f"Unexpected target keys: {sorted(unexpected)}.")
                target = cls.import_object(str(spec["target"]))
                params = spec.get("params", {})
                if not isinstance(params, Mapping):
                    raise TypeError("'params' must be a mapping when 'target' is used.")
                return target(
                    **{key: cls.build(value, plugins=registry) for key, value in params.items()}
                )
            return {key: cls.build(value, plugins=registry) for key, value in spec.items()}
        if isinstance(spec, list):
            return [cls.build(value, plugins=registry) for value in spec]
        if isinstance(spec, tuple):
            return tuple(cls.build(value, plugins=registry) for value in spec)
        return spec
