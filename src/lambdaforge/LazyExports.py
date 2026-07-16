"""Object-backed lazy package export resolver."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any


class LazyExports(ModuleType):
    """Module object that keeps class exports stable while loading them lazily.

    Python normally assigns an imported submodule to its parent package using
    the submodule's final name. LambdaForge class modules intentionally have
    that same name, so this object also replaces a cached submodule attribute
    with the public class whenever it is requested.
    """

    @classmethod
    def install(cls, package_name: str, exports: Mapping[str, tuple[str, str]]) -> None:
        """Upgrade an existing package module and attach its lazy export map."""
        package = sys.modules[package_name]
        package.__class__ = cls
        package.__dict__["_lazy_exports"] = dict(exports)

    def __getattribute__(self, name: str) -> Any:
        """Resolve configured names before ordinary module attribute lookup."""
        namespace = ModuleType.__getattribute__(self, "__dict__")
        exports = namespace.get("_lazy_exports", {})
        if name in exports:
            current = namespace.get(name)
            if current is None or isinstance(current, ModuleType):
                module_name, attribute_name = exports[name]
                module = importlib.import_module(module_name)
                current = getattr(module, attribute_name)
                namespace[name] = current
            return current
        return ModuleType.__getattribute__(self, name)
