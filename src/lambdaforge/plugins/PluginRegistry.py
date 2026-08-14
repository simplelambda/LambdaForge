"""Lazy discovery and contract validation for installed entry-point plugins."""

from __future__ import annotations

import importlib.metadata as metadata
import os
from collections.abc import Mapping
from contextvars import ContextVar, Token
from threading import RLock
from typing import Any, ClassVar

from lambdaforge.plugins.PluginDescriptor import PluginDescriptor
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginReference import PluginReference
from lambdaforge.plugins.PluginResolutionError import PluginResolutionError
from lambdaforge.plugins.PluginUsageSession import PluginUsageSession


class PluginRegistry:
    """Discover plugins lazily and track successful class resolutions.

    Merely listing plugins reads installed distribution metadata and does not
    import plugin modules. :meth:`resolve` imports only the explicitly selected
    entry point, rejects ambiguous names, and checks its framework contract
    before returning the class to an object factory. Run-scoped usage sessions
    isolate provenance from earlier validation and execution activity.
    """

    _default: ClassVar[PluginRegistry | None] = None
    _default_lock: ClassVar[RLock] = RLock()
    _default_process_id: ClassVar[int] = os.getpid()

    def __init__(self) -> None:
        self._entries: dict[PluginKind, dict[str, tuple[metadata.EntryPoint, ...]]] = {}
        self._loaded: dict[PluginReference, type[Any]] = {}
        self._loaded_descriptors: dict[PluginReference, PluginDescriptor] = {}
        self._resolved: dict[PluginDescriptor, None] = {}
        self._process_id = os.getpid()
        self._lock = RLock()
        self._usage_sessions: ContextVar[tuple[PluginUsageSession, ...]] = ContextVar(
            f"lambdaforge_plugin_usage_{id(self)}",
            default=(),
        )

    @classmethod
    def default(cls) -> PluginRegistry:
        """Return the process-local registry used by public convenience APIs."""
        process_id = os.getpid()
        if process_id != cls._default_process_id:
            cls._default_process_id = process_id
            cls._default_lock = RLock()
            cls._default = None
        with cls._default_lock:
            if cls._default is None:
                cls._default = cls()
            return cls._default

    def discover(
        self,
        kind: PluginKind | str | None = None,
    ) -> tuple[PluginDescriptor, ...]:
        """List deterministic plugin metadata without loading plugin modules."""
        self._ensure_process_local_state()
        kinds = tuple(PluginKind) if kind is None else (PluginKind.from_value(kind),)
        descriptors = [
            self._descriptor(plugin_kind, entry_point)
            for plugin_kind in kinds
            for entries in self._entries_for(plugin_kind).values()
            for entry_point in entries
        ]
        return tuple(sorted(descriptors, key=PluginDescriptor.sort_key))

    def usage_session(self) -> PluginUsageSession:
        """Create an isolated collector for successful resolutions in one run."""
        self._ensure_process_local_state()
        return PluginUsageSession(self)

    def resolved_plugins(self) -> tuple[PluginDescriptor, ...]:
        """Return successful resolutions made by this registry in this process.

        This process-level ledger is useful for diagnostics. Reproducibility
        artifacts should use :meth:`usage_session` so earlier validations or
        sequential runs cannot contaminate a run-specific manifest.
        """
        self._ensure_process_local_state()
        with self._lock:
            return tuple(sorted(self._resolved, key=PluginDescriptor.sort_key))

    def resolve(
        self,
        reference: PluginReference | Mapping[str, Any],
        *,
        record_usage: bool = True,
    ) -> type[Any]:
        """Load and validate one plugin class, optionally recording real usage."""
        self._ensure_process_local_state()
        if not isinstance(record_usage, bool):
            raise TypeError("record_usage must be a boolean.")
        plugin = PluginReference.from_value(reference)
        with self._lock:
            cached = self._loaded.get(plugin)
            if cached is not None:
                if record_usage:
                    self._record_resolution(self._loaded_descriptors[plugin])
                return cached

            entries = self._entries_for(plugin.kind).get(plugin.name, ())
            if not entries:
                available = sorted(self._entries_for(plugin.kind))
                suffix = f" Available names: {available}." if available else ""
                raise PluginResolutionError(
                    f"No plugin {plugin.name!r} is installed in group "
                    f"{plugin.kind.entry_point_group!r}.{suffix}"
                )
            if len(entries) > 1:
                providers = ", ".join(
                    self._provider_label(self._descriptor(plugin.kind, entry_point))
                    for entry_point in entries
                )
                raise PluginResolutionError(
                    f"Plugin {plugin.name!r} is ambiguous in group "
                    f"{plugin.kind.entry_point_group!r}; providers: {providers}."
                )

            entry_point = entries[0]
            descriptor = self._descriptor(plugin.kind, entry_point)
            try:
                loaded = entry_point.load()
            except Exception as error:
                raise PluginResolutionError(
                    f"Could not load plugin {plugin.name!r} from "
                    f"{self._provider_label(descriptor)}: {error.__class__.__name__}: {error}"
                ) from error

            if not isinstance(loaded, type):
                raise PluginResolutionError(
                    f"Plugin {plugin.name!r} from {self._provider_label(descriptor)} must expose "
                    f"a class, got {type(loaded).__name__}."
                )
            expected = self._expected_base(plugin.kind)
            if not issubclass(loaded, expected):
                raise PluginResolutionError(
                    f"Plugin {plugin.name!r} from {self._provider_label(descriptor)} must subclass "
                    f"{expected.__module__}.{expected.__name__}, got "
                    f"{loaded.__module__}.{loaded.__name__}."
                )
            self._loaded[plugin] = loaded
            self._loaded_descriptors[plugin] = descriptor
            if record_usage:
                self._record_resolution(descriptor)
            return loaded

    def refresh(self, kind: PluginKind | str | None = None) -> None:
        """Discard discovery and load caches after environment metadata changes."""
        self._ensure_process_local_state()
        with self._lock:
            if kind is None:
                self._entries.clear()
                self._loaded.clear()
                self._loaded_descriptors.clear()
                return
            plugin_kind = PluginKind.from_value(kind)
            self._entries.pop(plugin_kind, None)
            self._loaded = {
                reference: loaded
                for reference, loaded in self._loaded.items()
                if reference.kind is not plugin_kind
            }
            self._loaded_descriptors = {
                reference: descriptor
                for reference, descriptor in self._loaded_descriptors.items()
                if reference.kind is not plugin_kind
            }

    def _record_resolution(self, descriptor: PluginDescriptor) -> None:
        """Record a verified class in process and active-context ledgers."""
        self._resolved[descriptor] = None
        for session in self._usage_sessions.get():
            session._record(descriptor)

    def _activate_usage_session(
        self,
        session: PluginUsageSession,
    ) -> Token[tuple[PluginUsageSession, ...]]:
        """Push one run collector into the current execution context."""
        self._ensure_process_local_state()
        current = self._usage_sessions.get()
        return self._usage_sessions.set((*current, session))

    def _deactivate_usage_session(
        self,
        token: Token[tuple[PluginUsageSession, ...]],
    ) -> None:
        """Restore the context that preceded one run collector."""
        self._usage_sessions.reset(token)

    def _entries_for(
        self,
        kind: PluginKind,
    ) -> dict[str, tuple[metadata.EntryPoint, ...]]:
        with self._lock:
            cached = self._entries.get(kind)
            if cached is not None:
                return cached
            grouped: dict[str, list[metadata.EntryPoint]] = {}
            for entry_point in metadata.entry_points(group=kind.entry_point_group):
                grouped.setdefault(entry_point.name, []).append(entry_point)
            entries = {
                name: tuple(
                    sorted(
                        values,
                        key=lambda item: (
                            self._distribution_name(item) or "",
                            item.value,
                        ),
                    )
                )
                for name, values in grouped.items()
            }
            self._entries[kind] = entries
            return entries

    def _ensure_process_local_state(self) -> None:
        """Replace locks and metadata caches inherited through a process fork."""
        process_id = os.getpid()
        if process_id == self._process_id:
            return
        self._process_id = process_id
        self._entries = {}
        self._loaded = {}
        self._loaded_descriptors = {}
        self._resolved = {}
        self._lock = RLock()
        self._usage_sessions = ContextVar(
            f"lambdaforge_plugin_usage_{id(self)}_{process_id}",
            default=(),
        )

    @staticmethod
    def _expected_base(kind: PluginKind) -> type[Any]:
        if kind is PluginKind.MODEL:
            from torch.nn import Module

            return Module
        if kind is PluginKind.METRIC:
            from lambdaforge.metrics.Metric import Metric

            return Metric
        if kind is PluginKind.ACTIVATION:
            from lambdaforge.nn.activations.base import Activation

            return Activation
        if kind is PluginKind.NORMALIZATION:
            from lambdaforge.nn.normalizations.Normalization import Normalization

            return Normalization
        if kind is PluginKind.LOSS:
            from lambdaforge.nn.losses.Loss import Loss

            return Loss
        if kind is PluginKind.DISTANCE:
            from lambdaforge.nn.distances.Distance import Distance

            return Distance
        if kind is PluginKind.POOLING:
            from lambdaforge.nn.pooling.Pooling import Pooling

            return Pooling
        if kind is PluginKind.SIMILARITY:
            from lambdaforge.nn.similarities.Similarity import Similarity

            return Similarity
        if kind is PluginKind.KERNEL:
            from lambdaforge.nn.kernels.Kernel import Kernel

            return Kernel
        if kind is PluginKind.ENCODING:
            from lambdaforge.nn.encodings.Encoding import Encoding

            return Encoding
        if kind is PluginKind.REGULARIZATION:
            from lambdaforge.nn.regularization.Regularization import Regularization

            return Regularization
        if kind is PluginKind.DATASET:
            from torch.utils.data import Dataset

            return Dataset
        if kind is PluginKind.CALLBACK:
            from lambdaforge.integrations.Lightning import CallbackBase

            return CallbackBase
        if kind is PluginKind.LOGGER:
            from lambdaforge.integrations.Lightning import LoggerType

            return LoggerType
        if kind is PluginKind.TASK:
            from lambdaforge.tasks.Task import Task

            return Task
        raise AssertionError(f"Unhandled plugin kind: {kind!r}.")

    @classmethod
    def _descriptor(
        cls,
        kind: PluginKind,
        entry_point: metadata.EntryPoint,
    ) -> PluginDescriptor:
        distribution = getattr(entry_point, "dist", None)
        return PluginDescriptor(
            kind=kind,
            name=entry_point.name,
            value=entry_point.value,
            distribution=cls._distribution_name(entry_point),
            version=getattr(distribution, "version", None),
        )

    @staticmethod
    def _distribution_name(entry_point: metadata.EntryPoint) -> str | None:
        distribution = getattr(entry_point, "dist", None)
        if distribution is None:
            return None
        name = distribution.metadata.get("Name")
        return str(name) if name else None

    @staticmethod
    def _provider_label(descriptor: PluginDescriptor) -> str:
        distribution = descriptor.distribution or "unknown distribution"
        version = f" {descriptor.version}" if descriptor.version else ""
        return f"{distribution}{version} ({descriptor.value})"
