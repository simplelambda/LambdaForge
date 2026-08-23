"""Supported categories for installed LambdaForge entry-point plugins."""

from __future__ import annotations

from enum import Enum


class PluginKind(str, Enum):
    """Map user-facing plugin kinds to namespaced entry-point groups."""

    MODEL = "model"
    METRIC = "metric"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    LOSS = "loss"
    DISTANCE = "distance"
    POOLING = "pooling"
    SIMILARITY = "similarity"
    KERNEL = "kernel"
    ENCODING = "encoding"
    REGULARIZATION = "regularization"
    DATASET = "dataset"
    CALLBACK = "callback"
    LOGGER = "logger"

    @property
    def entry_point_group(self) -> str:
        """Return the canonical distribution metadata group for this kind."""
        if self is PluginKind.MODEL:
            return "lambdaforge.models"
        if self is PluginKind.METRIC:
            return "lambdaforge.metrics"
        if self is PluginKind.ACTIVATION:
            return "lambdaforge.activations"
        if self is PluginKind.NORMALIZATION:
            return "lambdaforge.normalizations"
        if self is PluginKind.LOSS:
            return "lambdaforge.losses"
        if self is PluginKind.DISTANCE:
            return "lambdaforge.distances"
        if self is PluginKind.POOLING:
            return "lambdaforge.pooling"
        if self is PluginKind.SIMILARITY:
            return "lambdaforge.similarities"
        if self is PluginKind.KERNEL:
            return "lambdaforge.kernels"
        if self is PluginKind.ENCODING:
            return "lambdaforge.encodings"
        if self is PluginKind.REGULARIZATION:
            return "lambdaforge.regularization"
        if self is PluginKind.DATASET:
            return "lambdaforge.datasets"
        if self is PluginKind.CALLBACK:
            return "lambdaforge.callbacks"
        if self is PluginKind.LOGGER:
            return "lambdaforge.loggers"
        raise AssertionError(f"Unhandled plugin kind: {self!r}.")

    @classmethod
    def from_value(cls, value: PluginKind | str) -> PluginKind:
        """Normalize an enum or YAML string into a supported plugin kind."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            choices = ", ".join(kind.value for kind in cls)
            raise ValueError(f"Unknown plugin kind {value!r}. Options: {choices}.") from error
