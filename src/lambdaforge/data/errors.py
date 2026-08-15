"""Actionable failures for managed dataset resolution and lifecycle operations."""

from __future__ import annotations


class DatasetResolutionError(RuntimeError):
    """Base class for expected dataset failures rendered concisely by the CLI."""


class UnknownDatasetError(DatasetResolutionError):
    """Report an unknown logical dataset with an actionable next step."""

    def __init__(self, selector: str, known: tuple[str, ...] = ()) -> None:
        self.selector = selector
        self.known = known
        message = f"Dataset {selector!r} is not registered."
        if known:
            message += f"\nKnown datasets: {', '.join(known)}"
        message += f"\nNext: lf datasets build {selector.split('@', 1)[0]} --on CLUSTER"
        super().__init__(message)


class AmbiguousDatasetVersionError(DatasetResolutionError):
    """Require explicit version selection rather than guessing scientific data."""

    def __init__(self, name: str, versions: tuple[str, ...]) -> None:
        self.name = name
        self.versions = versions
        super().__init__(
            f"Dataset {name!r} has multiple versions: {', '.join(versions)}.\n"
            f"Use an exact reference such as dataset:{name}@{versions[-1]}."
        )


class MissingDatasetPlacementError(DatasetResolutionError):
    """Explain available placements and the exact materialization command."""

    def __init__(self, selector: str, cluster: str, available: tuple[str, ...]) -> None:
        self.selector = selector
        self.cluster = cluster
        self.available = available
        rendered = ", ".join(available) if available else "none"
        super().__init__(
            f"Dataset {selector} is registered but not materialized on {cluster}.\n"
            f"Available placements: {rendered}\n"
            f"Next: lf datasets materialize {selector} --on {cluster} --apply"
        )


class MissingDatasetRecipeError(DatasetResolutionError):
    """Identify a recipe required for BUILD rather than leaking an internal KeyError."""

    def __init__(self, selector: str, known: tuple[str, ...] = ()) -> None:
        self.selector = selector
        self.known = known
        rendered = ", ".join(known) if known else "none"
        super().__init__(
            f"No DatasetRecipe is known for {selector!r}.\n"
            f"Known recipes: {rendered}\n"
            f"Next: create/discover a kind: dataset recipe, then run "
            f"lf datasets build {selector.split('@', 1)[0]}"
        )


class InvalidDatasetBuildError(DatasetResolutionError):
    """Prevent incomplete stage evidence from becoming a published version."""


class UnsafeDatasetOperationError(DatasetResolutionError):
    """Reject unsafe mutation while preserving recoverable registry/data state."""


class MissingManagedEnvironmentError(RuntimeError):
    """Explain how to prepare project code before a remote dataset operation."""

    def __init__(self, cluster: str) -> None:
        super().__init__(
            f"Cluster {cluster!r} has no active managed LambdaForge environment. "
            f"Run: lf clusters bootstrap {cluster}"
        )


class OfflineClusterError(RuntimeError):
    """Represent an unavailable context without hiding the selected cluster."""

    def __init__(self, cluster: str, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"Cluster {cluster!r} is unavailable{suffix}.")
