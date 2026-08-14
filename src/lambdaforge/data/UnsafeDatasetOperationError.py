"""Error raised when a destructive dataset operation fails safety checks."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class UnsafeDatasetOperationError(DatasetResolutionError):
    """Reject unsafe mutation while preserving recoverable registry/data state."""
