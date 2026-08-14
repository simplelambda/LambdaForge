"""Error raised for invalid or failed dataset build evidence."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class InvalidDatasetBuildError(DatasetResolutionError):
    """Prevent incomplete stage evidence from becoming a published version."""
