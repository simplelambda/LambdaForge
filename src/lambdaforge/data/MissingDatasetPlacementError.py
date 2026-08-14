"""Error raised when an immutable dataset lacks the requested placement."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class MissingDatasetPlacementError(DatasetResolutionError):
    """Explain available placements and the exact materialization command."""

    def __init__(self, selector: str, cluster: str, available: tuple[str, ...]) -> None:
        rendered = ", ".join(available) if available else "none"
        super().__init__(
            f"Dataset {selector} is registered but not materialized on {cluster}.\n"
            f"Available placements: {rendered}\n"
            f"Next: lf datasets materialize {selector} --on {cluster} --apply"
        )
