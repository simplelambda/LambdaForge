"""Error raised when neither a managed dataset nor external descriptor exists."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class UnknownDatasetError(DatasetResolutionError):
    """Report an unknown logical dataset with an actionable next step."""

    def __init__(self, selector: str, known: tuple[str, ...] = ()) -> None:
        message = f"Dataset {selector!r} is not registered."
        if known:
            message += f"\nKnown datasets: {', '.join(known)}"
        message += f"\nNext: lf datasets build {selector.split('@', 1)[0]} --on CLUSTER"
        super().__init__(message)
