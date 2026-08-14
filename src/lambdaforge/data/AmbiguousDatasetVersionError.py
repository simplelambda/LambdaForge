"""Error raised when an unversioned dataset selector has multiple versions."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class AmbiguousDatasetVersionError(DatasetResolutionError):
    """Require explicit version selection rather than guessing scientific data."""

    def __init__(self, name: str, versions: tuple[str, ...]) -> None:
        super().__init__(
            f"Dataset {name!r} has multiple versions: {', '.join(versions)}.\n"
            f"Use an exact reference such as dataset:{name}@{versions[-1]}."
        )
