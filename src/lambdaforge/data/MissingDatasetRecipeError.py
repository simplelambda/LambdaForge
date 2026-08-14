"""Error raised when BUILD needs a recipe that is not discoverable."""

from lambdaforge.data.DatasetResolutionError import DatasetResolutionError


class MissingDatasetRecipeError(DatasetResolutionError):
    """Identify the absent recipe rather than leaking an internal KeyError."""

    def __init__(self, selector: str, known: tuple[str, ...] = ()) -> None:
        rendered = ", ".join(known) if known else "none"
        super().__init__(
            f"No DatasetRecipe is known for {selector!r}.\n"
            f"Known recipes: {rendered}\n"
            f"Next: create/discover a kind: dataset recipe, then run "
            f"lf datasets build {selector.split('@', 1)[0]}"
        )
