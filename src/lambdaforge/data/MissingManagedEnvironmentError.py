"""Typed failure for remote operations requiring a prepared managed environment."""


class MissingManagedEnvironmentError(RuntimeError):
    """Explain how to prepare project code before a remote dataset operation."""

    def __init__(self, cluster: str) -> None:
        super().__init__(
            f"Cluster {cluster!r} has no active managed LambdaForge environment. "
            f"Run: lf clusters bootstrap {cluster}"
        )
