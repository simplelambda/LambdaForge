"""Typed remote dataset connectivity failure."""


class OfflineClusterError(RuntimeError):
    """Represent an unavailable execution/placement context without hiding its name."""

    def __init__(self, cluster: str, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"Cluster {cluster!r} is unavailable{suffix}.")
