"""Base error for actionable logical dataset resolution failures."""


class DatasetResolutionError(RuntimeError):
    """Mark expected dataset resolution failures for concise CLI rendering."""
