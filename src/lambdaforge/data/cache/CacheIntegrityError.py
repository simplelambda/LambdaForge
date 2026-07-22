"""Error raised when a persistent cache record fails envelope verification."""


class CacheIntegrityError(ValueError):
    """Report malformed, corrupt or unauthenticated persistent cache bytes."""
