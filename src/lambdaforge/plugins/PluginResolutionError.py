"""Public error raised when an installed plugin cannot be resolved safely."""


class PluginResolutionError(ValueError):
    """Report missing, conflicting, unloadable or contract-invalid plugins."""
