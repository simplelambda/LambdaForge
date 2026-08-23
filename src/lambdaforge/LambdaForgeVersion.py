"""Single authority for the installed LambdaForge framework version."""

from lambdaforge._version import VERSION


class LambdaForgeVersion:
    """Read-only class namespace over the packaging-owned version constant."""

    CURRENT = VERSION
