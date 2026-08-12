"""Environment-backed ephemeral credentials."""

from __future__ import annotations

import os
from collections.abc import Mapping

from lambdaforge.controlplane.CredentialProvider import CredentialProvider


class EnvironmentCredentialProvider(CredentialProvider):
    """Resolve ``env:NAME`` without copying its value into configuration."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self._environment = environment if environment is not None else os.environ

    def get(self, reference: str, *, prompt: str) -> str:
        """Read the named variable and return only its value."""
        del prompt
        if not reference.startswith("env:") or not reference[4:]:
            raise ValueError("Environment credentials must use env:VARIABLE.")
        name = reference[4:]
        try:
            value = self._environment[name]
        except KeyError as error:
            raise RuntimeError(f"Credential environment variable {name!r} is not set.") from error
        if not value:
            raise RuntimeError(f"Credential environment variable {name!r} is empty.")
        return value
