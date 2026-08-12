"""Hidden interactive credential input."""

from __future__ import annotations

import getpass
from collections.abc import Callable

from lambdaforge.controlplane.CredentialProvider import CredentialProvider


class InteractiveCredentialProvider(CredentialProvider):
    """Read a password from the terminal without echoing or persisting it."""

    def __init__(self, reader: Callable[[str], str] = getpass.getpass) -> None:
        self._reader = reader

    def get(self, reference: str, *, prompt: str) -> str:
        """Prompt once; ``reference`` is deliberately ignored."""
        del reference
        value = self._reader(prompt)
        if not value:
            raise ValueError("An empty cluster password is not allowed.")
        return value
