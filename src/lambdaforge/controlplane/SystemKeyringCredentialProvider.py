"""System-keyring credential provider."""

from __future__ import annotations

from typing import Any

from lambdaforge.controlplane.CredentialProvider import CredentialProvider
from lambdaforge.controlplane.SecretRedactor import SecretRedactor


class SystemKeyringCredentialProvider(CredentialProvider):
    """Store passwords in the operating system keyring through optional ``keyring``."""

    SERVICE = "lambdaforge"

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    def get(self, reference: str, *, prompt: str) -> str:
        """Retrieve one keyring entry without exposing it in an error."""
        del prompt
        key = self._key(reference)
        value = self._module().get_password(self.SERVICE, key)
        if value is None:
            raise RuntimeError(
                f"No password exists for {reference!r}; run 'lambdaforge clusters credentials "
                "set NAME' or choose interactive/env authentication."
            )
        return str(value)

    def set(self, reference: str, secret: str) -> None:
        """Write one password to the platform keyring."""
        if not secret:
            raise ValueError("An empty cluster password is not allowed.")
        try:
            self._module().set_password(self.SERVICE, self._key(reference), secret)
        except Exception as error:
            raise RuntimeError(
                "System keyring could not store the cluster credential: "
                f"{SecretRedactor.redact(error, (secret,))}"
            ) from None

    def delete(self, reference: str) -> None:
        """Delete one password from the platform keyring."""
        self._module().delete_password(self.SERVICE, self._key(reference))

    def available(self) -> bool:
        """Return whether the optional provider imports successfully."""
        try:
            self._module()
        except RuntimeError:
            return False
        return True

    def _module(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            import keyring
        except ImportError as error:
            raise RuntimeError(
                "System keyring support is unavailable. Install "
                "'lambdaforge[cluster-password]' or use interactive/env authentication."
            ) from error
        return keyring

    @staticmethod
    def _key(reference: str) -> str:
        if not reference.startswith("keyring:") or not reference[8:]:
            raise ValueError("Keyring credentials must use keyring:IDENTIFIER.")
        return reference[8:]
