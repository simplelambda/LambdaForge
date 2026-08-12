"""Credential-provider boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CredentialProvider(ABC):
    """Resolve or mutate credentials outside cluster configuration."""

    @abstractmethod
    def get(self, reference: str, *, prompt: str) -> str:
        """Return one secret for immediate in-memory use."""

    def set(self, reference: str, secret: str) -> None:
        """Store a secret when supported by the provider."""
        del reference, secret
        raise NotImplementedError("This credential provider is read-only.")

    def delete(self, reference: str) -> None:
        """Delete a secret when supported by the provider."""
        del reference
        raise NotImplementedError("This credential provider is read-only.")
