"""Cluster credential orchestration."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentCredentialProvider import EnvironmentCredentialProvider
from lambdaforge.controlplane.InteractiveCredentialProvider import InteractiveCredentialProvider
from lambdaforge.controlplane.SystemKeyringCredentialProvider import SystemKeyringCredentialProvider


class CredentialService:
    """Resolve password references while keeping values outside profiles and storage."""

    def __init__(
        self,
        *,
        interactive: InteractiveCredentialProvider | None = None,
        environment: EnvironmentCredentialProvider | None = None,
        keyring: SystemKeyringCredentialProvider | None = None,
    ) -> None:
        self.interactive = interactive or InteractiveCredentialProvider()
        self.environment = environment or EnvironmentCredentialProvider()
        self.keyring = keyring or SystemKeyringCredentialProvider()

    def resolve(self, profile: ClusterProfile) -> str:
        """Resolve the selected password for immediate transport use."""
        reference = profile.auth.credential
        prompt = f"Password for {profile.user or ''}@{profile.host}: "
        if reference is None:
            return self.interactive.get("interactive", prompt=prompt)
        if reference.startswith("env:"):
            return self.environment.get(reference, prompt=prompt)
        return self.keyring.get(reference, prompt=prompt)

    def store(self, reference: str, secret: str) -> None:
        """Store a password under an explicit keyring reference."""
        self.keyring.set(reference, secret)

    def delete(self, reference: str) -> None:
        """Delete an explicit keyring reference."""
        self.keyring.delete(reference)
