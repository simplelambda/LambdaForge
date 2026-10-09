"""Construct control-plane providers from one cluster profile."""

import sys
from pathlib import PurePosixPath
from threading import Lock
from typing import cast

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.CredentialService import CredentialService
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExistingEnvironmentProvider import ExistingEnvironmentProvider
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.PasswordSshTransport import PasswordSshTransport
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SlurmProfile import SlurmProfile
from lambdaforge.controlplane.SlurmScheduler import SlurmScheduler
from lambdaforge.controlplane.SshTransport import SshTransport
from lambdaforge.controlplane.Transport import Transport


class ControlPlaneFactory:
    """Centralize provider selection and keep services dependency-injectable."""

    def __init__(self, credentials: CredentialService | None = None) -> None:
        self.credentials = credentials or CredentialService()
        self._transports: dict[str, tuple[dict[str, object], Transport]] = {}
        self._transport_lock = Lock()

    @staticmethod
    def reader_command(profile: ClusterProfile, *, preferred: str | None = None) -> tuple[str, ...]:
        """Read-only interpreter selection, without site GPU claims or activation side effects."""
        if profile.transport == "local":
            return (preferred or sys.executable,)
        assert profile.storage is not None
        choose = """
preferred=$1; fallback=$2; pointer=$3; shift 3
if [ "$preferred" != '-' ] && command -v "$preferred" >/dev/null 2>&1; then
 exec "$preferred" "$@"
fi
if [ -r "$pointer" ]; then
 IFS= read -r active < "$pointer"
 if [ -x "$active" ]; then exec "$active" "$@"; fi
fi
exec "$fallback" "$@"
"""
        return (
            "sh",
            "-c",
            choose,
            "lambdaforge-evidence-reader",
            preferred or "-",
            profile.python,
            str(PurePosixPath(profile.storage.state_root) / "active-environment"),
        )

    def transport(self, profile: ClusterProfile) -> Transport:
        """Build the configured transport."""
        descriptor = profile.to_dict()
        with self._transport_lock:
            cached = self._transports.get(profile.name)
            if cached is not None and cached[0] == descriptor:
                return cached[1]
            transport = self._build_transport(profile)
            self._transports[profile.name] = (descriptor, transport)
            return transport

    def _build_transport(self, profile: ClusterProfile) -> Transport:
        if profile.transport == "local":
            return LocalTransport()
        if profile.auth.mode == "password":
            return PasswordSshTransport(
                profile.host or "",
                user=profile.user,
                port=profile.port,
                timeout=profile.connection.connect_timeout,
                auth_timeout=profile.connection.auth_timeout,
                banner_timeout=profile.connection.banner_timeout,
                keepalive=profile.connection.keepalive,
                command_timeout=profile.connection.command_timeout,
                known_hosts=profile.known_hosts,
                password_provider=lambda: self.credentials.resolve(profile),
            )
        return SshTransport(
            profile.host or "",
            options=profile.ssh_options,
            user=profile.user,
            port=profile.port,
            connection=profile.connection,
        )

    def scheduler(self, profile: ClusterProfile, transport: Transport) -> Scheduler:
        """Build the configured scheduler over the selected transport."""
        if profile.scheduler == "local":
            return ProcessScheduler(transport, profile)
        return SlurmScheduler(transport, profile=cast(SlurmProfile, profile.slurm_profile))

    def environment_provider(self, profile: ClusterProfile) -> EnvironmentProvider:
        """Build the explicit existing/managed environment policy."""
        if profile.environment == "managed":
            return ManagedEnvironmentProvider()
        return ExistingEnvironmentProvider()
