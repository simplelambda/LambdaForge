"""Construct control-plane providers from one cluster profile."""

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.EnvironmentProvider import EnvironmentProvider
from lambdaforge.controlplane.ExistingEnvironmentProvider import ExistingEnvironmentProvider
from lambdaforge.controlplane.LocalScheduler import LocalScheduler
from lambdaforge.controlplane.LocalTransport import LocalTransport
from lambdaforge.controlplane.ManagedEnvironmentProvider import ManagedEnvironmentProvider
from lambdaforge.controlplane.Scheduler import Scheduler
from lambdaforge.controlplane.SlurmScheduler import SlurmScheduler
from lambdaforge.controlplane.SshTransport import SshTransport
from lambdaforge.controlplane.Transport import Transport


class ControlPlaneFactory:
    """Centralize provider selection and keep services dependency-injectable."""

    def transport(self, profile: ClusterProfile) -> Transport:
        """Build the configured transport."""
        if profile.transport == "local":
            return LocalTransport()
        return SshTransport(profile.host or "", options=profile.ssh_options)

    def scheduler(self, profile: ClusterProfile, transport: Transport) -> Scheduler:
        """Build the configured scheduler over the selected transport."""
        if profile.scheduler == "local":
            return LocalScheduler(transport)
        return SlurmScheduler(transport, options=profile.scheduler_options)

    def environment_provider(self, profile: ClusterProfile) -> EnvironmentProvider:
        """Build the explicit existing/managed environment policy."""
        if profile.environment == "managed":
            return ManagedEnvironmentProvider()
        return ExistingEnvironmentProvider()
