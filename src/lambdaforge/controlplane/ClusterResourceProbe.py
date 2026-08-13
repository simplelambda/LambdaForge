"""Extensible cluster resource probe contract."""

from abc import ABC, abstractmethod

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ResourceSnapshot import ResourceSnapshot
from lambdaforge.controlplane.Transport import Transport


class ClusterResourceProbe(ABC):
    """Observe one host/scheduler without claiming stronger enforcement."""

    @abstractmethod
    def probe(self, profile: ClusterProfile, transport: Transport) -> ResourceSnapshot:
        """Return one bounded read-only observation."""
