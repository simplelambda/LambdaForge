"""Backward-compatible name for the durable process scheduler."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.Transport import Transport


class LocalScheduler(ProcessScheduler):
    """Preserve the public 0.5 name while providing durable asynchronous execution."""

    def __init__(self, transport: Transport, profile: ClusterProfile | None = None) -> None:
        super().__init__(transport, profile or ClusterProfile("local"))
