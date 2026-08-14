"""Backward-compatible name for the durable process scheduler."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ProcessScheduler import ProcessScheduler
from lambdaforge.controlplane.Transport import Transport


class LocalScheduler(ProcessScheduler):
    """Provide durable asynchronous execution on the local host."""

    def __init__(self, transport: Transport, profile: ClusterProfile | None = None) -> None:
        super().__init__(transport, profile or ClusterProfile("local"))
