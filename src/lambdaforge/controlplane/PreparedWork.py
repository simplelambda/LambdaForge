"""Owned preparation context for execution-only provider entrypoints."""

from dataclasses import dataclass

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ExecutionBundle import ExecutionBundle
from lambdaforge.controlplane.Transport import Transport


@dataclass(frozen=True, slots=True)
class PreparedWork:
    """The same bundle/environment/workspace prepared by ordinary ControlPlane submission.

    An internal entrypoint builder returns only module argv. ControlPlane retains interpreter,
    TLS/environment assignments and site GPU wrapping, so a shard cannot bypass those policies.
    This object is not exposed to consumer Work code.
    """

    profile: ClusterProfile
    transport: Transport
    bundle: ExecutionBundle
    python: str
    config: str
    work_dir: str
    job_id: str
    dry_run: bool
