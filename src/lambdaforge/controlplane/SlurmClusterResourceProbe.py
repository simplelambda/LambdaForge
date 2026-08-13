"""SLURM-aware resource observation layered over direct host facts."""

from __future__ import annotations

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.ClusterResourceProbe import ClusterResourceProbe
from lambdaforge.controlplane.DirectClusterResourceProbe import DirectClusterResourceProbe
from lambdaforge.controlplane.ResourceSnapshot import ResourceSnapshot
from lambdaforge.controlplane.Transport import Transport


class SlurmClusterResourceProbe(ClusterResourceProbe):
    """Report scheduler partitions/queue separately from login-node observations."""

    def probe(self, profile: ClusterProfile, transport: Transport) -> ResourceSnapshot:
        direct = DirectClusterResourceProbe().probe(profile, transport)
        partitions = transport.run(
            ("sinfo", "--noheader", "--format=%P|%a|%D|%C|%G|%m"), timeout=20.0
        )
        queue = transport.run(
            ("squeue", "--noheader", "--me", "--format=%i|%T|%P|%C|%m|%M|%l"),
            timeout=20.0,
        )
        scheduler_view = {
            "mode": "slurm",
            "partitions": [line for line in partitions.stdout.splitlines() if line.strip()]
            if partitions.returncode == 0
            else [],
            "user_jobs": [line for line in queue.stdout.splitlines() if line.strip()]
            if queue.returncode == 0
            else [],
            "errors": [
                value
                for value in (
                    partitions.stderr.strip() if partitions.returncode else "",
                    queue.stderr.strip() if queue.returncode else "",
                )
                if value
            ],
            "note": "Login-node usage and schedulable partition capacity are distinct.",
        }
        return ResourceSnapshot(
            direct.cluster,
            direct.online,
            direct.scheduler,
            direct.observed,
            direct.available,
            scheduler_view,
            direct.requested,
            direct.observed_at_utc,
            direct.error,
        )
