"""Native explicit first-use peer SSH trust, shared by CLI and Console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lambdaforge.controlplane.SecretRedactor import SecretRedactor

if TYPE_CHECKING:
    from lambdaforge.data.DatasetService import DatasetService


class SiteSshTrustService:
    """Inspect offered fingerprints and add them only after operator approval."""

    def __init__(self, datasets: DatasetService) -> None:
        self.datasets = datasets

    def inspect(
        self,
        initiator: str,
        peer: str,
        *,
        apply: bool = False,
        fingerprints: list[str] | None = None,
        expected_file: str | None = None,
    ) -> dict[str, Any]:
        host = self.datasets.clusters.get(initiator)
        target = self.datasets.clusters.get(peer)
        if host.transport != "ssh" or target.transport != "ssh" or host.name == target.name:
            raise ValueError("Peer trust requires two distinct SSH clusters.")
        if apply and (not fingerprints or not expected_file):
            raise ValueError("Review and explicitly approve fingerprints and trust file first.")
        transport = self.datasets.factory.transport(host)
        python = self.datasets._python(host.name, transport)
        helper = Path(__file__).with_name("SiteSshTrust.py").read_text(encoding="utf-8")
        result = transport.run(
            (
                python,
                "-c",
                helper,
                "approve" if apply else "inspect",
                json.dumps(
                    {
                        "host": target.host,
                        "port": target.port,
                        "user": target.user,
                        "fingerprints": fingerprints,
                        "expected_file": expected_file,
                    }
                ),
            ),
            timeout=45 if apply else 25,
        )
        if result.returncode:
            raise ValueError(
                "Cannot inspect/approve peer host trust: "
                + SecretRedactor.redact(result.stderr[-2500:])
            )
        if len(result.stdout) > 65536:
            raise ValueError("Peer trust response exceeds its metadata bound.")
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("Invalid peer trust response.")
        return {**value, "initiator": initiator, "peer": peer}
