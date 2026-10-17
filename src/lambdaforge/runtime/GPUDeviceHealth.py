"""Expiring per-UUID observation circuits; not scientific retry or hardware diagnosis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def reconcile_devices(
    previous: Sequence[Mapping[str, Any]],
    sampled: Sequence[Mapping[str, Any]],
    *,
    now: float,
    initial: float,
    maximum: float,
) -> list[dict[str, Any]]:
    """Preserve siblings; reset only with new qualified measurement of the same UUID."""
    before = {str(row["uuid"]): row for row in previous}
    current = {str(row["uuid"]): row for row in sampled}
    if len(current) != len(sampled) or len(current) > 512:
        raise ValueError("GPU observation identities are duplicate/oversized.")
    rows = []
    for uuid in dict.fromkeys((*current, *before)):
        prior = before.get(uuid, {})
        row = dict(current.get(uuid, prior))
        old = prior.get("circuit", {})
        if row.get("quality") == "measured" and uuid in current:
            row["circuit"] = {"state": "closed", "failures": 0, "retry_after": 0}
        elif row.get("suppressed"):
            row["circuit"] = dict(old)
        else:
            failures = int(old.get("failures", 0)) + 1
            error = row.get("error") if uuid in current else "device_disappeared"
            row.update(
                quality="unavailable", free_bytes=None, compute_process_count=None, processes=[]
            )
            row["circuit"] = {
                "state": "open",
                "failures": failures,
                "reason": error or "nvml_unavailable",
                "scope": "device-observation",
                "opened_at": now,
                "retry_after": now + min(maximum, initial * 2 ** min(10, failures - 1)),
            }
        rows.append(row)
    return rows


def suppressed_devices(devices: Sequence[Mapping[str, Any]], now: float) -> list[dict[str, Any]]:
    """Only cached metadata, no grants or inferred GPU indices."""
    values = []
    for row in devices:
        circuit = row.get("circuit", {})
        if circuit.get("state") == "open" and now < float(circuit.get("retry_after", 0)):
            values.append(
                {
                    key: row.get(key)
                    for key in (
                        "uuid",
                        "index",
                        "parent_uuid",
                        "name",
                        "aliases",
                        "error",
                        "circuit",
                    )
                }
            )
    return values
