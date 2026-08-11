"""Typed operational execution identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Identify placement and resource policy without changing scientific reuse."""

    digest: str
    payload: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        cluster: str = "local",
        resources: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> ExecutionIdentity:
        """Build an operational identity for jobs and audit records."""
        payload = {
            "execution_identity_version": 1,
            "cluster": cluster,
            "resources": dict(resources or {}),
            "environment": dict(environment or {}),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(f"sha256:{hashlib.sha256(encoded).hexdigest()}", payload)
