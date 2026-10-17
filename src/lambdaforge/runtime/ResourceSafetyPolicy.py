"""Explicit operational settings, separate from scientific identity and authored ceilings."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceSafetyPolicy:
    profile: str = "balanced"
    telemetry_ttl: float = 5.0
    admission_ttl: float = 0.5
    minimum_interval: float = 0.25
    probe_timeout: float = 5.0
    backoff_initial: float = 2.0
    backoff_maximum: float = 60.0
    startup_seconds: float = 60.0
    fairness_wait_seconds: float = 30.0
    waiter_ttl_seconds: float = 120.0
    priority: float = 1.0
    device_backoff_initial: float = 5.0
    device_backoff_maximum: float = 120.0

    def __post_init__(self) -> None:
        if self.profile not in {"balanced", "throughput", "shared-host", "conservative"}:
            raise ValueError("Unknown resource_safety profile.")
        for key, value in asdict(self).items():
            if key != "profile" and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"resource_safety.{key} must be finite and positive.")
        if not self.minimum_interval <= self.admission_ttl <= self.telemetry_ttl:
            raise ValueError("minimum_interval <= admission_ttl <= telemetry_ttl is required.")
        if self.backoff_initial > self.backoff_maximum:
            raise ValueError("Resource observer initial backoff exceeds maximum.")
        if self.device_backoff_initial > self.device_backoff_maximum:
            raise ValueError("Resource device initial backoff exceeds maximum.")
        if self.waiter_ttl_seconds <= self.fairness_wait_seconds:
            raise ValueError("Resource waiter TTL must exceed fairness wait.")
        if self.probe_timeout > 30 or self.backoff_maximum > 3600 or self.startup_seconds > 3600:
            raise ValueError("Resource operational timeouts/windows exceed safety bounds.")

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | str | None) -> ResourceSafetyPolicy:
        if value is None:
            return cls()
        if isinstance(value, str):
            value = {"profile": value}
        if not isinstance(value, Mapping):
            raise TypeError("resource_safety must be a profile name or mapping.")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown resource_safety fields: {sorted(unknown)}.")
        profile = str(value.get("profile", "balanced"))
        defaults: dict[str, Any] = {
            "balanced": {},
            "throughput": {"telemetry_ttl": 3.0, "fairness_wait_seconds": 20.0},
            "shared-host": {"telemetry_ttl": 10.0, "fairness_wait_seconds": 15.0},
            "conservative": {"startup_seconds": 120.0, "device_backoff_initial": 10.0},
        }.get(profile, {})
        return cls(**{**defaults, **dict(value), "profile": profile})

    @classmethod
    def inherited(cls) -> ResourceSafetyPolicy:
        raw = os.environ.get("LAMBDAFORGE_RESOURCE_SAFETY")
        if raw is None:
            return cls()
        if len(raw) > 8192:
            raise ValueError("Resource policy exceeds operational metadata limit.")
        return cls.from_value(json.loads(raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
