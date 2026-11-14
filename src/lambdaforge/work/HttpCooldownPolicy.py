"""Public bounded HTTP cooldown parsing; headers always take precedence over local retries."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any


@dataclass(frozen=True)
class HttpCooldownPolicy:
    """Optionally declare a provider JSON delay field (seconds), never infer empty success."""

    json_field: str | None = None

    def delay(self, headers: Any, body: bytes = b"", *, now: float | None = None) -> float:
        """Return the largest explicit finite nonnegative Retry-After delay."""
        now = time.time() if now is None else now
        delays = [0.0]
        raw = headers.get("Retry-After") if headers else None
        if raw is not None:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                try:
                    value = parsedate_to_datetime(raw).timestamp() - now
                except (TypeError, ValueError, OverflowError):
                    value = 0.0
            if math.isfinite(value):
                delays.append(max(0.0, value))
        if self.json_field and body:
            try:
                value = json.loads(body).get(self.json_field)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, int | float)
                    and math.isfinite(value)
                ):
                    delays.append(max(0.0, float(value)))
            except (ValueError, AttributeError, UnicodeError):
                pass  # Malformed advisory JSON never changes the failed HTTP outcome.
        return max(delays)
