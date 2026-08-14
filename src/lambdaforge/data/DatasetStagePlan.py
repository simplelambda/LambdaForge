"""One recipe-stage reuse decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetStagePlan:
    """Explain whether one stage is reused, executed, missing or invalid."""

    stage: str
    action: str
    reason: str
    fingerprint: str | None = None
    required: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return stable plan data for human/JSON renderers."""
        return {
            "stage": self.stage,
            "action": self.action,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
            "required": self.required,
        }
