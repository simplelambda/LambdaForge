"""Human-readable scientific identity comparison."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IdentityExplanation:
    """Summarize the paths that preserve or change scientific identity."""

    same: bool
    current: str
    previous: str | None
    changes: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable explanation."""
        return {
            "same_scientific_identity": self.same,
            "current": self.current,
            "previous": self.previous,
            "changes": [dict(item) for item in self.changes],
        }

    def summary(self) -> str:
        """Render a concise explanation suitable for terminal users."""
        if self.previous is None:
            return f"Scientific identity: {self.current}\nNo previous configuration was selected."
        if self.same:
            return (
                f"Scientific identity is unchanged ({self.current}); cached success may be reused."
            )
        lines = [f"Scientific identity changed: {self.previous} -> {self.current}"]
        lines.extend(
            f"- {item['path']}: {item.get('previous')!r} -> {item.get('current')!r}"
            for item in self.changes
        )
        return "\n".join(lines)
