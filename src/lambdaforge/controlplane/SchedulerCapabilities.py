"""Explicit scheduler control capabilities."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchedulerCapabilities:
    """Prevent callers from pretending unsupported lifecycle operations succeeded."""

    supports_pause: bool = False
    supports_resume: bool = False
    durable: bool = True
    resources_released_when_paused: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "supports_pause": self.supports_pause,
            "supports_resume": self.supports_resume,
            "durable": self.durable,
            "resources_released_when_paused": self.resources_released_when_paused,
        }
