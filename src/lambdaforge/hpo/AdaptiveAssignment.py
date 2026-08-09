"""Immutable adaptive action-to-resource assignment."""

from dataclasses import dataclass

from lambdaforge.hpo.AdaptiveAction import AdaptiveAction
from lambdaforge.hpo.AdaptiveResource import AdaptiveResource


@dataclass(frozen=True, slots=True)
class AdaptiveAssignment:
    """Bind one admitted action to one logical resource."""

    action: AdaptiveAction
    resource: AdaptiveResource

    def to_dict(self) -> dict[str, object]:
        """Return a structured scheduling decision."""
        return {
            "action": self.action.to_dict(),
            "resource": self.resource.name,
            "device": self.resource.device,
        }
