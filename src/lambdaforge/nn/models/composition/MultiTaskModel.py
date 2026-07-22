"""Shared-backbone multi-task model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from lambdaforge.nn.models.Model import Model


class MultiTaskModel(Model):
    """Apply named task heads to one shared backbone representation.

    ``heads`` is a YAML-friendly mapping from stable task names to arbitrary
    modules. Tasks listed in ``detach_backbone_for`` receive detached shared
    features, allowing a head to train without updating the backbone. A
    mapping output is intentional because tasks may have unrelated shapes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        heads: Mapping[str, nn.Module],
        detach_backbone_for: Sequence[str] | None = None,
        return_features: bool = False,
        features_key: str = "features",
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module.")
        if not isinstance(heads, Mapping):
            raise TypeError("heads must be a mapping from task names to modules.")
        if not heads:
            raise ValueError("heads cannot be empty.")
        if any(not isinstance(name, str) or not name or "." in name for name in heads):
            raise ValueError("Every task name must be non-empty and cannot contain '.'.")
        if any(not isinstance(head, nn.Module) for head in heads.values()):
            raise TypeError("Every task head must be a torch.nn.Module.")
        if not isinstance(features_key, str) or not features_key:
            raise ValueError("features_key must be a non-empty string.")
        if return_features and features_key in heads:
            raise ValueError("features_key cannot collide with a task name.")
        if detach_backbone_for is not None and (
            not isinstance(detach_backbone_for, Sequence)
            or isinstance(detach_backbone_for, str | bytes)
        ):
            raise TypeError("detach_backbone_for must be a sequence of task names or None.")

        detached = set(detach_backbone_for or ())
        unknown = detached - set(heads)
        if unknown:
            raise ValueError(f"Unknown detached task names: {sorted(unknown)}.")

        self.backbone = backbone
        self.heads = nn.ModuleDict(dict(heads))
        self.detach_backbone_for = detached
        self.return_features = return_features
        self.features_key = features_key

    def shared_features(self, *args: Any, **kwargs: Any) -> Tensor:
        """Return the validated shared backbone representation."""
        features = self.backbone(*args, **kwargs)
        if not isinstance(features, Tensor):
            raise TypeError("backbone must return a Tensor.")
        return features

    def forward_task(self, task: str, *args: Any, **kwargs: Any) -> Tensor:
        """Evaluate one named task without computing unused heads."""
        if task not in self.heads:
            raise KeyError(f"Unknown task {task!r}; available tasks: {list(self.heads)}.")
        features = self.shared_features(*args, **kwargs)
        if task in self.detach_backbone_for:
            features = features.detach()
        output = self.heads[task](features)
        if not isinstance(output, Tensor):
            raise TypeError(f"Task head {task!r} must return a Tensor.")
        return output

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Return a mapping containing all task predictions."""
        features = self.shared_features(*args, **kwargs)
        outputs: dict[str, Tensor] = {}
        for task, head in self.heads.items():
            task_features = features.detach() if task in self.detach_backbone_for else features
            output = head(task_features)
            if not isinstance(output, Tensor):
                raise TypeError(f"Task head {task!r} must return a Tensor.")
            outputs[task] = output
        if self.return_features:
            outputs[self.features_key] = features
        return outputs
