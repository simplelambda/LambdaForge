"""Validated declarative specification for one post-run action."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.postrun.PostRunCheckpoint import PostRunCheckpoint
from lambdaforge.tasks.artifacts import ArtifactDeclaration, ArtifactType


@dataclass(frozen=True, slots=True)
class PostRunActionSpec:
    """Separate action policy and identity from the training fingerprint."""

    name: str
    target: str
    params: Mapping[str, Any]
    checkpoint: PostRunCheckpoint
    required: bool
    artifacts: tuple[ArtifactDeclaration, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> PostRunActionSpec:
        """Parse one small YAML action without constructing project code."""
        if not isinstance(value, Mapping):
            raise TypeError(f"post_run.actions[{index}] must be a mapping.")
        unexpected = set(value) - {
            "name",
            "target",
            "params",
            "checkpoint",
            "required",
            "artifacts",
        }
        if unexpected:
            raise ValueError(f"Unexpected post_run.actions[{index}] keys: {sorted(unexpected)}.")
        target = value.get("target")
        if not isinstance(target, str) or "." not in target:
            raise ValueError(f"post_run.actions[{index}].target must be importable.")
        raw_name = value.get("name", f"action-{index + 1}-{target.rsplit('.', 1)[-1]}")
        name = str(raw_name)
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise ValueError(
                f"post_run.actions[{index}].name must contain only letters, digits, _, . or -."
            )
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            raise TypeError(f"post_run.actions[{index}].params must be a mapping.")
        required = value.get("required", True)
        if not isinstance(required, bool):
            raise TypeError(f"post_run.actions[{index}].required must be a bool.")
        raw_artifacts = value.get("artifacts", ())
        if not isinstance(raw_artifacts, Sequence) or isinstance(
            raw_artifacts, (str, bytes, bytearray)
        ):
            raise TypeError(f"post_run.actions[{index}].artifacts must be a list.")
        artifacts = tuple(ArtifactDeclaration.from_value(item) for item in raw_artifacts)
        paths = [artifact.path for artifact in artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError(f"post_run action {name!r} declares one artifact path twice.")
        return cls(
            name=name,
            target=target,
            params=FrozenJsonMapping(params),
            checkpoint=PostRunCheckpoint(str(value.get("checkpoint", "best"))),
            required=required,
            artifacts=artifacts,
        )

    def static_identity(self, training_fingerprint: str) -> str:
        """Hash action code/config policy separately from the trained model."""
        payload = {
            "identity_version": 1,
            "training_fingerprint": training_fingerprint,
            "action": self.to_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical user-controlled action configuration."""
        return {
            "name": self.name,
            "target": self.target,
            "params": copy.deepcopy(dict(self.params)),
            "checkpoint": self.checkpoint.value,
            "required": self.required,
            "artifacts": [
                {
                    **({"name": artifact.name} if artifact.name is not None else {}),
                    "path": artifact.path,
                    "kind": cast(ArtifactType, artifact.kind).value,
                    **(
                        {"media_type": artifact.media_type}
                        if artifact.media_type is not None
                        else {}
                    ),
                    "metadata": copy.deepcopy(dict(artifact.metadata)),
                }
                for artifact in self.artifacts
            ],
        }
