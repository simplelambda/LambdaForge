"""Optional Weights & Biases logger adapter for LambdaForge experiments."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Literal

from lambdaforge.integrations.Lightning import Lightning, WandbLoggerBase
from lambdaforge.tracking.TrackingBackend import TrackingBackend
from lambdaforge.tracking.TrackingDependencyGuard import TrackingDependencyGuard


class WeightsAndBiasesTrackingLogger(WandbLoggerBase):
    """Configure Lightning's native W&B logger without forcing online mode.

    All native run identity and 'wandb.init' options remain configurable.
    'offline' is deliberately not forced: callers can choose local/offline or
    remote tracking per YAML. Model-checkpoint publication remains disabled by
    default through 'log_model=False'.

    'add_file_policy' is forwarded on Lightning releases that expose it. Its
    native 'mutable' default is omitted on older supported releases;
    requesting 'immutable' there raises a clear compatibility error.
    """

    _SUPPORTS_ADD_FILE_POLICY = (
        "add_file_policy" in inspect.signature(Lightning.WandbLogger.__init__).parameters
    )

    def __init__(
        self,
        name: str | None = None,
        save_dir: str | Path = ".",
        version: str | None = None,
        offline: bool = False,
        dir: str | Path | None = None,
        id: str | None = None,
        anonymous: bool | None = None,
        project: str | None = None,
        log_model: Literal["all"] | bool = False,
        experiment: Any | None = None,
        prefix: str = "",
        checkpoint_name: str | None = None,
        add_file_policy: Literal["mutable", "immutable"] = "mutable",
        **kwargs: Any,
    ) -> None:
        """Validate the optional dependency and initialize the native logger."""
        TrackingDependencyGuard(TrackingBackend.WEIGHTS_AND_BIASES).require()
        arguments: dict[str, Any] = {
            "name": name,
            "save_dir": save_dir,
            "version": version,
            "offline": offline,
            "dir": dir,
            "id": id,
            "anonymous": anonymous,
            "project": project,
            "log_model": log_model,
            "experiment": experiment,
            "prefix": prefix,
            "checkpoint_name": checkpoint_name,
            **kwargs,
        }
        if self._SUPPORTS_ADD_FILE_POLICY:
            arguments["add_file_policy"] = add_file_policy
        elif add_file_policy != "mutable":
            raise TypeError(
                "WeightsAndBiasesTrackingLogger.add_file_policy='immutable' requires "
                "a Lightning release whose WandbLogger exposes that parameter."
            )
        super().__init__(**arguments)
