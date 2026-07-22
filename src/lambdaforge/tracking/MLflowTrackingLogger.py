"""Optional MLflow logger adapter for LambdaForge experiments."""

from __future__ import annotations

import inspect
import os
from typing import Any, Literal

from lambdaforge.integrations.Lightning import Lightning, MLFlowLoggerBase
from lambdaforge.tracking.TrackingBackend import TrackingBackend
from lambdaforge.tracking.TrackingDependencyGuard import TrackingDependencyGuard


class MLflowTrackingLogger(MLFlowLoggerBase):
    """Configure Lightning's native MLflow logger without a mandatory service.

    The adapter keeps every constructor option explicit so it can be represented
    directly in experiment YAML. With no 'tracking_uri', Lightning uses the
    'MLFLOW_TRACKING_URI' environment variable when present and otherwise
    stores runs below 'save_dir'. Checkpoint uploads remain opt-in through
    'log_model'.

    Parameters mirror Lightning 2.2+ and include the later 'synchronous'
    option. On an older supported Lightning release, leaving 'synchronous'
    unset preserves compatibility; requesting it produces a version-specific
    error instead of silently discarding the setting.
    """

    _SUPPORTS_SYNCHRONOUS = (
        "synchronous" in inspect.signature(Lightning.MLFlowLogger.__init__).parameters
    )

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        synchronous: bool | None = None,
    ) -> None:
        """Validate the optional dependency and initialize the native logger."""
        TrackingDependencyGuard(TrackingBackend.MLFLOW).require()
        resolved_tracking_uri = (
            tracking_uri if tracking_uri is not None else os.getenv("MLFLOW_TRACKING_URI")
        )
        arguments: dict[str, Any] = {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "tracking_uri": resolved_tracking_uri,
            "tags": tags,
            "save_dir": save_dir,
            "log_model": log_model,
            "prefix": prefix,
            "artifact_location": artifact_location,
            "run_id": run_id,
        }
        if self._SUPPORTS_SYNCHRONOUS:
            arguments["synchronous"] = synchronous
        elif synchronous is not None:
            raise TypeError(
                "MLflowTrackingLogger.synchronous requires a Lightning release "
                "whose MLFlowLogger exposes that parameter."
            )
        super().__init__(**arguments)
