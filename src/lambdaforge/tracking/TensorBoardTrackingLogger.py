"""Optional TensorBoard logger adapter for LambdaForge experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lambdaforge.integrations.Lightning import TensorBoardLoggerBase
from lambdaforge.tracking.TrackingBackend import TrackingBackend
from lambdaforge.tracking.TrackingDependencyGuard import TrackingDependencyGuard


class TensorBoardTrackingLogger(TensorBoardLoggerBase):
    """Configure Lightning's native TensorBoard logger from Python or YAML.

    'save_dir' may be a local path or a remote location supported by the
    native logger. Extra keyword arguments are forwarded unchanged to the
    selected TensorBoard summary writer, preserving controls such as
    'max_queue', 'flush_secs' and 'filename_suffix'.
    """

    def __init__(
        self,
        save_dir: str | Path,
        name: str | None = "lightning_logs",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Validate the optional dependency and initialize the native logger."""
        TrackingDependencyGuard(TrackingBackend.TENSORBOARD).require()
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            sub_dir=sub_dir,
            **kwargs,
        )
