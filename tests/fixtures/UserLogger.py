"""Example project-defined logger used by YAML extension tests."""

import json
from pathlib import Path
from typing import Any

from lambdaforge.integrations.Lightning import LoggerType


class UserLogger(LoggerType):
    """Write logged scalar dictionaries as JSON lines for integration tests."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)

    @property
    def name(self) -> str:
        """Return the logger family name."""
        return "user_logger"

    @property
    def version(self) -> str:
        """Return a stable test version."""
        return "1"

    def log_hyperparams(self, params: Any) -> None:
        """Accept hyperparameters without imposing a storage schema."""
        del params

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Append one JSON record containing every scalar sent by Lightning."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"step": step, "metrics": dict(metrics)}
        with open(self.output_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    def save(self) -> None:
        """Satisfy the Lightning logger contract."""

    def finalize(self, status: str) -> None:
        """Accept the terminal status supplied by Lightning."""
        del status
