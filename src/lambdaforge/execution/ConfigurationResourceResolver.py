"""Resolve the scheduler request for current Work YAML."""

from __future__ import annotations

from pathlib import Path

from lambdaforge.execution.ResourceRequest import ResourceRequest
from lambdaforge.work import WorkConfig


class ConfigurationResourceResolver:
    """Return the conservative outer allocation of one Work execution."""

    @staticmethod
    def resolve(source: str | Path) -> ResourceRequest:
        """Validate current YAML and aggregate sequential/parallel resources."""
        return WorkConfig.from_yaml(source).resources
