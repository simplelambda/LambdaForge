"""Installed versioned authoring contracts, not a second configuration normalizer."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Literal


class ConfigurationSchemas:
    """Metadata-only schema discovery; native Python classes retain effective defaults.

    Work.run scientific signatures cannot be replaced by static JSON. Work validation imports
    selected consumer Python and requires the existing operator/session authorization.
    """

    @staticmethod
    def schema(kind: Literal["work", "cluster"] = "work") -> dict[str, Any]:
        if kind not in {"work", "cluster"}:
            raise ValueError("Schema kind must be work or cluster.")
        filename = "work.schema.json" if kind == "work" else "cluster-profile-v1.json"
        value = json.loads(files("lambdaforge.schemas").joinpath(filename).read_text())
        value["x-contract-version"] = "work-config-v1" if kind == "work" else "cluster-profile-v1"
        return value

    @staticmethod
    def resolve_cluster(name: str, value: dict[str, Any]) -> dict[str, Any]:
        """Schema check plus the actual native normalizer; never connect, claim or submit."""
        from jsonschema import Draft202012Validator

        from lambdaforge.controlplane.ClusterProfile import ClusterProfile

        Draft202012Validator(ConfigurationSchemas.schema("cluster")).validate(value)
        return ClusterProfile.from_mapping(name, value).to_dict()
