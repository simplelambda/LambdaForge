"""Stable scientific identity for one materialized experiment configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class RunFingerprint:
    """Hash only settings that can affect a run's scientific result.

    Output locations, retry/resume controls, aggregation, retention and descriptive
    metadata deliberately do not change the identity. Expanded model, data, task,
    loss, metric, optimizer, trainer and extension settings do.
    """

    VERSION = 1
    _OPERATIONAL_SECTIONS = frozenset(
        {"$schema", "schema_version", "execution", "aggregation", "retention", "metadata"}
    )
    _OPERATIONAL_EXPERIMENT_KEYS = frozenset(
        {
            "base_name",
            "ckpt_path",
            "name",
            "output_root",
            "required_artifacts",
            "rerun_completed",
            "resume",
            "seeds",
            "variant",
        }
    )

    @classmethod
    def digest(cls, config: Mapping[str, Any]) -> str:
        """Return a versioned SHA-256 digest for a concrete run mapping."""
        canonical = json.dumps(
            cls.payload(config),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    @classmethod
    def payload(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """Return the normalized scientific payload used by :meth:`digest`."""
        adaptive_max_budget: Any = None
        metadata = config.get("metadata")
        if isinstance(metadata, Mapping):
            adaptive = metadata.get("adaptive")
            if isinstance(adaptive, Mapping):
                adaptive_max_budget = adaptive.get("max_budget")
        selected = {
            str(key): value
            for key, value in config.items()
            if str(key) not in cls._OPERATIONAL_SECTIONS and str(key) != "sweep"
        }
        bindings = None
        extensions = selected.get("extensions")
        if isinstance(extensions, Mapping):
            cleaned_extensions = dict(extensions)
            bindings = cleaned_extensions.pop("_lambdaforge_dataset_bindings", None)
            authoring = cleaned_extensions.get("authoring")
            if isinstance(authoring, Mapping):
                scientific_authoring = {
                    str(key): value
                    for key, value in authoring.items()
                    if str(key) not in {"data_catalog", "environment", "resources"}
                }
                if scientific_authoring:
                    cleaned_extensions["authoring"] = scientific_authoring
                else:
                    cleaned_extensions.pop("authoring", None)
            selected["extensions"] = cleaned_extensions
        if isinstance(bindings, Sequence):
            selected = copy.deepcopy(selected)
            identities: list[dict[str, Any]] = []
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    continue
                path = binding.get("path")
                reference = binding.get("reference")
                if isinstance(path, str) and isinstance(reference, str):
                    cls._set_path(selected, path, {"dataset_reference": reference})
                    identities.append(
                        {
                            "path": path,
                            "reference": reference,
                            "identity": binding.get("identity", {}),
                        }
                    )
            selected["dataset_identities"] = identities
        experiment = selected.get("experiment")
        if isinstance(experiment, Mapping):
            selected["experiment"] = {
                str(key): value
                for key, value in experiment.items()
                if str(key) not in cls._OPERATIONAL_EXPERIMENT_KEYS
            }
        trainer = selected.get("trainer")
        if adaptive_max_budget is not None and isinstance(trainer, Mapping):
            selected["trainer"] = dict(trainer)
            selected["trainer"]["max_epochs"] = adaptive_max_budget
        normalized = cls._normalize(selected)
        if not isinstance(normalized, dict):
            raise TypeError("A run fingerprint requires a top-level mapping.")
        return {"fingerprint_version": cls.VERSION, "config": normalized}

    @staticmethod
    def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
        node = config
        parts = path.split(".")
        for part in parts[:-1]:
            current = node.get(part)
            if not isinstance(current, dict):
                return
            node = current
        if parts[-1] in node:
            node[parts[-1]] = value

    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._normalize(item) for item in value]
        if isinstance(value, Enum):
            return cls._normalize(value.value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)
