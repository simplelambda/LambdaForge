"""Stable scientific identity for one generic task configuration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any


class TaskFingerprint:
    """Hash task logic and extensions while excluding operational settings."""

    VERSION = 1
    _OPERATIONAL_KEYS = frozenset(
        {
            "$schema",
            "schema_version",
            "kind",
            "name",
            "output_root",
            "resume",
            "rerun_completed",
            "required_artifacts",
            "execution",
            "metadata",
        }
    )

    @classmethod
    def digest(cls, config: Mapping[str, Any]) -> str:
        """Return a versioned SHA-256 digest for a materialized task mapping."""
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
        selected = {
            str(key): value
            for key, value in config.items()
            if str(key) not in cls._OPERATIONAL_KEYS
        }
        if "_resolved_inputs" in selected:
            selected.pop("inputs", None)
        task = selected.get("task")
        if (
            isinstance(task, Mapping)
            and task.get("target") == "lambdaforge.preprocessing.PreprocessingTask"
            and isinstance(task.get("params"), Mapping)
        ):
            cleaned_task = dict(task)
            cleaned_task["params"] = {
                str(key): item
                for key, item in task["params"].items()
                if str(key) not in {"workers", "workload", "checkpoint_interval"}
            }
            selected["task"] = cleaned_task
        extensions = selected.get("extensions")
        if isinstance(extensions, Mapping):
            cleaned_extensions = dict(extensions)
            authoring = cleaned_extensions.get("authoring")
            if isinstance(authoring, Mapping):
                scientific_authoring = {
                    str(key): item
                    for key, item in authoring.items()
                    if str(key)
                    not in {
                        "outputs",
                        "resources",
                        "data_catalog",
                        "environment",
                        "code_version",
                    }
                }
                if scientific_authoring:
                    cleaned_extensions["authoring"] = scientific_authoring
                else:
                    cleaned_extensions.pop("authoring", None)
            if cleaned_extensions:
                selected["extensions"] = cleaned_extensions
            else:
                selected.pop("extensions", None)
        normalized = cls._normalize(selected)
        if not isinstance(normalized, dict):
            raise TypeError("A task fingerprint requires a top-level mapping.")
        return {"fingerprint_version": cls.VERSION, "config": normalized}

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
