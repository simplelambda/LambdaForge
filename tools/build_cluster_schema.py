"""Generate installed cluster metadata from actual default native policy objects.

Metadata generation only: no provider, connections, claims or scientific imports. The native
normalizer remains the authority for cross-field constraints, including unsupported claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.controlplane.python_runtime import PythonRuntimePolicy
from lambdaforge.controlplane.TorchInstallationPolicy import TorchInstallationPolicy
from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy


def shape(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": ["string", "null"], "default": None}
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {k: shape(v) for k, v in value.items()},
            "additionalProperties": not bool(value),
        }
    if isinstance(value, list):
        return {"type": "array", "items": {"type": "string"}, "default": value}
    return {
        "type": "boolean"
        if type(value) is bool
        else "integer"
        if type(value) is int
        else "number"
        if type(value) is float
        else "string",
        "default": value,
    }


def build() -> dict[str, Any]:
    defaults = ClusterProfile("schema").to_dict()
    slurm = ClusterProfile("schema", scheduler="slurm").to_dict()
    defaults.pop("name")
    schema = shape(defaults)
    p = schema["properties"]
    for key in ("resource_mapping", "scheduler_directives", "scheduler_commands", "job_script"):
        p[key] = shape(slurm[key])
    p["transport"]["enum"], p["scheduler"]["enum"] = ["local", "ssh"], ["local", "slurm"]
    p["environment"]["enum"] = ["existing", "managed"]
    p["port"].update(minimum=1, maximum=65535)
    p["connection"]["properties"]["command_timeout"] = {
        "type": ["number", "null"],
        "exclusiveMinimum": 0,
        "default": None,
        "x-unit": "seconds",
    }
    p["pytorch"]["properties"]["channel"]["enum"] = sorted(TorchInstallationPolicy.CHANNELS)
    p["pytorch"]["properties"]["require_cuda"] = {
        "enum": [True, False, None, "auto"],
        "default": "auto",
    }
    p["pytorch"] = {"oneOf": [p["pytorch"], {"enum": sorted(TorchInstallationPolicy.CHANNELS)}]}
    p["storage"]["properties"]["lease_root"] = {"type": ["string", "null"], "default": None}
    p["auth"]["properties"]["mode"]["enum"] = ["openssh", "password"]
    p["auth"]["properties"]["credential"].update(
        pattern=r"^(env:|keyring:).+", **{"x-secret-reference": True}
    )
    p["auth"] = {"oneOf": [p["auth"], {"enum": ["openssh", "password"]}]}
    gpu = p["gpu_access"]
    gpu["properties"]["mode"]["enum"] = ["auto", "exclusive", "shared", "scheduler", "command"]
    gpu["allOf"] = [
        {
            "if": {"properties": {"mode": {"const": "command"}}, "required": ["mode"]},
            "then": {
                "required": ["command_prefix"],
                "properties": {"command_prefix": {"minItems": 1}},
            },
        }
    ]
    gpu["description"] = (
        "Provider owns grants. Bare claim/release argv is not a supported "
        "owned reservation contract."
    )
    p["gpu_access"] = {"oneOf": [gpu, {"enum": ["auto", "exclusive", "shared", "scheduler"]}]}
    safety = p["resource_safety"]
    safety["properties"]["profile"]["enum"] = [
        "balanced",
        "throughput",
        "shared-host",
        "conservative",
    ]
    for key, item in safety["properties"].items():
        if key != "profile":
            item["exclusiveMinimum"] = 0
            item["x-unit"] = "relative-weight" if key == "priority" else "seconds"
    p["resource_safety"] = {
        "oneOf": [safety, {"enum": ["balanced", "throughput", "shared-host", "conservative"]}]
    }
    p["resource_safety"]["x-profile-defaults"] = {
        name: ResourceSafetyPolicy.from_value(name).to_dict()
        for name in ("balanced", "throughput", "shared-host", "conservative")
    }
    for key in ("cache_max_size",):
        p["storage"]["properties"][key] = {
            "type": ["number", "string", "null"],
            "x-unit": "bytes",
            "default": None,
        }
    p["storage"]["properties"]["cache_max_age"] = {
        "type": ["number", "string", "null"],
        "x-unit": "seconds",
        "default": None,
    }
    p["storage"]["properties"]["safety"]["properties"]["min_free"] = {
        "type": ["number", "string"],
        "x-unit": "bytes",
        "default": 0,
    }
    p["storage"]["properties"]["terminal_jobs"]["properties"]["grace_period"] = {
        "type": ["number", "string"],
        "x-unit": "seconds",
        "default": 172800,
    }
    runtime = shape(PythonRuntimePolicy.from_value({}).to_dict())
    runtime["properties"]["strategy"]["enum"] = ["auto", "existing", "managed"]
    p["python"] = {
        "oneOf": [
            {"type": "string", "default": "python"},
            runtime,
        ]
    }
    p["data_environment"]["default"] = None
    for key in p["scheduler_commands"]["properties"]:
        command = shape({"command": "command", "args": [], "job_id_pattern": None})
        command["properties"]["arguments"] = {"type": "array", "items": {"type": "string"}}
        command["properties"]["executable"] = {"type": "string"}
        p["scheduler_commands"]["properties"][key] = {
            "oneOf": [
                command,
                {"type": "string"},
                {"type": "array", "minItems": 1, "items": {"type": "string"}},
                {"type": "null"},
            ]
        }
    for rule in p["resource_mapping"]["properties"].values():
        rule["properties"].update(
            {"omit": {"type": "boolean"}, "option": {"type": "string"}, "value": {"type": "string"}}
        )
    for key in ("state_root", "cache_root", "run_root"):
        p["storage"]["properties"][key]["x-default-expression"] = (
            "workspace/.lambdaforge/"
            + {"state_root": "state", "cache_root": "cache", "run_root": "jobs"}[key]
        )
    schema.update(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://simplelambda.github.io/lambdaforge/schemas/cluster-profile-v1.json",
            "title": "Native LambdaForge ClusterProfile YAML",
            "x-contract-version": "cluster-profile-v1",
            "x-native-authority": "ClusterProfile.from_mapping",
            "allOf": [
                {
                    "if": {
                        "properties": {"transport": {"const": "ssh"}},
                        "required": ["transport"],
                    },
                    "then": {
                        "required": ["host", "workspace"],
                        "properties": {
                            "workspace": {"type": "string", "pattern": "^/"},
                            "host": {"type": "string", "minLength": 1},
                        },
                    },
                }
            ],
        }
    )
    return schema


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "src/lambdaforge/schemas/cluster-profile-v1.json"
    path.write_text(json.dumps(build(), indent=2) + "\n")
