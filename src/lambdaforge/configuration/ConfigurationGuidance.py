"""Installed-schema authoring guidance; never another Work grammar or execution planner."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class ConfigurationGuidance:
    """Static field queries avoid consumer imports; semantic diff uses native WorkConfig."""

    @staticmethod
    def schema() -> dict[str, Any]:
        return json.loads(files("lambdaforge").joinpath("schemas/work.schema.json").read_text())

    @staticmethod
    def _read(path: str | Path) -> dict[str, Any]:
        source = Path(path).expanduser()
        if source.stat().st_size > 1024**2:
            raise ValueError("Authoring inspection requires a YAML of at most 1 MiB.")
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Work YAML requires an object.")
        return value

    @classmethod
    def explain_field(cls, path: str | Path, field: str) -> dict[str, Any]:
        """Describe an exact installed schema field without importing Work or running science."""
        schema = cls.schema()
        node = schema
        configured: Any = cls._read(path)
        present = True
        for part in field.split("."):
            if not part or part not in node.get("properties", {}):
                raise ValueError(
                    f"Field {field!r} is not a named static schema field. Work.run parameters "
                    "are signature-owned: use lf explain YAML for trusted native validation."
                )
            node = node["properties"][part]
            if "$ref" in node:
                node = {**schema["$defs"][node["$ref"].split("/")[-1]], **node}
            if isinstance(configured, Mapping) and part in configured:
                configured = configured[part]
            else:
                present = False
                configured = None
        normalized: Any = None
        normalization = "not resolved: static schema guidance only"
        # This is the native policy normalizer, not a second handwritten set of defaults.
        if field.startswith("execution."):
            from lambdaforge.work.design import ExecutionPolicy

            raw = cls._read(path)
            policy = ExecutionPolicy.from_mapping(raw.get("execution")).to_dict()
            normalized = policy[field.split(".")[1]]
            normalization = "native ExecutionPolicy; consumer Python was not imported"
        from lambdaforge.agents.contracts import redact

        return redact(
            {
                "field": field,
                "description": node.get(
                    "description", "See the installed Work schema and lf explain."
                ),
                "type": node.get("type", node.get("oneOf", node.get("anyOf"))),
                "accepted": {
                    key: node[key]
                    for key in (
                        "enum",
                        "const",
                        "minimum",
                        "maximum",
                        "exclusiveMinimum",
                        "pattern",
                    )
                    if key in node
                },
                "configured": configured,
                "configured_present": present,
                "normalized": normalized,
                "normalization": normalization,
                "default": node.get("default"),
                "default_documented": "default" in node,
                "examples": node.get("examples", []),
                "related": node.get("x-lambdaforge-related", []),
                "common_errors": node.get("x-lambdaforge-common-errors", []),
                "project_code_imported": False,
                "schema_id": schema["$id"],
            }
        )

    @classmethod
    def diff(cls, previous: str | Path, proposed: str | Path) -> dict[str, Any]:
        """Native normalized differences, not recovery permission or Study extension policy."""
        from lambdaforge.work.config import WorkConfig
        from lambdaforge.work.extension_design import exact, plain

        cls._read(previous)
        cls._read(proposed)
        # Consumer module imports are trusted but may print; never mix them into JSON stdout.
        with redirect_stdout(sys.stderr):
            old = WorkConfig.from_yaml(previous)
            new = WorkConfig.from_yaml(proposed)
        left, right = old.resolved_configuration(), new.resolved_configuration()
        for config, snapshot in ((old, left), (new, right)):
            snapshot["work_definitions"] = [
                [
                    {
                        "class": run.work_class,
                        "parameters": plain(run.parameters),
                        "seeds": plain(run.seeds),
                        "analysis_semantics": plain(run.analysis_semantics),
                    }
                    for run in level.runs
                ]
                for level in config.levels
            ]
        changes = []

        def category_for(path: str) -> str:
            if any(token in path.split(".") for token in ("resources", "execution")):
                return "operational"
            if "analysis" in path:
                return "analysis"
            if path.endswith("name") or path.endswith("source"):
                return "display"
            return "scientific"

        def visit(a: Any, b: Any, path: str) -> None:
            if exact(a) == exact(b):
                return
            if isinstance(a, Mapping) and isinstance(b, Mapping):
                for key in sorted(set(a) | set(b)):
                    child_path = f"{path}.{key}" if path else str(key)
                    if key not in a or key not in b:
                        changes.append(
                            {
                                "path": child_path,
                                "category": category_for(child_path),
                                "change": "added" if key not in a else "removed",
                                "previous": plain(a.get(key)),
                                "proposed": plain(b.get(key)),
                            }
                        )
                    else:
                        visit(a[key], b[key], child_path)
                return
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for index, (first, second) in enumerate(zip(a, b, strict=True)):
                    visit(first, second, f"{path}[{index}]")
                return
            changes.append(
                {
                    "path": path,
                    "category": category_for(path),
                    "previous": plain(a),
                    "proposed": plain(b),
                }
            )

        visit(left, right, "")
        for key, category in (("with", "scientific"), ("products", "products")):
            if exact(old.raw.get(key)) != exact(new.raw.get(key)):
                changes.append(
                    {
                        "path": key,
                        "category": category,
                        "previous": plain(old.raw.get(key)),
                        "proposed": plain(new.raw.get(key)),
                    }
                )
        from lambdaforge.agents.contracts import redact

        return redact(
            {
                "previous": str(old.source),
                "proposed": str(new.source),
                "changes": changes,
                "compatible_with_recovery": "not evaluated",
                "compatible_with_extension": (
                    "not evaluated; use the native Study extension preview"
                ),
                "input_content": (
                    "not compared; path/declaration differences are not byte equivalence"
                ),
                "dependencies": (
                    "not attested; YAML diff does not prove executable dependency identity"
                ),
            }
        )
