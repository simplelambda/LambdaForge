"""Executable documentation checks for the only current Work architecture."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from lambdaforge.work import WorkConfig

ROOT = Path(__file__).resolve().parents[1]


def test_all_packaged_work_examples_validate_and_import() -> None:
    for path in (ROOT / "examples").glob("*.yaml"):
        if path.name == "lambdaforge.clusters.yaml":
            continue
        report = WorkConfig.validate_file(path)
        assert report.valid, (path, report.errors)


def test_readme_yaml_uses_the_current_schema() -> None:
    schema = json.loads(
        (ROOT / "src/lambdaforge/schemas/work.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    for document in ("README.md", "README.es.md"):
        text = (ROOT / document).read_text(encoding="utf-8")
        blocks = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
        assert blocks
        for block in blocks:
            validator.validate(yaml.safe_load(block))


def test_normal_documentation_contains_no_superseded_yaml_language() -> None:
    for relative in ("README.md", "README.es.md", "docs/MANUAL.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for removed in ("kind:", "schema_version", "required_artifacts", "bindings:", "needs:"):
            assert removed not in text, (relative, removed)
