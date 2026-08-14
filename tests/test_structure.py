"""Stable architectural checks that encourage cohesion instead of file ceremony."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "lambdaforge"


def source_modules() -> list[Path]:
    """Return framework modules in deterministic order."""
    return sorted(SOURCE_ROOT.rglob("*.py"))


def test_every_source_module_has_documentation() -> None:
    """Require each module to identify its cohesive responsibility."""
    offenders: list[str] = []
    for path in source_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree, clean=False) is None:
            offenders.append(str(path.relative_to(SOURCE_ROOT)))
    assert not offenders, "Undocumented modules found:\n" + "\n".join(offenders)


def test_every_public_top_level_class_has_documentation() -> None:
    """Require useful local contracts without prescribing one class per file."""
    offenders: list[str] = []
    for path in source_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ClassDef)
                and not node.name.startswith("_")
                and ast.get_docstring(node, clean=False) is None
            ):
                offenders.append(f"{path.relative_to(SOURCE_ROOT)}: {node.name}")
    assert not offenders, "Undocumented public classes found:\n" + "\n".join(offenders)


def test_no_undifferentiated_utility_modules() -> None:
    """Keep shared helpers named for a responsibility rather than growing a junk drawer."""
    forbidden = {"util.py", "utils.py", "helpers.py", "common_utils.py"}
    offenders = [
        str(path.relative_to(SOURCE_ROOT)) for path in source_modules() if path.name in forbidden
    ]
    assert not offenders, "Undifferentiated utility modules found:\n" + "\n".join(offenders)


def test_internal_packages_do_not_contain_product_readmes() -> None:
    """Keep the canonical manual as the only conceptual documentation source."""
    assert not list(SOURCE_ROOT.rglob("README*.md"))
