"""Canonical documentation, examples and local-link regression checks."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from lambdaforge.cli import CommandLineInterface
from lambdaforge.configuration import AuthoringConfig, ConfigurationKind
from lambdaforge.controlplane import ClusterCatalog
from lambdaforge.data import DataCatalog, DatasetRecipe
from lambdaforge.experiments import ExperimentValidator
from lambdaforge.tasks import TaskRun
from lambdaforge.workflows import Workflow

ROOT = Path(__file__).parents[1]


def _markdown_anchors(text: str) -> set[str]:
    """Approximate GitHub heading anchors, including deterministic duplicate suffixes."""
    anchors: set[str] = set()
    occurrences: dict[str, int] = {}
    for line in text.splitlines():
        if not re.match(r"^#{1,6} ", line):
            continue
        heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        heading = re.sub(r"<[^>]+>", "", heading).replace("`", "").lower()
        base = re.sub(r"[^\w\- ]", "", heading)
        base = re.sub(r"\s+", "-", base).strip("-")
        duplicate = occurrences.get(base, 0)
        occurrences[base] = duplicate + 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return anchors


def test_every_local_markdown_link_resolves() -> None:
    """Check local paths without pretending to validate remote URLs."""
    missing: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if {".git", ".venv", "build", "dist"}.intersection(document.parts):
            continue
        text = document.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip()
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = target.split("#", 1)[0].strip("<>")
            if relative and not (document.parent / relative).resolve().exists():
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{document.relative_to(ROOT)}:{line}: {target}")
    assert not missing, "Missing local documentation targets:\n" + "\n".join(missing)


def test_every_local_markdown_anchor_resolves() -> None:
    """Catch stale section links when the canonical manual is reorganized."""
    missing: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if {".git", ".venv", "build", "dist"}.intersection(document.parts):
            continue
        text = document.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = match.group(1).strip().strip("<>")
            if "#" not in target or target.startswith(("http://", "https://", "mailto:")):
                continue
            relative, anchor = target.split("#", 1)
            target_path = (document.parent / relative).resolve() if relative else document.resolve()
            if target_path.suffix.lower() != ".md" or anchor in _markdown_anchors(
                target_path.read_text(encoding="utf-8")
            ):
                continue
            line = text.count("\n", 0, match.start()) + 1
            missing.append(f"{document.relative_to(ROOT)}:{line}: {target}")
    assert not missing, "Missing local documentation anchors:\n" + "\n".join(missing)


def test_documentation_has_one_landing_page_and_one_manual() -> None:
    """Prevent release history and package notes from fragmenting the product guide again."""
    assert 150 <= len((ROOT / "README.md").read_text(encoding="utf-8").splitlines()) <= 300
    assert sorted(path.name for path in (ROOT / "docs").glob("*.md")) == ["MANUAL.md"]
    assert not list((ROOT / "src" / "lambdaforge").rglob("README*.md"))
    assert not list(ROOT.glob("*.es.md"))


def test_manual_covers_the_supported_product_domains() -> None:
    """Keep normal user and maintainer questions searchable in one document."""
    manual = (ROOT / "docs" / "MANUAL.md").read_text(encoding="utf-8")
    required = {
        "## 2. Installation",
        "## 6. Friendly authoring",
        "## 7. Generic tasks and preprocessing",
        "## 9. Workflows",
        "## 10. Local and multi-cluster control plane",
        "## 11. Persistent jobs and data placement",
        "### Adaptive experiment optimization",
        "## 14. Artifact stores, registry and reports",
        "## 16. CLI reference",
        "## 17. Public API",
        "## 19. Architecture",
        "## 21. Configuration migrations",
        "## 22. Execution and process safety",
        "## 24. Artifact retention",
        "## 26. Extension contracts",
        "## 27. Security model",
        "## 28. Current limitations",
    }
    assert not sorted(value for value in required if value not in manual)
    assert "Version 0.6" not in manual
    assert "LambdaForge 0.5" not in manual


def test_every_canonical_yaml_example_uses_a_real_loader_and_valid_schema() -> None:
    """Validate templates without requiring a hypothetical consumer package to be installed."""
    examples = ROOT / "examples"
    DataCatalog.from_yaml(examples / "data-catalog.yaml")
    assert "atlas" in ClusterCatalog.load(examples / "lambdaforge.clusters.yaml").names()

    for path in sorted(examples.glob("*.yaml")):
        if path.name in {"data-catalog.yaml", "lambdaforge.clusters.yaml"}:
            continue
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), path
        materialized = AuthoringConfig.from_yaml(path).materialize()
        if materialized.kind is ConfigurationKind.DATASET:
            report = DatasetRecipe.from_yaml(path).validate(check_imports=False)
        elif materialized.kind is ConfigurationKind.WORKFLOW:
            report = Workflow.from_yaml(path).validate(check_imports=False)
        elif materialized.kind is ConfigurationKind.TASK:
            report = TaskRun.from_yaml(path).validate(check_imports=False)
        else:
            report = ExperimentValidator().validate_file(path, check_imports=False)
        assert report.is_valid, f"{path.name}: {report.summary()}"


def test_agent_guide_is_operational_and_points_to_the_manual() -> None:
    """Keep the agent entry point useful without duplicating the full manual."""
    agent = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agent.splitlines()) < 300
    for text in (
        "## Fast routes",
        "## Dataset lifecycle",
        "## Training, post-run and HPO",
        "## Clusters, environments and jobs",
        "## Results and publication discipline",
        "## Repository modification rules",
        "docs/MANUAL.md",
    ):
        assert text in agent


def test_documented_top_level_commands_exist() -> None:
    """Reject examples that name a removed top-level CLI command."""
    parser = CommandLineInterface._parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    supported = set(command_action.choices) | {"plan"}
    documented: set[str] = set()
    for path in (ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "docs" / "MANUAL.md"):
        documented.update(
            re.findall(
                r"^(?:\$\s+)?(?:lf|lambdaforge)\s+([a-z][a-z-]*)\b",
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
        )
    assert documented <= supported, f"Unknown documented commands: {sorted(documented - supported)}"


def test_behavior_tests_are_not_named_after_releases() -> None:
    """Keep test names stable across package releases."""
    release_named = [
        path.relative_to(ROOT)
        for path in ROOT.joinpath("tests").rglob("test*.py")
        if re.search(r"(?:roadmap|_0\d{2})(?:_|\.py)", path.name)
    ]
    assert not release_named
