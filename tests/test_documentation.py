"""Documentation completeness and local-link regression tests."""

import re
from pathlib import Path


class TestDocumentation:
    """Keep human and agent entry points navigable as public capabilities grow."""

    ROOT = Path(__file__).parents[1]

    def test_every_local_markdown_link_resolves(self) -> None:
        missing: list[str] = []
        for document in sorted(self.ROOT.rglob("*.md")):
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
                    missing.append(f"{document.relative_to(self.ROOT)}:{line}: {target}")
        assert not missing, "Missing local documentation targets:\n" + "\n".join(missing)

    def test_human_and_agent_guides_cover_every_public_namespace(self) -> None:
        guides = [
            (self.ROOT / "README.md").read_text(encoding="utf-8"),
            (self.ROOT / "README.es.md").read_text(encoding="utf-8"),
            (self.ROOT / "AGENTS.md").read_text(encoding="utf-8"),
            (self.ROOT / "AGENTS.es.md").read_text(encoding="utf-8"),
        ]
        namespaces = {
            "lambdaforge.configuration",
            "lambdaforge.controlplane",
            "lambdaforge.data",
            "lambdaforge.execution",
            "lambdaforge.experiments",
            "lambdaforge.hpo",
            "lambdaforge.integrations",
            "lambdaforge.results",
            "lambdaforge.visualization",
            "lambdaforge.artifacts",
            "lambdaforge.metrics",
            "lambdaforge.nn",
            "lambdaforge.observability",
            "lambdaforge.operations",
            "lambdaforge.plugins",
            "lambdaforge.preprocessing",
            "lambdaforge.registry",
            "lambdaforge.reproducibility",
            "lambdaforge.storage",
            "lambdaforge.tasks",
            "lambdaforge.tracking",
            "lambdaforge.training",
            "lambdaforge.workflows",
        }
        for guide in guides:
            assert namespaces <= {name for name in namespaces if name in guide}

    def test_readme_indices_reference_existing_headings(self) -> None:
        for document in sorted(self.ROOT.rglob("README*.md")):
            if ".git" in document.parts:
                continue
            text = document.read_text(encoding="utf-8")
            headings: set[str] = set()
            fenced = False
            for line in text.splitlines():
                if line.startswith("```"):
                    fenced = not fenced
                    continue
                if fenced or (match := re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)) is None:
                    continue
                label = re.sub(r"<[^>]+>", "", match.group(1)).replace("`", "")
                slug = re.sub(r"[^\w\- ]", "", label.lower())
                headings.add(re.sub(r"\s", "-", slug).strip("-"))
            anchors = set(re.findall(r"\]\(#([^)]+)\)", text))
            relative = document.relative_to(self.ROOT)
            assert anchors <= headings, f"Broken {relative} anchors: {sorted(anchors - headings)}"

    def test_root_readmes_use_portable_seed_uncertainty_notation(self) -> None:
        for relative in ("README.md", "README.es.md"):
            text = (self.ROOT / relative).read_text(encoding="utf-8")
            assert "tau² / n + (v₁ + ... + vₙ) / n²" in text
            assert "\\operatorname{Var}" not in text

    def test_agent_manual_contains_each_operational_route_and_example(self) -> None:
        manual = (self.ROOT / "AGENTS.md").read_text(encoding="utf-8")
        required = {
            "## Install into a consumer project",
            "## Safe experiment workflow",
            "## Adaptive optimization contract",
            "## Generic tasks and preprocessing",
            "## Workflows, composition and secrets",
            "## Model operations, resources and data movement",
            "## Registry, observability and reproducibility",
            "## Minimal YAML contract",
            "## Extension contracts",
            "## Results and publication discipline",
            "## Repository modification rules",
            "## Targeted documentation routes",
            "examples/experiment.yaml",
            "examples/preprocessing.yaml",
            "examples/workflow.yaml",
            "examples/adaptive-hpo.yaml",
        }
        assert not sorted(value for value in required if value not in manual)

    def test_every_maintained_human_guide_has_a_visible_translation(self) -> None:
        """Keep technical, package, security and agent documentation paired."""
        documents = [
            self.ROOT / "README.md",
            self.ROOT / "AGENTS.md",
            self.ROOT / "CHANGELOG.md",
            self.ROOT / "SECURITY.md",
            *sorted((self.ROOT / "docs").glob("*.md")),
            *sorted((self.ROOT / "src/lambdaforge").glob("*/README*.md")),
            *sorted((self.ROOT / "src/lambdaforge").glob("*/*/README*.md")),
        ]
        for document in documents:
            if document.name.endswith(".es.md"):
                counterpart = document.with_name(document.name.removesuffix(".es.md") + ".md")
                expected = counterpart.name
            else:
                counterpart = document.with_name(document.stem + ".es.md")
                expected = counterpart.name
            assert counterpart.is_file(), f"Missing translation for {document}"
            assert expected in document.read_text(encoding="utf-8"), document

    def test_release_051_commands_and_limits_are_documented_in_both_languages(self) -> None:
        required = {
            "lambdaforge results sync JOB",
            "lambdaforge plot learning",
            "lambdaforge artifact inspect",
            "lambdaforge debug",
            "allow_pickle=False",
            "automatic placement",
        }
        english = (self.ROOT / "README.md").read_text(encoding="utf-8")
        spanish = (self.ROOT / "README.es.md").read_text(encoding="utf-8")
        assert not sorted(value for value in required if value not in english)
        spanish_required = {
            "lambdaforge results sync JOB",
            "lambdaforge plot learning",
            "lambdaforge artifact inspect",
            "lambdaforge debug",
            "allow_pickle=False",
            "placement automático",
        }
        assert not sorted(value for value in spanish_required if value not in spanish)
