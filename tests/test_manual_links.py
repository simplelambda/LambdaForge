"""The consolidated documentation ships as one indexed public manual per language."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def anchors(text: str) -> set[str]:
    result = set(re.findall(r'<a id="([^"]+)"', text))
    counts: dict[str, int] = {}
    fenced = False
    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        match = re.match(r"^#{1,6} (.+?)\s*#*$", line)
        if fenced or not match:
            continue
        slug = re.sub(r"[^\w\- ]", "", match[1].lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def test_only_manuals_and_essential_internal_references_remain():
    assert {p.relative_to(ROOT / "docs").as_posix() for p in (ROOT / "docs").rglob("*.md")} == {
        "MANUAL.md",
        "MANUAL.es.md",
        "ai/FRAMEWORK_CONTRACTS.md",
        "ai/IMPLEMENTATION.md",
        "ai/IMPLEMENTATION.es.md",
        "ai/AGENT_GUIDE.es.md",
    }
    for language in ("", ".es"):
        manual = (ROOT / f"docs/MANUAL{language}.md").read_text()
        assert "FusedGromovWassersteinDistance" in manual
        assert "#guide-study-extensions" in manual
        assert "#guide-dataset-reconstruction" in manual
        assert len(re.findall(r"^- \[", manual, re.MULTILINE)) >= 30


def test_local_documentation_links_and_chapter_anchors_resolve():
    documents = list((ROOT / "docs").rglob("*.md")) + list(ROOT.glob("*.md"))
    documents += list((ROOT / "src/lambdaforge/agents/assets").rglob("*.md"))
    errors = []
    cached_anchors: dict[Path, set[str]] = {}
    for document in documents:
        text = document.read_text()
        for raw in re.findall(r"\]\(([^\s)]+)\)", text):
            link = urlsplit(raw)
            if link.scheme or link.netloc:
                continue
            target = (document.parent / unquote(link.path)).resolve() if link.path else document
            if not target.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing {raw}")
            elif link.fragment and target.suffix == ".md":
                if target not in cached_anchors:
                    cached_anchors[target] = anchors(target.read_text())
                if unquote(link.fragment) not in cached_anchors[target]:
                    errors.append(f"{document.relative_to(ROOT)}: missing anchor {raw}")
    assert not errors, "\n".join(errors)
