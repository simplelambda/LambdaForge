"""Smoke an installed LambdaForge wheel from outside its source checkout."""

from __future__ import annotations

from importlib.metadata import distribution
from importlib.resources import files

import lambdaforge
from lambdaforge import Work


def main() -> None:
    """Verify imports, packaged resources and one minimal authoring operation."""
    installed = distribution("lambdaforge")
    assert lambdaforge.__version__ == installed.version
    assert files("lambdaforge").joinpath("schemas/work.schema.json").is_file()
    installed_files = tuple(installed.files or ())
    for relative in (
        "share/lambdaforge/AGENTS.md",
        "share/lambdaforge/AGENTS.es.md",
        "share/lambdaforge/CHANGELOG.md",
        "share/lambdaforge/CHANGELOG.es.md",
        "share/lambdaforge/README.md",
        "share/lambdaforge/README.es.md",
        "share/lambdaforge/SECURITY.md",
        "share/lambdaforge/SECURITY.es.md",
        "share/lambdaforge/examples/work.yaml",
        "share/lambdaforge/examples/sequence.yaml",
        "share/lambdaforge/docs/MANUAL.md",
        "share/lambdaforge/docs/MANUAL.es.md",
    ):
        matches = tuple(
            item
            for item in installed_files
            if item.as_posix().endswith(relative) and installed.locate_file(item).is_file()
        )
        assert len(matches) == 1, relative
    scripts = {
        entry.name: entry.value
        for entry in installed.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["lf"] == scripts["lambdaforge"]
    assert lambdaforge.__all__ == ["Work", "__version__", "clustering"]
    assert Work.__module__ == "lambdaforge.work.Work"
    print(lambdaforge.__version__)


if __name__ == "__main__":
    main()
