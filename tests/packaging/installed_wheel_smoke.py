"""Smoke an installed LambdaForge wheel from outside its source checkout."""

from __future__ import annotations

import json
from importlib.metadata import distribution
from importlib.resources import files

import lambdaforge
from lambdaforge.configuration import AuthoringConfig


def main() -> None:
    """Verify imports, packaged resources and one minimal authoring operation."""
    installed = distribution("lambdaforge")
    assert lambdaforge.__version__ == installed.version
    assert files("lambdaforge").joinpath("schemas/experiment.schema.json").is_file()
    assert files("lambdaforge").joinpath("schemas/task.schema.json").is_file()
    assert files("lambdaforge").joinpath("schemas/authoring.schema.json").is_file()
    assert files("lambdaforge").joinpath("schemas/dataset.schema.json").is_file()
    installed_files = tuple(installed.files or ())
    for relative in (
        "share/lambdaforge/AGENTS.md",
        "share/lambdaforge/CHANGELOG.md",
        "share/lambdaforge/SECURITY.md",
        "share/lambdaforge/examples/preprocessing.yaml",
        "share/lambdaforge/examples/dataset-recipe.yaml",
        "share/lambdaforge/docs/MANUAL.md",
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
    materialized = AuthoringConfig(
        {
            "name": "wheel-only",
            "model": "torch.nn.Identity",
            "loss": "torch.nn.MSELoss",
            "trainer": {"epochs": 1},
        }
    ).materialize()
    print(json.dumps({"version": lambdaforge.__version__, "kind": materialized.kind.value}))


if __name__ == "__main__":
    main()
