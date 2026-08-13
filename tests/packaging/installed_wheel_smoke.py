"""Smoke an installed LambdaForge wheel from outside its source checkout."""

from __future__ import annotations

import json
from importlib.metadata import distribution
from importlib.resources import files

import lambdaforge
from lambdaforge.configuration import AuthoringConfig


def main() -> None:
    """Verify imports, packaged resources and one minimal authoring operation."""
    assert lambdaforge.__version__ == "0.6.0"
    assert files("lambdaforge").joinpath("schemas/experiment.schema.json").is_file()
    assert files("lambdaforge").joinpath("schemas/task.schema.json").is_file()
    assert files("lambdaforge").joinpath("schemas/authoring.schema.json").is_file()
    assert files("lambdaforge.tasks").joinpath("README.md").is_file()
    installed = distribution("lambdaforge")
    installed_files = tuple(installed.files or ())
    for relative in (
        "share/lambdaforge/AGENTS.md",
        "share/lambdaforge/AGENTS.es.md",
        "share/lambdaforge/examples/preprocessing-simple.yaml",
        "share/lambdaforge/docs/RESULTS.md",
        "share/lambdaforge/docs/RESULTS.es.md",
    ):
        matches = tuple(
            item
            for item in installed_files
            if item.as_posix().endswith(relative) and installed.locate_file(item).is_file()
        )
        assert len(matches) == 1, relative
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
