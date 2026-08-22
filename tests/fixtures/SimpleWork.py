"""Importable ordinary functions used by the simple-work integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lambdaforge as lf


def process(source: Path, factor: int = 1, seed: int | None = None) -> dict[str, Any]:
    """Read an integer, report it and create one declared output."""
    value = int(source.read_text(encoding="utf-8")) * factor
    output = lf.current().run_dir / "value.txt"
    output.write_text(str(value), encoding="utf-8")
    lf.metric("value", value, step=0)
    lf.artifact("value", output, role="report", media_type="text/plain")
    return {"value": value, "seed": seed}


def publish(source: Path) -> dict[str, Any]:
    """Publish a one-member dataset through the runtime facade."""
    copied = lf.current().run_dir / "member.txt"
    copied.write_bytes(source.read_bytes())
    return lf.publish_dataset(
        "simple-members",
        "1",
        ({"id": "one", "split": "train", "assets": {"text": copied}},),
    )


def identity(value: int = 1) -> int:
    """Return one JSON scalar for workflow tests."""
    return value


def score(value: int, seed: int | None = None) -> dict[str, int | None]:
    """Emit a comparable objective from an ordinary seeded function."""
    lf.metric("score", value)
    return {"value": value, "seed": seed}


def symlink_artifact() -> None:
    """Attempt to register an intentionally unsafe artifact path."""
    run_dir = lf.current().run_dir
    actual = run_dir / "actual"
    actual.mkdir()
    (actual / "value.txt").write_text("unsafe", encoding="utf-8")
    (run_dir / "linked").symlink_to(actual, target_is_directory=True)
    lf.artifact("unsafe", "linked/value.txt")


class Multiplier:
    """Small explicit class escape hatch used by authoring tests."""

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def calculate(self, value: int) -> int:
        """Return the configured product."""
        return value * self.factor


def inspect_dataset(dataset: Path) -> dict[str, bool]:
    """Receive a resolved managed DatasetVersion root as an ordinary path."""
    return {"manifest": (dataset / "dataset-artifact.json").is_file()}
