"""Tiny installed callables that keep packaged YAML examples locally inspectable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lambdaforge as lf


def prepare(source: Path) -> dict[str, int]:
    """Copy a tiny JSONL input and report its record count."""
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    output = lf.current().run_dir / "processed.jsonl"
    output.write_text("".join(json.dumps(record) + "\n" for record in records))
    lf.metric("records", len(records))
    lf.artifact("processed", output, role="dataset")
    return {"records": len(records)}


def train(
    dataset: Path,
    hidden_dim: int,
    dropout: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Emit deterministic toy evidence; real projects replace this function."""
    score = float(hidden_dim) / 256.0 - float(dropout) + (seed % 10) / 1000.0
    lf.metric("val_auroc", score, step=0)
    return {"val_auroc": score, "input": str(dataset)}


def train_a() -> dict[str, str]:
    """Represent one tiny parallel example branch."""
    return {"model": "a"}


def train_b() -> dict[str, str]:
    """Represent one tiny parallel example branch."""
    return {"model": "b"}


def compare() -> dict[str, bool]:
    """Represent a final workflow comparison."""
    return {"compared": True}
