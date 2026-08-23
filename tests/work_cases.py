"""Importable Work fixtures used by process-isolated runtime tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import lambdaforge as lf


def function_target() -> None:
    pass


class ForeignClass:
    def run(self) -> None:
        pass


class CompleteWork(lf.Work):
    """Exercise the user-owned Work surface."""

    def run(self, source: Path, count: int = 2) -> dict[str, Any]:
        assert source == self.inputs["source"].path
        assert self.config.parameter("count") == count
        mapped = self.map(range(count), lambda value: value * 2, key=str, workers=2)
        report = self.run_dir / "report.txt"
        report.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        self.outputs.value("numbers", mapped)
        self.outputs.artifact("report", report, role="report")
        self.metrics.log("score", float(sum(mapped)), step=count)
        self.progress.update(completed=count, total=count, message="done")
        self.checkpoints.save_json("state.json", {"done": True})
        return {"count": count}


class Producer(lf.Work):
    def run(self, value: int = 4) -> dict[str, int]:
        self.outputs.value("number", value)
        return {"produced": value}


class Consumer(lf.Work):
    def run(self, number: int) -> dict[str, int]:
        return {"consumed": number}


class InputWork(lf.Work):
    def run(self, source: Path) -> str:
        return str(source)


class SeedWork(lf.Work):
    def run(self, scale: float = 1.0) -> dict[str, float | int | None]:
        score = float((self.seed or 0) * scale)
        self.metrics.log("score", score)
        return {"seed": self.seed, "score": score}


class ResumeWork(lf.Work):
    def run(self) -> dict[str, bool]:
        if not self.resuming:
            self.checkpoints.save_json("state.json", {"ready": True})
            raise RuntimeError("retry me")
        return self.checkpoints.load_json("state.json")


class PublishDataset(lf.Work):
    def run(self) -> dict[str, str]:
        asset = self.run_dir / "member.txt"
        asset.write_text("member", encoding="utf-8")
        dataset = self.outputs.dataset(
            name="test-work-dataset",
            version="1",
            members=({"id": "one", "split": "train", "path": asset},),
            metadata={"purpose": "test"},
        )
        return {"content_id": str(dataset["dataset_id"])}


class ImmutableWork(lf.Work):
    def run(self, source: Path, count: int) -> dict[str, bool]:
        rejected: dict[str, bool] = {}
        attempts = {
            "inputs": lambda: self.inputs.__setitem__("other", self.inputs["source"]),
            "parameters": lambda: self.config.parameters.__setitem__("count", 9),
            "resources": lambda: setattr(self.resources, "cpu", 99),
        }
        for name, mutation in attempts.items():
            try:
                mutation()
            except (AttributeError, TypeError):
                rejected[name] = True
        return rejected


class DuplicateMapWork(lf.Work):
    def run(self) -> None:
        self.map([1, 2], str, key=lambda _value: "same")


class UnsafeJsonWork(lf.Work):
    def run(self) -> dict[str, float]:
        return {"invalid": float("nan")}


class CudaWork(lf.Work):
    def run(self) -> dict[str, float]:
        value = torch.tensor([2.0], device="cuda")
        score = float((value * value).cpu().item())
        self.metrics.log("cuda_score", score)
        return {"cuda_score": score}
