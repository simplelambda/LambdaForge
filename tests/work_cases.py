"""Importable Work fixtures used by process-isolated runtime tests."""

from __future__ import annotations

import json
import sys
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
        mapped = self.map(range(count), lambda value: value * 2, workers=2)
        report = self.run_dir / "report.txt"
        report.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        self.outputs.value("numbers", mapped)
        self.outputs.artifact("report", report, role="report")
        self.metrics.log("score", float(sum(mapped)), step=count)
        self.progress.update(completed=count, total=count, message="done")
        self.checkpoints.save_json("state.json", {"done": True})
        return {"count": count}


class LoggingWork(lf.Work):
    def run(self) -> dict[str, bool]:
        print("ordinary print is captured")
        self.log("managed message")
        self.log("a warning", level="warning")
        return {"logged": True}


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


class AdaptiveScoreWork(lf.Work):
    """Deterministic objective fixture for adaptive seed allocation tests."""

    def run(self, quality: float = 0.0) -> dict[str, float]:
        self.metrics.log("score", quality, step=1)
        return {"score": quality}


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


class ManagedInfrastructureWork(lf.Work):
    def run(self) -> dict[str, Any]:
        cached = self.cache.file(
            "fixtures/source.txt",
            build=lambda target: target.write_text("cache", encoding="utf-8"),
            validate=lambda value: value.read_text() == "cache",
        )
        checkpoint = self.checkpoints.file(
            "prepared/state.txt",
            build=lambda target: target.write_text("ready", encoding="utf-8"),
        )
        report = self.outputs.file(
            "report", filename="report.json", role="report", media_type="application/json"
        )
        report.write_json({"cache": cached.sha256, "checkpoint": checkpoint.read_text()})
        directory = self.outputs.directory("evidence", role="evidence")
        (directory / "value.txt").write_text("scientific evidence", encoding="utf-8")
        python = self.tools.require(sys.executable, version_args=["--version"])
        completed = self.tools.run(
            [python, "-c", "print('external success')"],
            name="Python fixture",
            threads=1,
        )
        return {"returncode": completed.returncode, "cached": cached.read_text()}


class PublishedOutputWork(lf.Work):
    def run(
        self,
        destination: str,
        value: str = "published",
        overwrite: bool = False,
        retain_internal: bool = False,
    ) -> None:
        report = self.outputs.file(
            "report",
            filename="report.txt",
            publish_to=destination,
            overwrite=overwrite,
            retain_internal=retain_internal,
        )
        report.write_text(value)


class PublishedDirectoryWork(lf.Work):
    def run(self, destination: str, overwrite: bool = False) -> None:
        directory = self.outputs.directory(
            "evidence",
            publish_to=destination,
            overwrite=overwrite,
        )
        (directory / "summary.txt").write_text("complete", encoding="utf-8")


class SelectiveMapWork(lf.Work):
    def run(self) -> dict[str, Any]:
        calls_path = self.checkpoints.path("map-calls.json")

        def process(item: dict[str, str]) -> dict[str, Any]:
            calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
            calls.append(item["id"])
            self.checkpoints.save_json("map-calls.json", calls)
            managed = self.cache.file(
                f"items/{item['id']}.txt",
                build=lambda target: target.write_text(item["id"], encoding="utf-8"),
            )
            return {
                "id": item["id"],
                "value": managed.read_text(),
                "file": managed if item["id"] == "a" else None,
            }

        results = self.resume_map(
            [{"id": "a"}, {"id": "b"}],
            process,
            key="id",
            workers=1,
            name="managed-items",
        )
        if not self.checkpoints.exists("failed-once.json"):
            self.checkpoints.save_json("failed-once.json", True)
            raise RuntimeError("retry after cache cleanup")
        return {
            "calls": self.checkpoints.load_json("map-calls.json"),
            "values": [result["value"] for result in results],
        }


class FetchMapWork(lf.Work):
    """Exercise retrying HTTP cache downloads from concurrent resumable items."""

    def run(self, base_url: str, count: int = 4) -> dict[str, Any]:
        def fetch(item: dict[str, str]) -> dict[str, str]:
            selected = self.cache.fetch(
                f"{base_url}/{item['id']}.gz",
                key=f"downloads/{item['id']}.txt",
                retries=2,
                retry_backoff=0,
                decompress="gzip",
            )
            return {"id": item["id"], "value": selected.read_text()}

        return {
            "items": self.resume_map(
                [{"id": str(index)} for index in range(count)],
                fetch,
                key="id",
                workers=min(4, count),
                name="downloads",
            )
        }


class MissingManagedOutputWork(lf.Work):
    def run(self) -> dict[str, bool]:
        self.outputs.file("missing", filename="missing.txt")
        return {"declared": True}
