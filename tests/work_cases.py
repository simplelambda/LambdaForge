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


class SequentialRecoveryWork(lf.Work):
    """Small acquisition session with resumable items and optional repeated interruptions."""

    def run(self, label: str, source: Path | None = None, failures: int = 1) -> None:
        if not self.checkpoints.exists("session.json"):
            self.checkpoints.save_json("session.json", {"created": 1})
        calls = (
            self.checkpoints.load_json("calls.json")
            if self.checkpoints.exists("calls.json")
            else []
        )
        invocation = (
            self.checkpoints.load_json("invocation.json")
            if self.checkpoints.exists("invocation.json")
            else 0
        )
        self.checkpoints.save_json("invocation.json", invocation + 1)

        def item(number: int) -> int:
            calls.append(number)
            self.checkpoints.save_json("calls.json", calls)
            if number == 2 and invocation < failures:
                raise RuntimeError("synthetic interruption")
            return number

        self.resume_map(range(4), item, key=lambda number: str(number), name="discovery", workers=1)
        self.outputs.value("label", label)
        path = self.run_dir / "answer.txt"
        path.write_text(source.read_text() if source else label)
        self.outputs.artifact("answer", path)


class FrozenAcquisitionRecoveryWork(lf.Work):
    """A tiny external observation retained through failed map ownership and code revisions."""

    def run(self) -> None:
        def acquire(number: int) -> object:
            managed = self.cache.fetch("https://fixture.invalid/frozen", key=f"source/{number}")
            return {"source": managed}

        self.resume_map([0, 1], acquire, key=lambda n: str(n), name="frozen", workers=1)
        if not self.checkpoints.exists("failed.json"):
            self.checkpoints.save_json("failed.json", True)
            raise RuntimeError("synthetic stop")


def read_frozen_item(item: object) -> str:
    """Top-level spawn target with an explicit native ManagedFile input."""
    return item["source"].read_text()


class BoundMapRecoveryWork(lf.Work):
    """Policy and dependency contracts are independent of executor/worker count."""

    def run(self, executor: str = "thread") -> None:
        source = self.checkpoints.file("source.txt", build=lambda p: p.write_text("frozen"))
        self.resume_map(
            [{"id": "one", "source": source}],
            read_frozen_item,
            key="id",
            workers=2,
            executor=executor,
            name="bound",
            scientific_policy={"equation": "v1", "cutoff": 3.0},
        )
        if not self.checkpoints.exists("failed.json"):
            self.checkpoints.save_json("failed.json", True)
            raise RuntimeError("synthetic stop")


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


class QueuedParallelWork(lf.Work):
    """A disposable two-seed barrier proves queue admission preserves native Run overlap."""

    def run(self, markers: str) -> dict[str, int]:
        import time

        root = Path(markers)
        assert self.seed in {1, 2}
        (root / f"seed-{self.seed}.started").write_text(str(time.monotonic()))
        deadline = time.monotonic() + 30
        while not all((root / f"seed-{seed}.started").is_file() for seed in (1, 2)):
            if time.monotonic() >= deadline:
                raise RuntimeError("The other required seed did not start concurrently")
            time.sleep(0.05)
        (root / f"seed-{self.seed}.finished").write_text(str(time.monotonic()))
        return {"seed": self.seed}


class StructuredSweepWork(lf.Work):
    """Finite JSON categories remain one parameter, including conditional bounds."""

    def run(
        self, family: str, bounds: list[float] | dict[str, float] | None = None
    ) -> dict[str, Any]:
        score = 0.5 + float(self.seed or 0) / 100
        self.metrics.log("score", score, step=1)
        return {"family": family, "bounds": bounds, "score": score}


class ScoredModelSnapshotWork(lf.Work):
    """Tiny CPU-only fixture binds a model snapshot to its own evaluated metrics."""

    def run(self, width: int = 64) -> dict[str, float]:
        score = float(self.seed or 0) / 10
        self.checkpoints.save_json("weights.json", {"width": width, "seed": self.seed})
        self.outputs.from_checkpoint(
            "model", "weights.json", metadata={"metrics": {"score": score}, "step": 2}
        )
        # Deliberately different current metric: selection must use the scored snapshot.
        self.metrics.log("score", -score, step=3)
        return {"score": -score}


class AdaptiveScoreWork(lf.Work):
    """Deterministic objective fixture for adaptive seed allocation tests."""

    def run(self, quality: float = 0.0) -> dict[str, float]:
        self.metrics.log("score", quality, step=1)
        return {"score": quality}


class FleetObservableWork(lf.Work):
    """Synthetic live read-model fixture; a test observer releases the native Run."""

    def run(self, quality: float, release: str) -> dict[str, float]:
        import time

        self.log("live Fleet scientific output")
        self.metrics.log("score", quality, step=7)
        deadline = time.monotonic() + 45
        while not Path(release).is_file():
            if time.monotonic() >= deadline:
                raise RuntimeError("Test observer did not read the active Run")
            time.sleep(0.05)
        return {"score": quality}


class FlatAdaptiveScoreWork(lf.Work):
    """Constant-objective fixture for candidate-budget termination tests."""

    def run(self, choice: int = 0) -> dict[str, int | float]:
        self.metrics.log("score", 0.5, step=1)
        return {"choice": choice, "score": 0.5}


class ConfirmationFailureWork(lf.Work):
    """Fail one dedicated confirmation seed to exercise unbiased incomplete selection."""

    def run(self, quality: float = 0.0) -> dict[str, float]:
        if self.seed == 102:
            raise RuntimeError("synthetic confirmation failure")
        self.metrics.log("score", quality, step=1)
        return {"score": quality}


class ConditionalSweepWork(lf.Work):
    """Signature fixture for exact finite conditional sweep expansion."""

    def run(self, model: str, depth: int | None = None) -> dict[str, str | int | None]:
        return {"model": model, "depth": depth}


class MembershipSweepWork(lf.Work):
    """Generic conditional scientific-design fixture, not consumer training code."""

    def run(
        self,
        pooling_type: str,
        area_mode: str = "point",
        attention_hidden: int = 16,
        topk_fraction: float = 0.1,
        local_scale: float = 1,
        beta: float = 1,
    ) -> dict[str, str]:
        return {"branch": pooling_type}


class PolicyNamedParameterWork(lf.Work):
    """Ensure structured search space names never collide with controller policy fields."""

    def run(self, trials: int, max_parallel: int) -> dict[str, int]:
        self.metrics.log("score", float(trials + max_parallel), step=1)
        return {"trials": trials, "max_parallel": max_parallel}


class FidelityScoreWork(lf.Work):
    """Cooperative cumulative-budget fixture for adaptive continuation tests."""

    def run(self, quality: float = 0.0) -> dict[str, float | int]:
        assert self.fidelity is not None
        if self.resuming:
            previous = self.checkpoints.load_json("fidelity.json")
            assert previous["completed"] == self.fidelity.current
        self.checkpoints.save_json("fidelity.json", {"completed": self.fidelity.target})
        score = quality + self.fidelity.target / self.fidelity.maximum
        self.metrics.log("score", score, step=self.fidelity.target)
        return {"score": score, "budget": self.fidelity.target}


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


class RecoverableStudyWork(lf.Work):
    """A consumer failure after a durable checkpoint, isolated to one candidate."""

    def run(self, choice: int) -> dict[str, float]:
        if choice == 0 and not self.resuming:
            self.checkpoints.save_json("progress.json", {"epoch": 3})
            raise RuntimeError("consumer bug after checkpoint")
        self.metrics.log("score", 0.5 + choice / 10, step=4)
        return {"score": 0.5 + choice / 10}


class FixedRecoveryWork(lf.Work):
    """Small owned evidence with a deliberately recoverable per-seed consumer failure."""

    def run(
        self, fail_seed: int = 54, failures: int = 1, choice: int = 0, source: Path | None = None
    ) -> dict[str, float]:
        count = (
            self.checkpoints.load_json("attempts.json")["count"]
            if self.checkpoints.exists("attempts.json")
            else 0
        )
        self.checkpoints.save_json("attempts.json", {"count": count + 1})
        if self.seed == fail_seed and choice == 0 and count < failures:
            raise RuntimeError("deliberate seed failure after application checkpoint")
        score = float(choice)
        self.metrics.log("score", score, step=3)
        self.outputs.file("report", filename="report.json").write_json({"seed": self.seed})
        return {"score": score}


class SingleFileRecoveryWork(lf.Work):
    """Durable producer fixture: failed/uncommitted files must not invalidate good siblings."""

    def run(self, source: Path, count: int = 3) -> dict[str, Any]:
        def produce(index: int, target: Path) -> None:
            if index == count - 1 and not self.checkpoints.exists("failed-once.json"):
                target.write_bytes(b"unfinished")
                self.checkpoints.save_json("failed-once.json", True)
                raise MemoryError("preventive application guard, not OS OOM")
            if index < count - 1 and self.checkpoints.exists(f"producers/{index}.bin"):
                raise AssertionError("an intact expensive producer was rebuilt")
            target.write_bytes(source.read_bytes() + str(index).encode())

        files = [
            self.checkpoints.file(
                f"producers/{index}.bin",
                build=lambda path, i=index: produce(i, path),
                validate=lambda managed: managed.read_bytes().startswith(source.read_bytes()),
            )
            for index in range(count)
        ]
        self.outputs.file("complete", filename="complete.json").write_json(
            {"files": [managed.sha256 for managed in files]}
        )
        return {"files": len(files), "resuming": self.resuming}


class BoundedInputRecoveryWork(SingleFileRecoveryWork):
    """Distinct roles can share authored names but have different bounded staging names."""

    def run(self, source: Path, reference: Path) -> dict[str, Any]:
        assert reference.read_text() == "reference"
        return super().run(source, count=3)


class ChainedBoundedRecoveryWork(SingleFileRecoveryWork):
    """Six-role legacy staging with three scientific failures and expensive retained files."""

    def run(
        self,
        legacy_train: Path,
        legacy_validation: Path,
        legacy_test: Path,
        legacy_positives: Path,
        legacy_negatives: Path,
        legacy_all: Path,
    ) -> dict[str, Any]:
        for path in (
            legacy_train,
            legacy_validation,
            legacy_test,
            legacy_positives,
            legacy_negatives,
            legacy_all,
        ):
            assert path.is_file()
        count = (
            self.checkpoints.load_json("invocations.json")
            if self.checkpoints.exists("invocations.json")
            else 0
        )
        self.checkpoints.save_json("invocations.json", count + 1)
        result = super().run(legacy_train)
        if count < 3:
            raise ValueError("synthetic independent scientific failure after restored files")
        return result


class SingleFileRecoveryDatasetWork(SingleFileRecoveryWork):
    """Publish only after every producer returned a validated managed file."""

    def run(self, source: Path, count: int = 3) -> dict[str, Any]:
        result = super().run(source, count)
        self.outputs.dataset(
            name="recovered-producers",
            version="1",
            source_checkpoint="producers",
            members=[
                {"id": str(index), "path": self.checkpoints.file(f"producers/{index}.bin").path}
                for index in range(count)
            ],
        )
        return result


class ProductConsumerWork(lf.Work):
    """Tiny product consumer uses public metadata/verified promoted artifacts only."""

    def run(self, selection: object, artifact: str | None = None) -> dict[str, object]:
        from lambdaforge.products import ProductInput

        assert isinstance(selection, ProductInput)
        summary = {
            "content_id": selection.content_id,
            "contract": selection.product.contract.identifier,
            "payload": dict(selection.payload),
        }
        if artifact is not None:
            self.outputs.file("copy", filename="copy.bin").write_bytes(
                selection.artifact(artifact).read_bytes()
            )
        self.outputs.value("source_product", summary)
        return summary
