"""Tiny CPU science demonstrating native managed evidence, not a model benchmark."""

import json

import lambdaforge as lf
from lambdaforge.products import ProductInput
from lambdaforge.work.ResultInput import ResultInput


class Example(lf.Work):
    """Score one explicit value and persist a checkpoint, report and HTML section."""

    def run(self, value: float = 1.0) -> dict[str, float]:  # type: ignore[override]
        if self.resuming and self.checkpoints.exists("state.json"):
            state = self.checkpoints.load_json("state.json")
            self.log(f"Restored compatible checkpoint at step {state['step']}")
        score = 1.0 / (1.0 + value * value)
        self.metrics.log("score", score, step=1)
        self.checkpoints.save_json("state.json", {"step": 1, "value": value}, step=1)
        report = self.outputs.file("report", filename="report.json", role="report")
        report.write_json({"score": score})
        self.outputs.value("summary", {"score": score, "seed": self.seed})
        viewer = self.outputs.html_section("prediction", title="Prediction", section="example")
        viewer.write_text(f"<!doctype html><html><body><h1>Score</h1><p>{score}</p></body></html>")
        return {"score": score}


class ReadHistorical(lf.Work):
    """Consume exact immutable Execution evidence, not a guessed mutable file path."""

    def run(self, previous: ResultInput) -> dict[str, float]:  # type: ignore[override]
        score = float(previous.result["score"])
        self.metrics.log("previous_score", score)
        return {"previous_score": score}


class ReadProduct(lf.Work):
    """Read an explicitly contracted sealed report through its registered member."""

    def run(self, report: ProductInput) -> dict[str, float]:  # type: ignore[override]
        payload = json.loads(report.artifact("report").read_text(encoding="utf-8"))
        self.metrics.log("published_score", payload["score"])
        return {"published_score": payload["score"]}
