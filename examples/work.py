"""Small importable Work examples; copy the pattern into a consumer package."""

from pathlib import Path

import lambdaforge as lf


class Summarize(lf.Work):
    """Count lines in an explicitly managed file."""

    def run(self, source: Path, label: str = "sample") -> dict[str, int | str]:
        count = len(source.read_text(encoding="utf-8").splitlines())
        report = self.outputs.file("report", filename="report.json", role="report")
        report.write_json({"label": label, "records": count})
        self.metrics.log("records", count)
        self.outputs.value("summary", {"label": label, "records": count})
        return {"label": label, "records": count}


class Consume(lf.Work):
    """Consume a named output from a prior Work."""

    def run(self, summary: dict[str, object]) -> dict[str, object]:
        return summary


class Train(lf.Work):
    """Demonstrate seed/search metadata without imposing a model abstraction."""

    def run(self, learning_rate: float = 0.001) -> dict[str, float]:
        score = float((self.seed or 1) * learning_rate)
        self.metrics.log("score", score)
        return {"score": score}
