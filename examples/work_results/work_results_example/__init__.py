"""Small executable scientific-contract demo, not an ML benchmark or application integration."""

import html
import json
from typing import Any

from lambdaforge import Work
from lambdaforge.products import ProductInput


class TrainModel(Work):
    """Produce an explicitly scored toy model snapshot for the portable evidence example."""

    def run(self, width: int = 2) -> dict[str, Any]:
        score = width / (width + 1) + (self.seed or 0) / 1000
        for step in (1, 2):
            self.metrics.log("score", score * step / 2, step=step)
        self.checkpoints.save_json("state.json", {"step": 2, "width": width})
        snapshot = self.outputs.file(
            "model",
            filename="weights.json",
            role="model",
            metadata={"metrics": {"score": score}, "step": 2},
        )
        snapshot.write_json({"width": width, "seed": self.seed, "toy_weight": width / 10})
        return {"score": score}


class VisualizeModel(Work):
    """Read independent exact model bytes and persist an offline interactive visualization."""

    def run(self, model: ProductInput) -> dict[str, Any]:
        if model.selected_artifact is None:
            selected = model.payload["models"][0]
            path = model.artifact(selected["artifact"])
        else:
            path = model.artifact()
        weights = json.loads(path.read_text())
        viewer = self.outputs.html_section("model-view", title="Model viewer")
        viewer.write_text(
            "<!doctype html><html><meta charset='utf-8'><body><h1>Persisted toy model</h1><pre>"
            + html.escape(json.dumps(weights, indent=2))
            + "</pre><button onclick=\"document.body.style.background='#d7edf7'\">"
            "Change theme</button>" + "</body></html>"
        )
        self.outputs.value(
            "summary", {"source_content_id": model.content_id, "width": weights["width"]}
        )
        self.metrics.log("visualized_models", 1)
        return {"model_content_id": model.content_id}
