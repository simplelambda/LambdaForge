"""Reference consumer Python for GUI-generated operations; no GUI or node runtime required.

Run prepare.yaml then train.yaml from the synthetic project described in the manual.
Every executable class uses the sole native Work.run lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lambdaforge import Work
from lambdaforge.data.samples import FieldSpec, SampleDataset, SampleSchema, collate_samples
from lambdaforge.etl import (
    JoinRecords,
    PartitionRecords,
    ReadRecords,
    TrainStandardizer,
    WriteRecords,
)


class PrepareSamples(Work):
    """Join explicitly identified CSV and JSON rows, split groups, publish native members."""

    def run(self, raw: Path, version: str = "1") -> dict[str, Any]:  # type: ignore[override]
        self.outputs.dataset_preflight(name="visual-demo", version=version)
        csv = ReadRecords("csv")(raw / "samples.csv")
        json = ReadRecords("json")(raw / "context.json")
        joined = JoinRecords()(csv, json)
        partitioned = PartitionRecords({"train": 0.7, "validation": 0.3}, seed=71)(joined)

        def process(row: dict[str, Any]) -> dict[str, Any]:
            import hashlib

            if self.stop_requested:
                raise RuntimeError("Preparation cancelled; no Dataset published.")
            stem = hashlib.sha256(row["id"].encode()).hexdigest()
            sample = self.checkpoints.file(
                f"samples/{stem}.npz",
                build=lambda path: WriteRecords("npz")(
                    path, {"x": np.asarray([float(row["x0"]), float(row["x1"])], np.float32)}
                ),
            )
            sidecar = self.checkpoints.file(
                f"samples/{stem}.json",
                build=lambda path: WriteRecords("json")(path, {"context": row["context"]}),
            )
            return {
                "id": row["id"],
                "sample": sample,
                "sidecar": sidecar,
                "targets": {"y": float(row["y"])},
                "partitions": row["partitions"],
            }

        # The native map persists logical ManagedFile dependencies, never machine paths.
        rows = self.resume_map(partitioned, process, key="id", name="prepare-samples", workers=1)
        members = [
            {
                "id": row["id"],
                "assets": {"sample": row["sample"].path, "sidecar": row["sidecar"].path},
                "targets": row["targets"],
                "partitions": row["partitions"],
            }
            for row in rows
        ]
        return dict(
            self.outputs.dataset(
                name="visual-demo",
                version=version,
                members=members,
                source_checkpoint="samples",
                metadata={
                    "recipe": "visual-research-v1",
                    "lineage": "CSV+JSON id join; group split",
                },
            )
        )


def schema(use_context: bool = True) -> SampleSchema:
    """Select fields in ordinary consumer Python; target is never bound as an input."""
    return SampleSchema(
        [
            FieldSpec(
                "features", asset="sample", format="npz", path=("x",), dtype="float32", shape=(2,)
            ),
            FieldSpec("context", asset="sidecar", format="json", path=("context",), shape=(2,)),
            FieldSpec(
                "optional",
                asset="extra",
                format="npy",
                shape=(1,),
                required=False,
                missing="default",
                default=[0.0],
                role="metadata",
            ),
            FieldSpec("target", source="targets", path=("y",), role="target", shape=()),
        ],
        {"x": "features", **({"context": "context"} if use_context else {})},
    )


class TrainSamples(Work):
    """Two named model inputs; selectable field/activation variants through native sweeps."""

    def run(  # type: ignore[override]
        self,
        dataset: Path,
        width: int = 4,
        activation: str = "relu",
        use_context: bool = True,
        epochs: int = 2,
    ) -> dict[str, float]:
        import torch
        from torch import nn

        from lambdaforge.catalog import ComponentFactory
        from lambdaforge.nn import MLP

        contract = schema(use_context)
        train = SampleDataset(dataset, contract, partitions={"split": "train"})
        validation = SampleDataset(dataset, contract, partitions={"split": "validation"})
        if not len(train) or not len(validation):
            raise ValueError("Example requires nonempty train and validation splits.")
        tx = collate_samples([train[i] for i in range(len(train))], contract)
        vx = collate_samples([validation[i] for i in range(len(validation))], contract)
        scaler = TrainStandardizer().fit(
            tx["fields"]["features"], ids=tx["ids"], splits=["train"] * len(train)
        )
        tx["fields"]["features"] = scaler.transform(tx["fields"]["features"])
        vx["fields"]["features"] = scaler.transform(vx["fields"]["features"])

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                factory = ComponentFactory(activation, "activation")
                self.component_provenance = factory.provenance()
                self.encoder = MLP(
                    4 if use_context else 2,
                    1,
                    hidden=[width],
                    activation=factory.component_type(),
                    activation_kwargs=factory.kwargs,
                )

            def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
                return self.encoder(torch.cat([x, context], dim=-1) if context is not None else x)

        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in contract.model_inputs(tx["fields"]).items()
        }
        v = {
            k: torch.as_tensor(value, dtype=torch.float32)
            for k, value in contract.model_inputs(vx["fields"]).items()
        }
        target = torch.as_tensor(tx["fields"]["target"], dtype=torch.float32).reshape(-1, 1)
        expected = torch.as_tensor(vx["fields"]["target"], dtype=torch.float32).reshape(-1, 1)
        loss = torch.tensor(float("nan"))
        for step in range(epochs):
            if self.stop_requested:
                break
            model.train()
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(**x), target).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                loss = torch.nn.functional.mse_loss(model(**v), expected)
            self.metrics.log("val_loss", float(loss), step=step + 1)
        self.checkpoints.save_json("scaler.json", scaler.state_dict())
        self.outputs.file("scaler", filename="scaler.json", role="transform-state").write_json(
            scaler.state_dict()
        )
        self.outputs.file("preview", filename="preview.json", role="report").write_json(
            train.preview(samples=1)
        )
        self.outputs.file("components", filename="components.json", role="provenance").write_json(
            model.component_provenance
        )
        return {"val_loss": float(loss)}


def create_raw(directory: Path) -> None:
    """Write only a new disposable synthetic input directory for this example."""
    import csv
    import json

    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "samples.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "x0", "x1", "y"])
        writer.writeheader()
        for i in range(30):
            writer.writerow({"id": f"sample-{i}", "x0": i / 30, "x1": i % 2, "y": i / 20})
    (directory / "context.json").write_text(
        json.dumps([{"id": f"sample-{i}", "context": [i % 3, 1]} for i in range(30)]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create synthetic raw inputs; never launch science."
    )
    parser.add_argument("raw", type=Path)
    create_raw(parser.parse_args().raw)
