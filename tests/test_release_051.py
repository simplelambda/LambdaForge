"""Focused LambdaForge 0.5.1 result, data, artifact and authoring contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from lambdaforge.artifacts import ArtifactService, NumpyArtifactInspector
from lambdaforge.configuration import AuthoringConfig
from lambdaforge.data import DataCatalog, DataService
from lambdaforge.experiments import ExperimentConfig, RunResult, RunStatus
from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.experiments.results import RunFingerprint
from lambdaforge.preprocessing import PreprocessingDebugService
from lambdaforge.results import ResultService
from lambdaforge.visualization import VisualizationService


class TestRelease051:
    """Verify only the new public contracts and their important failure paths."""

    @staticmethod
    def _result_tree(root: Path) -> Path:
        run = root / "baseline" / "base" / "seed-7"
        run.mkdir(parents=True)
        config = {
            "schema_version": "1.1",
            "experiment": {"name": "baseline", "variant": "base", "seed": 7},
            "data": {},
            "model": {"target": "tests.fixtures.UserModel.UserModel"},
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
        }
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        RunResult(
            name="baseline",
            run_dir=run,
            variant="base",
            seed=7,
            status=RunStatus.OK,
            attempt_id="attempt-baseline",
            config_fingerprint=RunFingerprint.digest(config),
            final_metrics={"val_loss": 0.3, "val_accuracy": 0.8},
        ).write_json(run / "result.json")
        with (run / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_loss"))
            writer.writeheader()
            writer.writerows(
                (
                    {"epoch": 0, "train_loss": 1.0, "val_loss": 0.8},
                    {"epoch": 1, "train_loss": 0.5, "val_loss": 0.3},
                )
            )
        return run

    def test_friendly_training_and_portable_dataset_reference(self, tmp_path: Path) -> None:
        catalog = tmp_path / "data.yaml"
        catalog.write_text(
            yaml.safe_dump(
                {
                    "datasets": {
                        "samples": {
                            "identity": {"version": "3"},
                            "loader": {
                                "target": "tests.fixtures.TinyMappingDataset.TinyMappingDataset",
                                "path_parameter": "root",
                                "params": {},
                            },
                            "locations": {
                                "local": str(tmp_path / "local-data"),
                                "cluster": "/cluster/data",
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        concise = {
            "name": "friendly",
            "data_catalog": str(catalog),
            "environment": "local",
            "data": {"train": "dataset:samples/train"},
            "model": "tests.fixtures.UserModel.UserModel",
            "loss": "torch.nn.BCEWithLogitsLoss",
            "trainer": {"epochs": 3},
            "resources": {"cpu": 2, "ram": "1GiB", "gpu": 1},
        }
        strict = AuthoringConfig(concise).materialize().to_dict()
        config = ExperimentConfig(strict)

        assert config["experiment"]["name"] == "friendly"
        assert config["trainer"]["max_epochs"] == 3
        assert config.resources.cpu_cores == 2
        assert config.resources.ram_bytes == 1024**3
        assert config.resources.gpu_count == 1
        assert config["data"]["train"]["params"]["root"].endswith("local-data/train")
        local_digest = RunFingerprint.digest(config.as_dict())
        strict["extensions"]["authoring"]["environment"] = "cluster"
        remote_digest = RunFingerprint.digest(ExperimentConfig(strict).as_dict())
        assert local_digest == remote_digest

    def test_result_series_plot_spec_and_exports(self, tmp_path: Path) -> None:
        run = self._result_tree(tmp_path)
        service = ResultService((tmp_path,))

        shown = service.show("baseline")
        series = service.metric_series("baseline", "val_loss")
        spec = VisualizationService(service).learning(
            "baseline", metrics=("val_loss",), aggregate="mean", uncertainty="std"
        )

        assert not shown["ambiguous"]
        assert [point.value for point in series.points] == [0.8, 0.3]
        assert spec.data[-1]["n"] == 1
        assert spec.data[-1]["lower"] is None
        assert json.loads(json.dumps(spec.to_dict()))["plot_type"] == "learning"
        assert service.export("baseline", tmp_path / "series.csv", metric_series=True).is_file()
        figure = VisualizationService(service).render(spec, run / "plots" / "learning.png")
        assert figure.is_file()
        assert figure.with_suffix(".png.plot.json").is_file()
        comparison = service.compare(("baseline",), metrics=("val_loss",), direction="minimize")
        assert comparison["comparisons"][0]["best_group"] == "baseline"
        artifacts = ArtifactService(results=service).list("baseline")
        assert {item["logical_name"] for item in artifacts} >= {
            "learning.png",
            "learning.png-spec",
        }

    def test_sweep_specs_aggregate_seeds_in_one_and_two_dimensions(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sweep.yaml"
        config = {
            "schema_version": "1.1",
            "experiment": {
                "name": "sweep",
                "output_root": str(tmp_path / "runs"),
                "seeds": [7, 17],
            },
            "data": {},
            "model": {
                "target": "tests.fixtures.UserModel.UserModel",
                "params": {"width": 8, "dropout": 0.1},
            },
            "losses": [{"target": "tests.fixtures.UserLoss.UserLoss"}],
            "sweep": {"grid": {"model.params.width": [8, 16], "model.params.dropout": [0.1, 0.2]}},
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        runner = ExperimentRunner()
        for run in ExperimentConfig(config).expand():
            run_dir = runner.experiment_run_dir(run)
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            width = int(run["model"]["params"]["width"])
            dropout = float(run["model"]["params"]["dropout"])
            seed = int(run["experiment"]["seed"])
            RunResult(
                name=str(run["experiment"]["name"]),
                run_dir=run_dir,
                variant=str(run["experiment"]["variant"]),
                seed=seed,
                status=RunStatus.OK,
                attempt_id=f"attempt-{width}-{dropout}-{seed}",
                config_fingerprint=RunFingerprint.digest(run),
                final_metrics={
                    "val_loss": width / 100 + dropout + seed / 10_000,
                    "val_accuracy": width / 20 - dropout + seed / 10_000,
                },
            ).write_json(run_dir / "result.json")

        service = VisualizationService(ResultService((tmp_path / "runs",)))
        one_dimensional = service.sweep(config_path, x="model.params.width", metrics=("val_loss",))
        two_dimensional = service.sweep(
            config_path,
            x="model.params.width",
            y="model.params.dropout",
            metrics=("val_loss",),
            view="heatmap",
        )

        assert len(one_dimensional.data) == 2
        assert {row["n"] for row in one_dimensional.data} == {4}
        assert len(two_dimensional.data) == 4
        assert {row["n"] for row in two_dimensional.data} == {2}
        assert service.render(two_dimensional, tmp_path / "sweep.png").is_file()
        normalized = service.sweep(
            config_path,
            x="model.params.width",
            metrics=("val_loss", "val_accuracy"),
            normalize=True,
        )
        assert normalized.metadata["normalization_method"] == (
            "per-metric min-max over observed cells"
        )
        assert {row["metric"] for row in normalized.data} == {"val_loss", "val_accuracy"}
        assert all(0.0 <= float(row["value"]) <= 1.0 for row in normalized.data)
        assert all("raw_value" in row for row in normalized.data)

    def test_npz_is_safe_bounded_and_explicitly_visualized(self, tmp_path: Path) -> None:
        artifact = tmp_path / "graph.npz"
        positions = np.arange(60_000, dtype=np.float32).reshape(20_000, 3)
        edges = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        np.savez(artifact, points=positions, links=edges)

        inspection = NumpyArtifactInspector(max_statistics_elements=100).inspect(
            artifact, item="points", rows=2
        )
        spec = ArtifactService().visualization_spec(
            artifact,
            visualization_type="graph",
            roles={"nodes": "points", "edges": "links"},
        )

        assert inspection.items[0]["statistics"] == "sampled:100"
        assert len(inspection.items[0]["preview"]) == 2
        assert spec.metadata["node_count"] == 20_000
        assert sum(row.get("kind") == "edge" for row in spec.data) == 3
        assert VisualizationService().render(spec, tmp_path / "graph.png").is_file()

    def test_dataset_and_preprocessing_debug_are_read_only(self, tmp_path: Path) -> None:
        source = tmp_path / "raw.jsonl"
        source.write_text('{"id":"a","text":"hello"}\n', encoding="utf-8")
        task = tmp_path / "preprocessing.yaml"
        task.write_text(
            yaml.safe_dump(
                {
                    "name": "debug-data",
                    "output_root": str(tmp_path / "runs"),
                    "inputs": {"raw": "raw.jsonl"},
                    "outputs": {"processed": "processed"},
                    "preprocess": {
                        "function": "tests.fixtures.normalize_record.normalize_record",
                        "key_field": "id",
                    },
                }
            ),
            encoding="utf-8",
        )
        debug = PreprocessingDebugService().debug(task, records=1)
        assert debug.ok
        assert not (tmp_path / "runs").exists()

        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        (dataset_root / "dataset-artifact.json").write_text(
            json.dumps(
                {
                    "dataset_artifact_version": 1,
                    "dataset_id": f"sha256:{'a' * 64}",
                    "name": "demo",
                    "version": "1",
                    "sample_count": 1,
                    "splits": {"all": 1},
                    "preprocessing_fingerprint": "sha256:config",
                    "source": {},
                    "artifacts": [],
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        catalog = DataCatalog(
            {"demo": {"identity": {"version": "1"}, "locations": {"local": str(dataset_root)}}}
        )
        inspected = DataService(catalog).inspect("dataset:demo")
        assert inspected["dataset_id"] == f"sha256:{'a' * 64}"
        assert inspected["sample_count"] == 1
