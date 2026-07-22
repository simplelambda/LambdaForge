"""Tests for optional Lightning tracking adapters and YAML integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from lambdaforge.experiments.ExperimentRunner import ExperimentRunner
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.integrations.Lightning import Lightning, LoggerType
from lambdaforge.plugins.PluginRegistry import PluginRegistry
from lambdaforge.tracking import (
    MLflowTrackingLogger,
    TensorBoardTrackingLogger,
    TrackingBackend,
    TrackingDependencyError,
    TrackingDependencyGuard,
    WeightsAndBiasesTrackingLogger,
)
from lambdaforge.training.LightningRunner import LightningRunner
from lambdaforge.training.LightningTrainConfig import LightningTrainConfig


class TestTrackingAdapters:
    """Verify optional dependency boundaries and lossless native forwarding."""

    def test_adapters_inherit_selected_modern_or_legacy_native_loggers(self) -> None:
        """Keep all wrappers native to the compatibility backend selected once."""
        backend_root = Lightning.backend_name.split(".")[0]
        adapters = (
            (MLflowTrackingLogger, Lightning.MLFlowLogger),
            (TensorBoardTrackingLogger, Lightning.TensorBoardLogger),
            (WeightsAndBiasesTrackingLogger, Lightning.WandbLogger),
        )

        for adapter, native in adapters:
            assert issubclass(adapter, native)
            assert issubclass(adapter, LoggerType)
            assert native.__module__.startswith(backend_root)

    def test_importing_public_adapters_does_not_import_optional_backends(self) -> None:
        """Delay every service SDK import until a concrete logger is constructed."""
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(repository / "src"), str(repository)])
        script = (
            "import sys; "
            "from lambdaforge.tracking import (MLflowTrackingLogger, "
            "TensorBoardTrackingLogger, WeightsAndBiasesTrackingLogger); "
            "classes=(MLflowTrackingLogger, TensorBoardTrackingLogger, "
            "WeightsAndBiasesTrackingLogger); "
            "blocked=('mlflow','tensorboard','tensorboardX','wandb'); "
            "loaded=[name for name in blocked if name in sys.modules]; "
            "assert len(classes) == 3; assert not loaded, loaded"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr

    def test_mlflow_forwards_every_native_option_exactly(self, monkeypatch) -> None:
        """Preserve MLflow identity, storage, artifact and synchronization options."""
        requirements: list[TrackingBackend] = []
        calls: list[dict[str, Any]] = []
        tags = {"team": "research", "seed": 7}

        monkeypatch.setattr(
            TrackingDependencyGuard,
            "require",
            lambda guard: requirements.append(guard.backend),
        )
        monkeypatch.setattr(MLflowTrackingLogger, "_SUPPORTS_SYNCHRONOUS", True)
        monkeypatch.setattr(
            Lightning.MLFlowLogger,
            "__init__",
            lambda _logger, **kwargs: calls.append(kwargs),
        )

        MLflowTrackingLogger(
            experiment_name="tabular",
            run_name="seed-7",
            tracking_uri="https://mlflow.invalid",
            tags=tags,
            save_dir=None,
            log_model="all",
            prefix="fold/",
            artifact_location="s3://bucket/artifacts",
            run_id="run-7",
            synchronous=True,
        )

        assert requirements == [TrackingBackend.MLFLOW]
        assert calls == [
            {
                "experiment_name": "tabular",
                "run_name": "seed-7",
                "tracking_uri": "https://mlflow.invalid",
                "tags": tags,
                "save_dir": None,
                "log_model": "all",
                "prefix": "fold/",
                "artifact_location": "s3://bucket/artifacts",
                "run_id": "run-7",
                "synchronous": True,
            }
        ]

    def test_mlflow_reads_tracking_uri_when_the_adapter_is_instantiated(
        self,
        monkeypatch,
    ) -> None:
        """Honor environment changes made after the adapter module was imported."""
        calls: list[dict[str, Any]] = []
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:runtime-mlruns")
        monkeypatch.setattr(TrackingDependencyGuard, "require", lambda _guard: None)
        monkeypatch.setattr(
            Lightning.MLFlowLogger,
            "__init__",
            lambda _logger, **kwargs: calls.append(kwargs),
        )

        MLflowTrackingLogger()

        assert calls[0]["tracking_uri"] == "file:runtime-mlruns"

    def test_tensorboard_forwards_writer_options_exactly(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """Preserve the full native logger and SummaryWriter configuration."""
        requirements: list[TrackingBackend] = []
        calls: list[dict[str, Any]] = []
        log_dir = tmp_path / "tensorboard"
        sub_dir = Path("fold-2")

        monkeypatch.setattr(
            TrackingDependencyGuard,
            "require",
            lambda guard: requirements.append(guard.backend),
        )
        monkeypatch.setattr(
            Lightning.TensorBoardLogger,
            "__init__",
            lambda _logger, **kwargs: calls.append(kwargs),
        )

        TensorBoardTrackingLogger(
            save_dir=log_dir,
            name="curves",
            version="seed-11",
            log_graph=True,
            default_hp_metric=False,
            prefix="val/",
            sub_dir=sub_dir,
            max_queue=17,
            flush_secs=3,
            filename_suffix=".events",
        )

        assert requirements == [TrackingBackend.TENSORBOARD]
        assert calls == [
            {
                "save_dir": log_dir,
                "name": "curves",
                "version": "seed-11",
                "log_graph": True,
                "default_hp_metric": False,
                "prefix": "val/",
                "sub_dir": sub_dir,
                "max_queue": 17,
                "flush_secs": 3,
                "filename_suffix": ".events",
            }
        ]

    def test_wandb_forwards_identity_offline_and_init_options_exactly(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """Keep W&B offline policy and arbitrary wandb.init options user-owned."""
        requirements: list[TrackingBackend] = []
        calls: list[dict[str, Any]] = []
        experiment = object()

        monkeypatch.setattr(
            TrackingDependencyGuard,
            "require",
            lambda guard: requirements.append(guard.backend),
        )
        monkeypatch.setattr(
            WeightsAndBiasesTrackingLogger,
            "_SUPPORTS_ADD_FILE_POLICY",
            True,
        )
        monkeypatch.setattr(
            Lightning.WandbLogger,
            "__init__",
            lambda _logger, **kwargs: calls.append(kwargs),
        )

        WeightsAndBiasesTrackingLogger(
            name="attention-ablation",
            save_dir=tmp_path,
            version="v-3",
            offline=True,
            dir=tmp_path / "wandb",
            id="run-3",
            anonymous=False,
            project="lambda-research",
            log_model=False,
            experiment=experiment,
            prefix="seed/",
            checkpoint_name="best",
            add_file_policy="immutable",
            entity="simplelambda",
            group="ablation",
            tags=["gat", "seed-3"],
        )

        assert requirements == [TrackingBackend.WEIGHTS_AND_BIASES]
        assert calls == [
            {
                "name": "attention-ablation",
                "save_dir": tmp_path,
                "version": "v-3",
                "offline": True,
                "dir": tmp_path / "wandb",
                "id": "run-3",
                "anonymous": False,
                "project": "lambda-research",
                "log_model": False,
                "experiment": experiment,
                "prefix": "seed/",
                "checkpoint_name": "best",
                "entity": "simplelambda",
                "group": "ablation",
                "tags": ["gat", "seed-3"],
                "add_file_policy": "immutable",
            }
        ]

    @pytest.mark.parametrize(
        ("adapter", "native", "backend", "kwargs"),
        [
            (
                MLflowTrackingLogger,
                Lightning.MLFlowLogger,
                TrackingBackend.MLFLOW,
                {},
            ),
            (
                TensorBoardTrackingLogger,
                Lightning.TensorBoardLogger,
                TrackingBackend.TENSORBOARD,
                {"save_dir": "logs"},
            ),
            (
                WeightsAndBiasesTrackingLogger,
                Lightning.WandbLogger,
                TrackingBackend.WEIGHTS_AND_BIASES,
                {},
            ),
        ],
    )
    def test_missing_dependency_fails_before_native_initialization(
        self,
        monkeypatch,
        adapter,
        native,
        backend: TrackingBackend,
        kwargs: dict[str, Any],
    ) -> None:
        """Raise the backend-specific extra hint before any native side effect."""
        native_calls: list[dict[str, Any]] = []

        def missing(guard: TrackingDependencyGuard) -> None:
            assert guard.backend is backend
            raise TrackingDependencyError(guard.backend)

        monkeypatch.setattr(TrackingDependencyGuard, "require", missing)
        monkeypatch.setattr(
            native,
            "__init__",
            lambda _logger, **native_kwargs: native_calls.append(native_kwargs),
        )

        with pytest.raises(TrackingDependencyError) as captured:
            adapter(**kwargs)

        assert captured.value.backend is backend
        assert captured.value.install_hint == f"pip install 'lambdaforge[{backend.extra}]'"
        assert native_calls == []

    def test_late_native_options_degrade_explicitly_on_lightning_22(
        self,
        monkeypatch,
    ) -> None:
        """Keep 2.2 defaults working and reject unsupported newer requests."""
        mlflow_calls: list[dict[str, Any]] = []
        wandb_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(TrackingDependencyGuard, "require", lambda _guard: None)
        monkeypatch.setattr(MLflowTrackingLogger, "_SUPPORTS_SYNCHRONOUS", False)
        monkeypatch.setattr(
            WeightsAndBiasesTrackingLogger,
            "_SUPPORTS_ADD_FILE_POLICY",
            False,
        )
        monkeypatch.setattr(
            Lightning.MLFlowLogger,
            "__init__",
            lambda _logger, **kwargs: mlflow_calls.append(kwargs),
        )
        monkeypatch.setattr(
            Lightning.WandbLogger,
            "__init__",
            lambda _logger, **kwargs: wandb_calls.append(kwargs),
        )

        MLflowTrackingLogger()
        WeightsAndBiasesTrackingLogger()

        assert "synchronous" not in mlflow_calls[0]
        assert "add_file_policy" not in wandb_calls[0]
        with pytest.raises(TypeError, match="synchronous requires"):
            MLflowTrackingLogger(synchronous=True)
        with pytest.raises(TypeError, match="add_file_policy='immutable' requires"):
            WeightsAndBiasesTrackingLogger(add_file_policy="immutable")

    def test_yaml_object_factory_and_runner_accept_native_adapter(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """Build a public target from YAML and pass it unchanged to Trainer."""
        native_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(TrackingDependencyGuard, "require", lambda _guard: None)
        monkeypatch.setattr(
            Lightning.TensorBoardLogger,
            "__init__",
            lambda _logger, **kwargs: native_calls.append(kwargs),
        )
        config = yaml.safe_load(
            f"""
experiment:
  name: tracking-yaml
  output_root: {tmp_path.as_posix()}
  variant: base
  seed: 5
trainer:
  max_epochs: 1
  checkpoint_policy: none
  logger:
    target: lambdaforge.tracking.TensorBoardTrackingLogger
    params:
      save_dir: {tmp_path.as_posix()}/events
      name: experiment
      version: seed-5
      max_queue: 9
"""
        )

        runner = ExperimentRunner()._build_runner(
            config,
            metrics=[],
            plugins=PluginRegistry(),
        )

        assert isinstance(runner, LightningRunner)
        assert isinstance(runner.config, LightningTrainConfig)
        assert isinstance(runner.config.logger, TensorBoardTrackingLogger)
        assert runner._build_logger() is runner.config.logger
        assert native_calls == [
            {
                "save_dir": f"{tmp_path.as_posix()}/events",
                "name": "experiment",
                "version": "seed-5",
                "log_graph": False,
                "default_hp_metric": True,
                "prefix": "",
                "sub_dir": None,
                "max_queue": 9,
            }
        ]

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("lambdaforge.tracking.MLflowTrackingLogger", MLflowTrackingLogger),
            (
                "lambdaforge.tracking.TensorBoardTrackingLogger",
                TensorBoardTrackingLogger,
            ),
            (
                "lambdaforge.tracking.WeightsAndBiasesTrackingLogger",
                WeightsAndBiasesTrackingLogger,
            ),
        ],
    )
    def test_public_lazy_exports_are_object_factory_targets(self, target: str, expected) -> None:
        """Expose concise stable target paths for all tracking backends."""
        assert ObjectFactory.import_object(target) is expected
