"""Optional tracking dependency and compatibility-boundary tests."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lambdaforge.integrations.Lightning import Lightning
from lambdaforge.tracking import (
    TrackingBackend,
    TrackingDependencyError,
    TrackingDependencyGuard,
)


class TestTrackingCore:
    """Verify lazy imports, actionable errors and stable public logger types."""

    def test_backend_metadata_is_canonical(self) -> None:
        assert [backend.value for backend in TrackingBackend] == [
            "mlflow",
            "tensorboard",
            "wandb",
        ]
        assert TrackingBackend.from_value("wandb") is TrackingBackend.WEIGHTS_AND_BIASES
        assert TrackingBackend.MLFLOW.dependency == "mlflow"
        assert TrackingBackend.TENSORBOARD.dependencies == ("tensorboard", "tensorboardX")
        assert TrackingBackend.TENSORBOARD.extra == "tensorboard"
        assert TrackingBackend.WEIGHTS_AND_BIASES.install_hint == "pip install 'lambdaforge[wandb]'"
        tensorboard_error = TrackingDependencyError(TrackingBackend.TENSORBOARD)
        assert tensorboard_error.dependencies == ("tensorboard", "tensorboardX")
        assert "'tensorboard' or 'tensorboardX'" in str(tensorboard_error)
        assert "lambdaforge[tensorboard]" in str(tensorboard_error)

    def test_guard_reports_missing_dependency_without_importing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        imported: list[str] = []
        guard_module = importlib.import_module("lambdaforge.tracking.TrackingDependencyGuard")
        monkeypatch.setattr(
            guard_module,
            "importlib",
            SimpleNamespace(
                util=SimpleNamespace(find_spec=lambda name: None),
                import_module=lambda name: imported.append(name),
            ),
        )
        guard = TrackingDependencyGuard(TrackingBackend.MLFLOW)

        assert not guard.is_available
        with pytest.raises(TrackingDependencyError) as captured:
            guard.require()

        assert imported == []
        assert captured.value.backend is TrackingBackend.MLFLOW
        assert captured.value.dependency == "mlflow"
        assert captured.value.dependencies == ("mlflow",)
        assert captured.value.extra == "mlflow"
        assert "lambdaforge[mlflow]" in str(captured.value)
        assert isinstance(captured.value, ImportError)

    def test_guard_imports_dependency_only_after_explicit_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = ModuleType("tracking_test_dependency")
        imported: list[str] = []
        guard_module = importlib.import_module("lambdaforge.tracking.TrackingDependencyGuard")

        def import_dependency(name: str):
            imported.append(name)
            return sentinel

        monkeypatch.setattr(
            guard_module,
            "importlib",
            SimpleNamespace(
                util=SimpleNamespace(find_spec=lambda name: object()),
                import_module=import_dependency,
            ),
        )
        guard = TrackingDependencyGuard("tensorboard")

        assert guard.is_available
        assert imported == []
        guard.require()
        assert imported == []
        assert guard.import_dependency() is sentinel
        assert imported == ["tensorboard"]

    def test_tensorboard_guard_accepts_tensorboard_x_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = ModuleType("tensorboardX")
        imported: list[str] = []
        guard_module = importlib.import_module("lambdaforge.tracking.TrackingDependencyGuard")
        monkeypatch.setattr(
            guard_module,
            "importlib",
            SimpleNamespace(
                util=SimpleNamespace(
                    find_spec=lambda name: object() if name == "tensorboardX" else None
                ),
                import_module=lambda name: imported.append(name) or sentinel,
            ),
        )

        guard = TrackingDependencyGuard(TrackingBackend.TENSORBOARD)

        assert guard.is_available
        assert guard.import_dependency() is sentinel
        assert imported == ["tensorboardX"]

    def test_public_lightning_loggers_inherit_selected_logger_base(self) -> None:
        assert issubclass(Lightning.MLFlowLogger, Lightning.Logger)
        assert issubclass(Lightning.TensorBoardLogger, Lightning.Logger)
        assert issubclass(Lightning.WandbLogger, Lightning.Logger)
        backend_root = Lightning.module.__name__.split(".")[0]
        assert Lightning.MLFlowLogger.__module__.startswith(backend_root)
        assert Lightning.TensorBoardLogger.__module__.startswith(backend_root)
        assert Lightning.WandbLogger.__module__.startswith(backend_root)

    def test_framework_and_tracking_imports_do_not_load_optional_backends(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(repository / "src"), str(repository)])
        script = (
            "import sys; import lambdaforge; import lambdaforge.tracking as tracking; "
            "core='lambdaforge.tracking.TrackingBackend'; "
            "assert core not in sys.modules; "
            "blocked=('mlflow','tensorboard','tensorboardX','wandb'); "
            "loaded=[name for name in blocked if name in sys.modules]; "
            "assert not loaded, loaded; "
            "assert tracking.TrackingBackend.MLFLOW.value == 'mlflow'; "
            "assert core in sys.modules"
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

    def test_native_lightning_logger_aliases_do_not_import_optional_backends(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(repository / "src"), str(repository)])
        script = (
            "import sys; "
            "from lambdaforge.integrations.Lightning import Lightning; "
            "aliases=(Lightning.MLFlowLogger, Lightning.TensorBoardLogger, "
            "Lightning.WandbLogger); "
            "blocked=('mlflow','tensorboard','tensorboardX','wandb'); "
            "loaded=[name for name in blocked if name in sys.modules]; "
            "assert len(aliases) == 3; assert not loaded, loaded"
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
