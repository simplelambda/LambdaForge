"""Compatibility tests for modern and legacy Lightning package names."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from lambdaforge.integrations.Lightning import Lightning


class TestLightningCompatibility:
    """Verify that the centralized adapter selects one coherent backend."""

    @staticmethod
    def _modern_backend_is_available() -> bool:
        """Return whether the modern Lightning import path can be resolved."""
        try:
            return find_spec("lightning.pytorch") is not None
        except ModuleNotFoundError:
            return False

    def test_adapter_prefers_modern_package_and_falls_back_to_legacy(self) -> None:
        expected = (
            "lightning.pytorch" if self._modern_backend_is_available() else "pytorch_lightning"
        )

        assert Lightning.module.__name__ == expected
        assert Lightning.backend_name == expected

    def test_exported_types_belong_to_the_selected_backend(self) -> None:
        backend_root = Lightning.module.__name__.split(".")[0]

        assert Lightning.Callback.__module__.startswith(backend_root)
        assert Lightning.CSVLogger.__module__.startswith(backend_root)
        assert Lightning.EarlyStopping.__module__.startswith(backend_root)
        assert Lightning.Logger.__module__.startswith(backend_root)
        assert Lightning.ModelCheckpoint.__module__.startswith(backend_root)

    def test_modern_backend_dependency_errors_are_not_masked_by_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        fake_backend = tmp_path / "lightning" / "pytorch"
        fake_backend.mkdir(parents=True)
        (fake_backend.parent / "__init__.py").write_text("", encoding="utf-8")
        (fake_backend / "__init__.py").write_text(
            "import lambdaforge_test_missing_dependency\n",
            encoding="utf-8",
        )
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(tmp_path), str(repository / "src"), str(repository)]
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from lambdaforge.integrations.Lightning import Lightning",
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        assert completed.returncode != 0
        assert "lambdaforge_test_missing_dependency" in completed.stderr
