"""Native product commands reuse the public registry and keep preview read-only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.products import ProductContract, ProductRegistry, StudyProduct


def test_native_publication_preview_and_inspection(tmp_path: Path, capsys: object) -> None:
    product = StudyProduct(
        "report",
        "ScientificReport",
        ProductContract("example/report:v1", ("meaning",)),
        {"value": 42},
        {"meaning": "count"},
        {"execution_id": "execution-1", "evidence_fingerprint": "evidence-1"},
    )
    manifest = tmp_path / "report.json"
    manifest.write_text(json.dumps(product.to_dict()))
    root = tmp_path / "products"
    for extra in ([], ["--apply"]):
        assert (
            CommandLineInterface.main(
                ["products", "publish", str(manifest), "--root", str(root), "--json", *extra]
            )
            == 0
        )
        if not extra:
            assert not root.exists()
    assert ProductRegistry(root).show("report") == product
    for operation in ("show", "verify", "provenance"):
        assert (
            CommandLineInterface.main(
                ["products", operation, "report", "--root", str(root), "--json"]
            )
            == 0
        )
    assert CommandLineInterface.main(["products", "list", "--root", str(root), "--json"]) == 0


def test_product_help_and_empty_listing_create_nothing(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    assert CommandLineInterface.main(["products", "--help"]) == 0
    assert CommandLineInterface.main(["products", "list", "--root", str(root)]) == 0
    assert not root.exists()


def test_metadata_import_does_not_load_torch_or_scientific_analysis() -> None:
    code = (
        "import sys; from lambdaforge.products import ProductContract, ProductRegistry; "
        "assert 'torch' not in sys.modules; "
        "assert 'lambdaforge.analysis.StudyAnalysis' not in sys.modules; "
        "assert 'lambdaforge.work.runner' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_help_does_not_import_executor_analysis_or_torch() -> None:
    code = (
        "import sys; from lambdaforge.cli.CommandLineInterface import CommandLineInterface; "
        "assert CommandLineInterface.main(['--help']) == 0; "
        "assert 'torch' not in sys.modules; "
        "assert 'lambdaforge.analysis.StudyAnalysis' not in sys.modules; "
        "assert 'lambdaforge.work.runner' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
