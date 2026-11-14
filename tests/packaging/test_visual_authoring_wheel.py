"""Reproducible dirty-tree wheel and installed public-operation/native Work acceptance."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_reproducible_wheel_and_installed_cpu_flow(tmp_path):
    root = Path(__file__).parents[2]
    builder = root / "tools/build_scientific_wheel.py"
    receipts = []
    for name in ("first", "second"):
        result = subprocess.run(
            [sys.executable, str(builder), str(tmp_path / name)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]
        receipts.append(json.loads((tmp_path / name / "build-receipt.json").read_text()))
    assert receipts[0]["wheel_sha256"] == receipts[1]["wheel_sha256"]
    assert receipts[0]["packaged_content_sha256"] == receipts[1]["packaged_content_sha256"]
    wheel = tmp_path / "first" / receipts[0]["wheel"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipts[0]["wheel_sha256"]
    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(installed),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(installed),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "LAMBDAFORGE_DATASET_REGISTRY": str(tmp_path / "datasets.json"),
    }
    # A new consumer, not the checkout and not an editable payload. Shipped sample is copied
    # into this fixture just as a client would generate/import its own ordinary Work module.
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    code = r"""
import importlib.util, json, pathlib, sys
import lambdaforge
installed = pathlib.Path(sys.argv[1])
assert pathlib.Path(lambdaforge.__file__).is_relative_to(installed)
from lambdaforge.catalog import OperationCatalog
assert "torch" not in sys.modules and "sklearn" not in sys.modules
catalog = OperationCatalog().discover()
assert "torch" not in sys.modules and "numpy" not in sys.modules
from importlib.resources import files
assert files("lambdaforge.schemas").joinpath("sample-schema-v1.json").is_file()
assert files("lambdaforge.schemas").joinpath("cluster-profile-v1.json").is_file()
assert files("lambdaforge.schemas").joinpath("table-schema-v1.json").is_file()
examples = installed / "share/lambdaforge/examples"
source = pathlib.Path("visual_research.py")
source.write_text((examples / source.name).read_text())
sys.path.insert(0, str(pathlib.Path.cwd()))
import visual_research
visual_research.create_raw(pathlib.Path("raw"))
from lambdaforge.etl import PreviewBudget, ReadRecords, ReadSQL
import sqlite3
db = pathlib.Path("fixture.sqlite")
with sqlite3.connect(db) as conn:
    conn.execute("CREATE TABLE samples(id TEXT PRIMARY KEY, x INTEGER)")
    conn.execute("INSERT INTO samples VALUES (?, ?)", ("one", 1))
assert ReadSQL("sqlite", file=str(db), table="samples")()[0]["id"] == "one"
pathlib.Path("large.csv").write_text("id,value\\none,1\\n".replace("\\n", chr(10)))
assert ReadRecords("csv").preview("large.csv")["rows"][0]["id"] == "one"
assert OperationCatalog().describe("WriteSQL")["preview"]["supported"] is False
project = chr(10).join(["[project]", 'name="installed-demo"', 'version="1"', ""])
pathlib.Path("pyproject.toml").write_text(project)
from lambdaforge.work import WorkConfig, WorkRunner
for name in ("visual_prepare.yaml", "visual_train.yaml", "visual_search.yaml"):
    path = pathlib.Path(name)
    path.write_text((examples / name).read_text())
    result = WorkRunner().run(WorkConfig.from_yaml(path))
    assert result.status == "succeeded", [(r.failure) for r in result.runs]
print(json.dumps({"installed": str(lambdaforge.__file__),
                  "operations": len(catalog["operations"])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(installed)],
        cwd=consumer,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    cli = subprocess.run(
        [sys.executable, "-m", "lambdaforge", "catalog", "--kind", "etl", "--json"],
        cwd=consumer,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode == 0, cli.stderr
    assert "ReadRecords" in {r["label"] for r in json.loads(cli.stdout)["operations"]}
