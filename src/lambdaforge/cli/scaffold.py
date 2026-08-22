"""Consumer-project scaffold templates and safe initialization."""

from __future__ import annotations

import json
from pathlib import Path

from packaging.version import Version

from lambdaforge._version import VERSION
from lambdaforge.configuration.AuthoringSchemaCatalog import AuthoringSchemaCatalog
from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog


def _framework_requirement() -> str:
    """Return the compatible minor release range for generated consumers."""
    release = Version(VERSION).release
    major = release[0]
    minor = release[1] if len(release) > 1 else 0
    return f"lambdaforge>={major}.{minor},<{major}.{minor + 1}"


def initialize(directory: Path, *, force: bool, template: str = "minimal") -> int:
    """Create a minimal installable consumer project without overwriting by default."""
    files = {
        "pyproject.toml": f"""[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["{_framework_requirement()}"]

[tool.setuptools.packages.find]
where = ["src"]
""",
        "src/my_project/__init__.py": '"""Project-local datasets, models and tasks."""\n',
        "src/my_project/tasks.py": '''"""Ordinary project functions executed by LambdaForge."""

import json

import lambdaforge as lf


def example(message: str = "ready") -> dict[str, str]:
    """Write one small artifact and return the final structured result."""
    path = lf.current().run_dir / "output.json"
    path.write_text(json.dumps({"status": message}) + "\\n", encoding="utf-8")
    lf.artifact("output", path, role="report", media_type="application/json")
    return {"status": message}
''',
        "experiments/task.yaml": """name: example
run: my_project.tasks.example
with:
  message: ready
resources:
  cpu: 1
""",
        "README.md": """# My AI project

Create an environment, install LambdaForge and this package, then run:

```bash
python -m pip install -e .
lambdaforge validate experiments/task.yaml
lambdaforge inspect experiments/task.yaml
lambdaforge run experiments/task.yaml --dry-run
lambdaforge run experiments/task.yaml
```
""",
        "schemas/lambdaforge-authoring.schema.json": json.dumps(
            AuthoringSchemaCatalog().schema(), indent=2
        )
        + "\n",
        # Retained for editors/tools that still open strict compatible task documents.
        "schemas/lambdaforge-task.schema.json": json.dumps(TaskSchemaCatalog().schema(), indent=2)
        + "\n",
        ".vscode/settings.json": json.dumps(
            {"yaml.schemas": {"./schemas/lambdaforge-authoring.schema.json": "experiments/*.yaml"}},
            indent=2,
        )
        + "\n",
        ".gitignore": """.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.hypothesis/
.ipynb_checkpoints/
.lambdaforge/
.env
.env.*
!.env.example
runs/
dist/
build/
*.egg-info/
*.whl
lambdaforge-dashboard.html
slurm-*.out
slurm-*.err
""",
    }
    preprocessing_files = {
        "src/my_project/preprocessing.py": '''"""Project preprocessing functions."""

import json
from pathlib import Path

import lambdaforge as lf


def preprocess(source: Path) -> dict[str, int]:
    """Process JSONL with normal Python and register the resulting directory."""
    output = lf.current().run_dir / "processed.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    output.write_text("".join(json.dumps(row) + "\\n" for row in records))
    lf.metric("records", len(records))
    lf.artifact("processed", output, role="dataset")
    return {"records": len(records)}
''',
        "data/raw.jsonl": '{"id": "example", "value": 1}\n',
        "experiments/preprocessing.yaml": """name: prepare-data
run: my_project.preprocessing.preprocess
with:
  source:
    file: ../data/raw.jsonl
resources:
  cpu: 2
  memory: 1GiB
""",
    }
    training_files = {
        "src/my_project/training.py": '''"""Small ordinary training function."""

import lambdaforge as lf
import torch


def train(hidden_dim: int, epochs: int = 2, seed: int = 0) -> dict[str, float]:
    """Replace this tiny loop with the project's real model and dataloaders."""
    model = torch.nn.Sequential(torch.nn.Linear(4, hidden_dim), torch.nn.Linear(hidden_dim, 1))
    optimizer = torch.optim.AdamW(model.parameters())
    loss_value = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(torch.randn(8, 4)).square().mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        lf.metric("loss", loss_value, step=epoch, split="train")
    return {"final_loss": loss_value, "seed": float(seed)}
''',
        "experiments/training.yaml": """name: baseline
run: my_project.training.train
with:
  hidden_dim: 16
  epochs: 2
seeds: [7]
resources:
  cpu: 1
""",
    }
    if template in {"preprocessing", "full"}:
        files.update(preprocessing_files)
    if template in {"training", "full"}:
        files.update(training_files)
    if template in {"preprocessing", "training"}:
        files.pop("src/my_project/tasks.py")
        files.pop("experiments/task.yaml")
    entry = {
        "minimal": "experiments/task.yaml",
        "preprocessing": "experiments/preprocessing.yaml",
        "training": "experiments/training.yaml",
        "full": "experiments/preprocessing.yaml",
    }[template]
    files["README.md"] = f"""# My AI project

Create an environment and install both packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e /absolute/path/to/LambdaForge
python -m pip install -e .
lambdaforge doctor
lambdaforge validate {entry}
lambdaforge inspect {entry} --resolved
lambdaforge run {entry} --dry-run
lambdaforge run {entry}
```
"""
    collisions = [directory / relative for relative in files if (directory / relative).exists()]
    if collisions and not force:
        raise ValueError(f"Refusing to overwrite {collisions[0]}; use --force if intentional.")
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Initialized LambdaForge consumer project: {directory.resolve()}")
    return 0
