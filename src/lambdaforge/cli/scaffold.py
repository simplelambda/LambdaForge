"""Consumer-project scaffold templates and safe initialization."""

from __future__ import annotations

from pathlib import Path

from packaging.version import Version

from lambdaforge._version import VERSION


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
        "src/my_project/__init__.py": '"""Project-local scientific Work classes."""\n',
        "src/my_project/work.py": '''"""Scientific Work managed by LambdaForge."""

import lambdaforge as lf


class Example(lf.Work):
    """Write one small artifact and return the final structured result."""

    def run(self, message: str = "ready") -> dict[str, str]:
        path = self.run_dir / "output.txt"
        path.write_text(message + "\\n", encoding="utf-8")
        self.outputs.artifact("output", path, role="report", media_type="text/plain")
        return {"status": message}
''',
        "experiments/work.yaml": """name: example
run: my_project.work.Example
with:
  message: ready
resources:
  cpu: 1
""",
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
        "src/my_project/preprocessing.py": '''"""Project preprocessing Work."""

import json
from pathlib import Path

import lambdaforge as lf


class Preprocess(lf.Work):
    """Process JSONL with normal Python and register the resulting directory."""
    def run(self, source: Path) -> dict[str, int]:
        output = self.run_dir / "processed.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        output.write_text("".join(json.dumps(row) + "\\n" for row in records))
        self.metrics.log("records", len(records))
        self.outputs.artifact("processed", output, role="dataset")
        return {"records": len(records)}
''',
        "data/raw.jsonl": '{"id": "example", "value": 1}\n',
        "experiments/preprocessing.yaml": """name: prepare-data
run: my_project.preprocessing.Preprocess
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


class Train(lf.Work):
    """Replace this tiny loop with the project's real model and dataloaders."""
    def run(self, hidden_dim: int, epochs: int = 2) -> dict[str, float]:
        model = torch.nn.Sequential(torch.nn.Linear(4, hidden_dim), torch.nn.Linear(hidden_dim, 1))
        optimizer = torch.optim.AdamW(model.parameters())
        loss_value = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = model(torch.randn(8, 4)).square().mean()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            self.metrics.log("loss", loss_value, step=epoch, split="train")
        return {"final_loss": loss_value, "seed": float(self.seed or 0)}
''',
        "experiments/training.yaml": """name: baseline
run: my_project.training.Train
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
        files.pop("src/my_project/work.py")
        files.pop("experiments/work.yaml")
    entry = {
        "minimal": "experiments/work.yaml",
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
lambdaforge explain {entry}
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
