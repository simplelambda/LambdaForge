"""Consumer-project scaffold templates and safe initialization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from lambdaforge.tasks.TaskSchemaCatalog import TaskSchemaCatalog


def initialize(directory: Path, *, force: bool, template: str = "minimal") -> int:
    """Create a minimal installable consumer project without overwriting by default."""
    files = {
        "pyproject.toml": """[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["lambdaforge>=0.7,<0.8"]

[tool.setuptools.packages.find]
where = ["src"]
""",
        "src/my_project/__init__.py": '"""Project-local datasets, models and tasks."""\n',
        "src/my_project/tasks.py": '''"""Project task implementations."""

import json

from lambdaforge.tasks import ArtifactDeclaration, Task, TaskContext, TaskOutput


class ExampleTask(Task):
    """Write one small JSON artifact through the public task contract."""

    def run(self, context: TaskContext) -> TaskOutput:
        """Create the configured example output."""
        path = context.output_path("output.json", create_parent=True)
        path.write_text(json.dumps({"status": "ready"}) + "\\n", encoding="utf-8")
        return TaskOutput(
            outputs={"status": "ready"},
            artifacts=[ArtifactDeclaration("output.json", media_type="application/json")],
        )
''',
        "experiments/task.yaml": """kind: task
schema_version: "1.0"
name: example
task:
  target: my_project.tasks.ExampleTask
required_artifacts: [output.json]
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
        "schemas/lambdaforge-task.schema.json": json.dumps(TaskSchemaCatalog().schema(), indent=2)
        + "\n",
        ".vscode/settings.json": json.dumps(
            {"yaml.schemas": {"./schemas/lambdaforge-task.schema.json": "experiments/task*.yaml"}},
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


def normalize_record(value: object) -> object:
    """Replace this example with the domain transformation."""
    return value
''',
        "data/raw.jsonl": '{"id": "example", "value": 1}\n',
        "experiments/preprocessing.yaml": """name: prepare-data
inputs:
  raw: ../data/raw.jsonl
outputs:
  processed: processed
preprocess:
  function: my_project.preprocessing.normalize_record
  input: raw
  output: processed
  key_field: id
  workers: 2
  workload: io
""",
    }
    training_files = {
        "src/my_project/models.py": '''"""Project model definitions."""

from torch import Tensor, nn


class ProjectModel(nn.Module):
    """Tiny baseline to replace with the research model."""

    def __init__(self, in_features: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return one binary logit per sample."""
        return self.head(x)
''',
        "src/my_project/data.py": '''"""Project dataset definitions."""

import torch
from torch.utils.data import Dataset


class ProjectDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy data to verify the complete training path."""

    def __init__(self, split: str, size: int = 32) -> None:
        generator = torch.Generator().manual_seed(7 if split == "train" else 17)
        self.x = torch.randn(size, 4, generator=generator)
        self.target = (self.x.sum(dim=1, keepdim=True) > 0).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "target": self.target[index]}
''',
        "experiments/training.yaml": """schema_version: "1.1"
experiment:
  name: baseline
  output_root: ../runs/experiments
  seeds: [7]
data:
  train: {target: my_project.data.ProjectDataset, params: {split: train}}
  val: {target: my_project.data.ProjectDataset, params: {split: val}}
  datamodule:
    target: lambdaforge.training.data.LightningDataModule
    params: {batch_size: 8, num_workers: 0}
model: {target: my_project.models.ProjectModel}
losses:
  - target: lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss
    params: {output_key: logits, target_key: target}
val_metrics:
  - target: lambdaforge.metrics.BinaryAccuracy
    params: {pred_key: logits, target_key: target}
optimizer: {ref: torch.optim.AdamW, params: {lr: 0.001}}
task:
  target: lambdaforge.training.LightningTask
  params: {model_input_key: x, model_output_key: logits}
trainer: {max_epochs: 2, accelerator: auto, devices: auto}
execution: {mode: sequential}
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
        print(f"ERROR: refusing to overwrite {collisions[0]}; use --force.", file=sys.stderr)
        return 1
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Initialized LambdaForge consumer project: {directory.resolve()}")
    return 0
