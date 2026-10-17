"""Consumer-project scaffold templates and safe initialization."""

from __future__ import annotations

import hashlib
import keyword
import re
import sys
import unicodedata
from pathlib import Path

from packaging.version import Version

from lambdaforge._version import VERSION


def _framework_requirement() -> str:
    """Return the compatible minor release range for generated consumers."""
    release = Version(VERSION).release
    major = release[0]
    minor = release[1] if len(release) > 1 else 0
    return f"lambdaforge>={major}.{minor},<{major}.{minor + 1}"


def project_names(name: str) -> tuple[str, str]:
    """Derive safe distribution/import names, avoiding keywords and framework/stdlib shadows."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    distribution = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:64] or "project"
    package = distribution.replace("-", "_")
    if package[0].isdigit():
        package = "project_" + package
    reserved = set(sys.stdlib_module_names) | {
        "lambdaforge",
        "torch",
        "numpy",
        "scipy",
        "sklearn",
        "lightning",
        "pytest",
        "packaging",
    }
    if keyword.iskeyword(package) or package in reserved:
        package += "_project"
    return distribution, package


def initialize(
    directory: Path,
    *,
    force: bool,
    template: str = "minimal",
    agents: str = "none",
    knowledge: str = "none",
    interactive: bool = False,
    install_deps: bool = False,
) -> int:
    """Create a minimal installable consumer project without overwriting by default."""
    resolved = directory.expanduser().resolve()
    if directory.expanduser().absolute() != resolved:
        raise ValueError("Scaffold destination must not contain symlinks.")
    if interactive:
        if not sys.stdin.isatty():
            raise ValueError("Interactive init needs a TTY; use --agents and --knowledge instead.")
        agents = input("Agent setup [none/portable/codex] (portable): ").strip() or "portable"
        knowledge = (
            input("Project memory [none/markdown/obsidian] (markdown): ").strip() or "markdown"
        )
    if agents not in {"none", "portable", "codex"} or knowledge not in {
        "none",
        "markdown",
        "obsidian",
    }:
        raise ValueError("Invalid agent or knowledge mode; no project was written.")
    if install_deps and agents == "none":
        raise ValueError("--install-deps requires --agents portable/codex; no project was written.")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", resolved.name).strip("-")[:40] or "project"
    project_id = f"{slug}-{hashlib.sha256(str(resolved).encode()).hexdigest()[:16]}"
    distribution, package = project_names(resolved.name)
    files = {
        "pyproject.toml": f"""[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "my-ai-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["{_framework_requirement()}"]

[tool.lambdaforge]
project_id = "{project_id}"

[tool.setuptools.packages.find]
where = ["src"]
""",
        "src/my_project/__init__.py": '"""Project-local scientific Work classes."""\n',
        "src/my_project/work.py": '''"""Scientific Work managed by LambdaForge."""

import lambdaforge as lf


class Example(lf.Work):
    """Write one small artifact and return the final structured result."""

    def run(self, message: str = "ready") -> dict[str, str]:
        report = self.outputs.file("output", filename="output.txt", role="report")
        report.write_text(message + "\\n")
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
        output = self.outputs.file("processed", filename="processed.jsonl", role="dataset")
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        output.write_text("".join(json.dumps(row) + "\\n" for row in records))
        self.metrics.log("records", len(records))
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
    # Render the existing templates, not another execution model. Existing projects are
    # unchanged unless the operator explicitly requests --force.
    files = {
        relative.replace("src/my_project/", f"src/{package}/"): content.replace(
            "my-ai-project", distribution
        ).replace("my_project", package)
        for relative, content in files.items()
    }
    module, target = {
        "minimal": ("work", "Example"),
        "preprocessing": ("preprocessing", "Preprocess"),
        "training": ("training", "Train"),
        "full": ("preprocessing", "Preprocess"),
    }[template]
    files["tests/test_work_signature.py"] = (
        "from lambdaforge import Work\n"
        f"from {package}.{module} import {target}\n\n\n"
        "def test_native_work_contract():\n"
        f"    assert issubclass({target}, Work)\n"
        f"    assert callable({target}.run)\n"
    )
    files["README.md"] = f"""# {distribution}

Create an environment and install both packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
lf doctor
lf validate {entry}
lf explain {entry}
lf run {entry} --dry-run
lf run {entry}
lf
```

Package: `{package}`. Scientific entry point: `{package}.{module}.{target}.run`.
The dry-run validates/plans; only the subsequent `lf run` executes science.
Optional: `lf clusters add --help` for remote profiles, `lf ai doctor` for agent setup.
Tests are in `tests/` (run with pytest when installed).
No cluster or agent is configured automatically.
"""
    collisions = [directory / relative for relative in files if (directory / relative).exists()]
    if collisions and not force:
        raise ValueError(f"Refusing to overwrite {collisions[0]}; use --force if intentional.")
    for relative in files:
        path = resolved / relative
        if path.resolve() != path:
            raise ValueError(f"Refusing a symbolic scaffold path: {path}")
    for relative, content in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Initialized LambdaForge consumer project: {directory.resolve()}")
    print(f"Package: {package} · distribution: {distribution} · template: {template}")
    print(f"Next: cd {resolved}\n  python -m pip install -e .\n  lf validate {entry}")
    print(f"  lf explain {entry}\n  lf run {entry} --dry-run\n  lf")
    if agents != "none" or knowledge != "none":
        from lambdaforge.agents.integration import IntegrationService

        integration = IntegrationService(resolved)
        plan = integration.preview(agents=agents, knowledge=knowledge)
        applied = integration.apply(plan_hash=plan["plan_hash"], agents=agents, knowledge=knowledge)
        print(f"Agent assets prepared; {len(applied['custom_files_preserved'])} custom files kept.")
        if agents == "codex":
            print(
                "Review .lambdaforge/agents/codex-config.example.toml; client config is unchanged."
            )
        if install_deps:
            from lambdaforge.agents.dependencies import DependencyService

            dependencies = DependencyService(integration.scope)
            dependency_plan = dependencies.preview()
            print(
                "Installing declared optional agent dependencies in "
                + dependency_plan["interpreter"]
            )
            installed = dependencies.install(plan_hash=dependency_plan["plan_hash"], consent=True)
            print("Agent dependencies: " + installed["status"])
            if installed["status"] == "error":
                print(installed["diagnostic"], file=sys.stderr)
                print(
                    "Project files are preserved. Use lf ai dependencies check/install to repair.",
                    file=sys.stderr,
                )
                return 1
        else:
            print(
                "Optional MCP SDK is not installed automatically: pip install 'lambdaforge[agents]'"
            )
    return 0
