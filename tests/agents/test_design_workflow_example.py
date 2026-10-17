"""Packaged Study instructions contain executable native authoring examples."""

import os
import re
import shlex
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import lambdaforge
from lambdaforge.cli.parser import build_parser


def test_packaged_design_workflows_validate_fixed_repeated_adaptive_without_running(tmp_path):
    reference = (
        files("lambdaforge.agents")
        .joinpath("assets/skills/lambdaforge-study/references/design-workflows.md")
        .read_text(encoding="utf-8")
    )
    python = re.findall(r"```python\n(.*?)\n```", reference, flags=re.DOTALL)[0]
    yaml = re.findall(r"```yaml\n(.*?)\n```", reference, flags=re.DOTALL)[0]
    package = tmp_path / "my_project"
    package.mkdir()
    (package / "__init__.py").write_text(python)
    (tmp_path / "study.yaml").write_text(yaml)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="skill-example"\nversion="1"\n')
    source_root = Path(lambdaforge.__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(source_root)}
    code = """
from pathlib import Path
import yaml
from lambdaforge.work.config import WorkConfig
raw = yaml.safe_load(Path('study.yaml').read_text())
assert WorkConfig.validate_file('study.yaml').valid
fixed = WorkConfig.from_yaml('study.yaml')
assert fixed.levels[0].runs[0].work_class == 'my_project.Training'
repeated = dict(raw)
repeated.pop('sweep')
repeated['with'] = {**repeated['with'], 'strength': 0.1}
assert WorkConfig.from_mapping(repeated, source=Path('repeated.yaml'))
adaptive = dict(raw)
adaptive.pop('sweep')
adaptive['search'] = {'space': {'strength': {'range': [0.0, 1.0]}}}
assert WorkConfig.from_mapping(adaptive, source=Path('adaptive.yaml'))
assert not list(Path('.').rglob('execution.json'))
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    parser = build_parser()
    for block in re.findall(r"```bash\n(.*?)\n```", reference, flags=re.DOTALL):
        for line in block.splitlines():
            if line.startswith("lf "):
                parser.parse_args([word for word in shlex.split(line)[1:] if word != "--json"])
