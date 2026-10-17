"""Packaged assets, personalized-file protection and native fixture validation."""

import importlib
import inspect
import json
import re
import shlex
import sys
from importlib.resources import files

import pytest
import yaml

from lambdaforge.agents.integration import (
    IntegrationService,
    example_assets,
    skill_metadata,
    skill_names,
)
from lambdaforge.cli.scaffold import initialize
from lambdaforge.work.config import WorkConfig


def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.1"\n')
    return tmp_path


def test_exactly_24_specialized_skills_have_valid_frontmatter_and_local_references():
    names = skill_names()
    assert len(names) == 24 and len(set(names)) == 24
    for name in names:
        metadata = skill_metadata(name)
        assert metadata["name"] == name
        assert len(metadata["description"]) > 60
        assert "references/procedure.md" in metadata["document"]
        assert "## Prerequisites and input contract" in metadata["document"]
        assert "## Validation" in metadata["document"]
    assert {
        "work.py",
        "work.yaml",
        "study.yaml",
        "repeated.yaml",
        "inputs.yaml",
    } <= example_assets().keys()


def test_skill_procedures_use_actual_cli_grammar_and_facade_names_without_execution():
    from lambdaforge.agents.services import AgentServices
    from lambdaforge.cli.parser import build_parser

    parser = build_parser()
    commands = 0
    tool_names = 0
    root = files("lambdaforge.agents").joinpath("assets/skills")
    for name in skill_names():
        procedure = root.joinpath(name, "references/procedure.md").read_text(encoding="utf-8")
        for block in re.findall(r"```bash\n(.*?)\n```", procedure, flags=re.DOTALL):
            for line in block.splitlines():
                if line.startswith("lf "):
                    # The native CLI removes these global switches before parser dispatch.
                    arguments = [
                        a
                        for a in shlex.split(line)[1:]
                        if a not in {"--json", "--debug", "--verbose"}
                    ]
                    parser.parse_args(arguments)
                    commands += 1
        for method in re.findall(r"`([a-z_]+)(?:\([^`]*\))?`", procedure):
            if method.startswith(("get_", "list_", "knowledge_", "prepare_", "request_", "apply_")):
                assert callable(getattr(AgentServices, method, None)), (name, method)
                tool_names += 1
    assert commands >= 30 and tool_names >= 15


def test_apply_is_idempotent_preview_is_read_only_and_customization_is_preserved(tmp_path):
    integration = IntegrationService(project(tmp_path))
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    plan = integration.preview(agents="codex", knowledge="obsidian")
    assert before == sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    applied = integration.apply(plan_hash=plan["plan_hash"], agents="codex", knowledge="obsidian")
    assert applied["applied"] and not applied["client_configuration_modified"]
    assert (tmp_path / "knowledge/.obsidian").is_dir()
    (tmp_path / "AGENTS.md").write_text("My researcher rules\n")
    skill = tmp_path / ".agents/skills/lambdaforge-work/SKILL.md"
    skill.write_text("Personalized procedure\n")
    second = integration.preview(agents="codex", knowledge="obsidian")
    kept = integration.apply(plan_hash=second["plan_hash"], agents="codex", knowledge="obsidian")
    assert {"AGENTS.md", ".agents/skills/lambdaforge-work/SKILL.md"} <= set(
        kept["custom_files_preserved"]
    )
    assert (tmp_path / "AGENTS.md").read_text() == "My researcher rules\n"
    assert skill.read_text() == "Personalized procedure\n"
    assert not (tmp_path / ".codex").exists()
    assert integration.status()["installed_skills"] == 24


def test_stale_plan_symlink_and_corrupt_manifest_are_rejected(tmp_path):
    integration = IntegrationService(project(tmp_path))
    plan = integration.preview()
    (tmp_path / "AGENTS.md").write_text("New researcher rules")
    with pytest.raises(ValueError, match="changed"):
        integration.apply(plan_hash=plan["plan_hash"])
    (tmp_path / ".agents").symlink_to(tmp_path.parent)
    with pytest.raises(PermissionError):
        integration.preview()
    (tmp_path / ".agents").unlink()
    manifest = tmp_path / ".lambdaforge/agents/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="manifest"):
        integration.preview()


def test_legacy_init_and_opt_in_consumer_setup(tmp_path):
    plain = tmp_path / "plain"
    initialize(plain, force=False)
    assert not (plain / "AGENTS.md").exists()
    assert not (plain / "knowledge").exists()
    prepared = tmp_path / "prepared"
    initialize(prepared, force=False, agents="portable", knowledge="markdown")
    assert (prepared / "AGENTS.md").is_file()
    assert len(list((prepared / ".agents/skills").glob("*/SKILL.md"))) == 24
    assert not (prepared / "knowledge/.obsidian").exists()


def test_packaged_work_and_study_examples_validate_native_signatures(tmp_path, monkeypatch):
    project(tmp_path)
    module = tmp_path / "agent_examples"
    module.mkdir()
    (module / "__init__.py").write_text("")
    assets = example_assets()
    (module / "work.py").write_text(assets["work.py"])
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    try:
        for name in ("work.yaml", "repeated.yaml", "study.yaml"):
            config = tmp_path / name
            config.write_text(assets[name])
            WorkConfig.validate_file(config)
        from lambdaforge.work import WorkRunner

        monkeypatch.chdir(tmp_path)
        for variable in (
            "LAMBDAFORGE_DATASET_REGISTRY",
            "LAMBDAFORGE_PRODUCT_ROOT",
            "LAMBDAFORGE_RUN_ROOT",
            "LAMBDAFORGE_STORAGE_POLICY",
        ):
            monkeypatch.delenv(variable, raising=False)
        result = WorkRunner().run(WorkConfig.from_yaml(tmp_path / "work.yaml"))
        assert result.status == "succeeded"
        for name, target, arguments in (
            (
                "history-reader",
                "ReadHistorical",
                {"previous": {"result": {"execution": result.execution_id}}},
            ),
            (
                "product-reader",
                "ReadProduct",
                {
                    "report": {
                        "product": {"name": "example-report", "contract": "example/agent-report:v1"}
                    }
                },
            ),
        ):
            consumer = WorkConfig.from_mapping(
                {
                    "name": name,
                    "run": "agent_examples.work." + target,
                    "with": arguments,
                    "resources": {"cpu": 1},
                },
                source=tmp_path / f"{name}.yaml",
            )
            assert WorkRunner().run(consumer).status == "succeeded"
        from lambdaforge.agents import AgentServices, KnowledgeService, ProjectScope

        evidence = AgentServices(tmp_path).get_result(result.execution_id)
        assert evidence.status == "ok"
        vault = KnowledgeService(ProjectScope(tmp_path))
        vault.initialize()
        note = vault.create(
            "decision",
            "Keep deterministic baseline",
            author_type="ai",
            body="Observed score 0.5; not a model benchmark.",
            evidence_refs=[{"kind": "execution", "id": result.execution_id}],
        )
        another_session = AgentServices(tmp_path)
        assert another_session.knowledge_search("baseline").status == "ok"
        assert another_session.knowledge_read(note["metadata"]["id"]).status == "ok"
        from agent_examples.work import Example

        assert "value" in inspect.signature(Example.run).parameters
    finally:
        sys.modules.pop("agent_examples.work", None)
        sys.modules.pop("agent_examples", None)
    bindings = yaml.safe_load(assets["inputs.yaml"])
    assert set(bindings) == {"dataset", "history", "model"}
