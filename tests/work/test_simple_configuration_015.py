from __future__ import annotations

import json
from pathlib import Path

from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.hpo.CandidateGenerator import DeterministicCandidateGenerator
from lambdaforge.hpo.SequentialSweep import PairedSweepSequentialAnalyzer
from lambdaforge.ProjectContext import ProjectContext
from lambdaforge.reproducibility.SeedProvider import SeedProvider
from lambdaforge.work.config import WorkConfig


def _project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="seed-tests"\nversion="1"\n'
        '[tool.lambdaforge]\nproject_id="stable-seed-tests"\n',
        encoding="utf-8",
    )
    return ProjectContext.discover(tmp_path)


def test_project_seed_stream_is_stable_disjoint_and_numpy_compatible(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = SeedProvider(project).stream("replicate").take(1000)
    restarted = SeedProvider(ProjectContext.discover(tmp_path)).stream("replicate").take(1000)
    confirmation = SeedProvider(project).stream("confirmation").take(1000)

    assert first == restarted
    assert len({value.value for value in first}) == 1000
    assert {value.value for value in first}.isdisjoint(value.value for value in confirmation)
    assert all(0 <= value.value <= 2**32 - 1 for value in (*first, *confirmation))


def test_minimal_adaptive_yaml_resolves_every_automatic_policy(tmp_path: Path) -> None:
    _project(tmp_path)
    source = tmp_path / "minimal-study.yaml"
    source.write_text(
        """run: tests.work_cases.AdaptiveScoreWork
search:
  space:
    quality: {range: [0.0, 1.0]}
objective: auprc
resources: {gpu: 2}
""",
        encoding="utf-8",
    )

    config = WorkConfig.from_yaml(source)
    definition = config.levels[0].runs[0]
    resolved = config.resolved_configuration()["levels"][0][0]

    assert config.name == "minimal-study"
    assert definition.seeds[0] is not None
    assert definition.search_policy is not None
    assert definition.search_policy.min_seeds == 1
    assert definition.search_policy.candidate_budget is None
    assert definition.search_policy.goal == "balanced"
    assert definition.search_policy.confirmation_auto
    assert definition.execution_policy.runs_per_gpu is None
    assert definition.execution_policy.max_parallel is None
    assert definition.objective == {"metric": "auprc", "mode": "max", "range": [0.0, 1.0]}
    assert resolved["search"]["candidate_generation"]["mode"] == "deterministic-incremental"
    assert resolved["design"]["seed_source"]["resolved"][0]["ordinal"] == 0


def test_explicit_seeds_and_fixed_sweep_replicates_remain_authoritative(tmp_path: Path) -> None:
    _project(tmp_path)
    explicit = WorkConfig.from_mapping(
        {
            "run": "tests.work_cases.AdaptiveScoreWork",
            "seeds": [4, 7, 32],
            "search": {"trials": 2, "quality": [0.0, 1.0]},
            "objective": {"metric": "score", "mode": "max"},
        },
        source=tmp_path / "explicit.yaml",
    ).levels[0].runs[0]
    fixed = WorkConfig.from_mapping(
        {
            "run": "tests.work_cases.AdaptiveScoreWork",
            "sweep": {"replicates": 3, "space": {"quality": [0.0, 1.0]}},
            "objective": {"metric": "accuracy", "mode": "max"},
        },
        source=tmp_path / "fixed.yaml",
    ).levels[0].runs[0]

    assert explicit.seeds == (4, 7, 32)
    assert [value["role"] for value in explicit.seed_metadata] == ["explicit"] * 3
    assert fixed.study_design is not None
    assert fixed.study_design.replication == "fixed"
    assert fixed.run_count == 6
    assert [value["ordinal"] for value in fixed.seed_metadata] == [0, 1, 2]


def test_auto_sweep_uses_shared_blocks_and_anytime_valid_decision(tmp_path: Path) -> None:
    _project(tmp_path)
    definition = WorkConfig.from_mapping(
        {
            "run": "tests.work_cases.AdaptiveScoreWork",
            "sweep": {"space": {"quality": [0.0, 1.0]}},
            "objective": "accuracy",
        },
        source=tmp_path / "auto-sweep.yaml",
    ).levels[0].runs[0]
    assert definition.study_design is not None
    assert definition.study_design.replication == "auto-blocks"
    assert definition.run_count == 2
    assert {requirement.seed for requirement in definition.study_design.evidence.required} == {
        definition.seeds[0]
    }

    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    collecting = analyzer.evaluate({1: {1: 0.0}, 2: {1: 1.0}})
    assert not collecting.stop
    values = {1: {}, 2: {}}
    decision = collecting
    for seed in range(1, 200):
        values[1][seed] = 0.0
        values[2][seed] = 1.0
        decision = analyzer.evaluate(values)
        if decision.stop:
            break
    assert decision.stop
    assert decision.conclusion == "PREFERRED"
    assert decision.policy_version == "paired-hoeffding-cs-v1"


def test_candidate_generator_extension_preserves_existing_prefix() -> None:
    generator = DeterministicCandidateGenerator(
        {
            "width": {"values": [32, 64, 128]},
            "dropout": {"range": [0.0, 0.5]},
        }
    )
    first = generator.prefix(12)
    assert generator.prefix(30)[:12] == first
    assert generator.extend(12, 18) == generator.prefix(30)[12:]


def test_config_resolve_and_seed_commands_are_machine_readable(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _project(tmp_path)
    source = tmp_path / "commands.yaml"
    source.write_text(
        "run: tests.work_cases.AdaptiveScoreWork\n"
        "search:\n  space:\n    quality: [0.0, 1.0]\n"
        "objective: accuracy\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert CommandLineInterface.main(["seeds", "--count", "2", "--json"]) == 0
    seeds = json.loads(capsys.readouterr().out)
    assert [value["ordinal"] for value in seeds["seeds"]] == [0, 1]
    assert CommandLineInterface.main(["config", "resolve", str(source), "--json"]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["name"] == "commands"
    assert resolved["levels"][0][0]["execution"]["runs_per_gpu"] == "auto"
