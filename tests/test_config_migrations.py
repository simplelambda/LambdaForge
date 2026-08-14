"""Versioned configuration migrations, previews and compatibility boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml
from ruamel.yaml.composer import ComposerError
from ruamel.yaml.constructor import DuplicateKeyError

from lambdaforge import LambdaForge
from lambdaforge.cli.CommandLineInterface import CommandLineInterface
from lambdaforge.experiments import ExperimentConfig, ExperimentValidator, RunLoader
from lambdaforge.experiments.migrations.ExperimentConfigMigration import (
    ExperimentConfigMigration,
)
from lambdaforge.experiments.migrations.ExperimentConfigMigrationRegistry import (
    ExperimentConfigMigrationRegistry,
)
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)
from lambdaforge.experiments.migrations.ExperimentSchemaCatalog import (
    ExperimentSchemaCatalog,
)
from lambdaforge.experiments.migrations.ExperimentSchemaVersion import (
    ExperimentSchemaVersion,
)
from lambdaforge.experiments.migrations.ExperimentV1ToV1_1Migration import (
    ExperimentV1ToV1_1Migration,
)
from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
    MigrationPreviewFormat,
)
from lambdaforge.experiments.migrations.RoundTripYamlCodec import RoundTripYamlCodec


class TestConfigMigrations:
    """Verify deterministic, preview-first migrations without runtime side effects."""

    class _SyntheticMigration(ExperimentConfigMigration):
        """Parameterizable migration used to exercise registry graph behavior."""

        def __init__(self, identifier: str, source: str, target: str) -> None:
            self._identifier = identifier
            self._source_version = ExperimentSchemaVersion(source)
            self._target_version = ExperimentSchemaVersion(target)

        @property
        def identifier(self) -> str:
            """Return the test-owned stable identifier."""
            return self._identifier

        @property
        def source_version(self) -> ExperimentSchemaVersion:
            """Return the configured source version."""
            return self._source_version

        @property
        def target_version(self) -> ExperimentSchemaVersion:
            """Return the configured target version."""
            return self._target_version

        @property
        def description(self) -> str:
            """Return a deterministic test description."""
            return f"Advance {self.source_version} to {self.target_version}."

        def apply(self, config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
            """Advance the version in one isolated test mapping."""
            config["schema_version"] = self.target_version.value
            return config

    @staticmethod
    def valid_config(
        *,
        versioned: bool,
        schema_version: str = "1.1",
    ) -> dict[str, Any]:
        """Return a minimal Schema-valid configuration without runtime imports."""
        config: dict[str, Any] = {
            "experiment": {"name": "migration_demo", "seeds": [3]},
            "data": {},
            "model": {"target": "unimportable_lambdaforge_test_target.Model"},
            "losses": [{"target": "unimportable_lambdaforge_test_target.Loss"}],
            "execution": {"mode": "sequential"},
        }
        if versioned:
            return {"schema_version": schema_version, **config}
        return config

    @staticmethod
    def legacy_yaml() -> str:
        """Return historical YAML containing presentation details to preserve."""
        return """# leading experiment comment
experiment: &identity
  name: "migration_demo"  # quoted identity
  seeds: [3]
data: {}
model:
  target: "unimportable_lambdaforge_test_target.Model"
losses:
  - target: 'unimportable_lambdaforge_test_target.Loss'
metadata:
  experiment_alias: *identity
execution:
  mode: sequential
"""

    @classmethod
    def current_yaml(cls) -> str:
        """Return the same fixture with an explicit current version."""
        return cls.legacy_yaml().replace(
            "# leading experiment comment\n",
            '# leading experiment comment\nschema_version: "1.1"\n',
            1,
        )

    def test_schema_version_is_strict_exact_and_orderable(self) -> None:
        current = ExperimentSchemaVersion.current()
        legacy = ExperimentSchemaVersion.unversioned()

        assert current.value == "1.1"
        assert str(current) == "1.1"
        assert current.to_json_value() == "1.1"
        assert legacy.is_unversioned
        assert legacy.to_json_value() is None
        assert legacy < ExperimentSchemaVersion("0.0") < current
        assert ExperimentSchemaVersion.from_value(None) == legacy
        assert ExperimentSchemaVersion.from_config({}) == legacy
        assert ExperimentSchemaVersion.from_config({"schema_version": "1.1"}) == current

    @pytest.mark.parametrize("value", [1.0, 1, True, [], {}])
    def test_schema_version_rejects_yaml_coercion(self, value: Any) -> None:
        with pytest.raises(TypeError, match="quoted.*MAJOR.MINOR"):
            ExperimentSchemaVersion.from_value(value)

    def test_explicit_null_version_is_not_treated_as_legacy_absence(self) -> None:
        with pytest.raises(TypeError, match="quoted.*MAJOR.MINOR"):
            ExperimentSchemaVersion.from_config({"schema_version": None})

    @pytest.mark.parametrize(
        "value",
        ["", "1", "v1.0", "01.0", "1.00", "1.0.0", " 1.0", "1.0 "],
    )
    def test_schema_version_rejects_informal_strings(self, value: str) -> None:
        with pytest.raises(ValueError, match="exact string form"):
            ExperimentSchemaVersion.from_value(value)

    def test_unversioned_mapping_migrates_without_mutating_any_input(self) -> None:
        source = self.valid_config(versioned=False)
        snapshot = copy.deepcopy(source)

        result = ExperimentConfigMigrator.default().preview_mapping(source)

        assert source == snapshot
        assert list(source) == list(snapshot)
        assert result.changed
        assert result.source is None
        assert result.source_version == ExperimentSchemaVersion.unversioned()
        assert result.target_version == ExperimentSchemaVersion.current()
        assert list(result.config)[0] == "schema_version"
        assert result.config["schema_version"] == "1.1"
        assert result.config["experiment"] == source["experiment"]
        assert [step.identifier for step in result.steps] == [
            "unversioned_to_1_0",
            "1_0_to_1_1",
        ]

        historical = ExperimentConfigMigrator.default().preview_mapping(
            source,
            target_version="1.0",
        )
        assert historical.config["schema_version"] == "1.0"
        assert [step.identifier for step in historical.steps] == ["unversioned_to_1_0"]

    def test_schema_1_0_migration_changes_only_the_version_declaration(self) -> None:
        source = self.valid_config(versioned=True, schema_version="1.0")
        snapshot = copy.deepcopy(source)

        result = ExperimentConfigMigrator.default().preview_mapping(source)

        assert source == snapshot
        assert result.source_version == ExperimentSchemaVersion("1.0")
        assert result.target_version == ExperimentSchemaVersion("1.1")
        assert [step.identifier for step in result.steps] == ["1_0_to_1_1"]
        assert result.config == {**source, "schema_version": "1.1"}

        direct = copy.deepcopy(source)
        migrated = ExperimentV1ToV1_1Migration().apply(direct)
        assert migrated == {**source, "schema_version": "1.1"}
        with pytest.raises(ValueError, match="requires schema_version '1.0'"):
            ExperimentV1ToV1_1Migration().apply(
                self.valid_config(versioned=True, schema_version="1.1")
            )

    def test_current_mapping_is_an_exact_no_op(self) -> None:
        source = self.valid_config(versioned=True)
        snapshot = copy.deepcopy(source)

        result = ExperimentConfigMigrator.default().preview_mapping(source)

        assert source == snapshot
        assert not result.changed
        assert result.steps == ()
        assert result.config == source
        assert result.diff() == ""
        assert result.migrated_yaml == result.original_yaml

    def test_round_trip_preview_preserves_comments_order_quotes_and_anchors(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "legacy.yaml"
        original = self.legacy_yaml()
        source.write_text(original, encoding="utf-8", newline="")

        result = ExperimentConfigMigrator.default().preview_file(source)
        migrated = result.migrated_yaml

        assert source.read_text(encoding="utf-8") == original
        assert "# leading experiment comment" in migrated
        assert 'name: "migration_demo"  # quoted identity' in migrated
        assert "target: 'unimportable_lambdaforge_test_target.Loss'" in migrated
        assert "&identity" in migrated
        assert "*identity" in migrated
        assert migrated.index("schema_version") < migrated.index("experiment:")
        assert migrated.index("experiment:") < migrated.index("data:")
        assert RoundTripYamlCodec().load_text(migrated)["schema_version"] == "1.1"

    def test_current_file_no_op_preserves_source_bytes_and_presentation(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "current.yaml"
        original = self.current_yaml()
        source.write_text(original, encoding="utf-8", newline="")

        result = ExperimentConfigMigrator.default().preview_file(source)

        assert not result.changed
        assert result.original_yaml == original
        assert result.migrated_yaml == original
        assert source.read_text(encoding="utf-8") == original

    def test_changed_file_preserves_crlf_newline_convention(self, tmp_path: Path) -> None:
        source = tmp_path / "legacy-crlf.yaml"
        original = self.legacy_yaml().replace("\n", "\r\n")
        source.write_bytes(original.encode("utf-8"))

        result = ExperimentConfigMigrator.default().preview_file(source)

        assert result.original_yaml == original
        assert "\r\n" in result.migrated_yaml
        assert "\n" not in result.migrated_yaml.replace("\r\n", "")
        assert source.read_bytes() == original.encode("utf-8")

    def test_round_trip_loader_rejects_invalid_document_shapes(self) -> None:
        codec = RoundTripYamlCodec()

        for text in ("", "null\n", "- one\n- two\n"):
            with pytest.raises(TypeError, match="root must be a mapping"):
                codec.load_text(text)
        with pytest.raises(DuplicateKeyError):
            codec.load_text("experiment: {}\nexperiment: {}\n")
        with pytest.raises(ComposerError):
            codec.load_text("experiment: {}\n---\nexperiment: {}\n")

    def test_registry_resolves_a_deterministic_synthetic_multi_step_path(self) -> None:
        first = self._SyntheticMigration("one_to_one_one", "1.0", "1.1")
        second = self._SyntheticMigration("one_one_to_two", "1.1", "2.0")
        registry = ExperimentConfigMigrationRegistry((second, first))

        assert registry.migrations == (first, second)
        assert registry.path(ExperimentSchemaVersion("1.0"), ExperimentSchemaVersion("2.0")) == (
            first,
            second,
        )
        assert registry.path(ExperimentSchemaVersion("1.1"), ExperimentSchemaVersion("1.1")) == ()

        extended = ExperimentConfigMigrationRegistry((first,)).with_migration(second)
        assert extended.path(ExperimentSchemaVersion("1.0"), ExperimentSchemaVersion("2.0")) == (
            first,
            second,
        )

    def test_registry_rejects_duplicates_ambiguity_and_non_forward_steps(self) -> None:
        first = self._SyntheticMigration("advance", "1.0", "1.1")
        duplicate_id = self._SyntheticMigration("advance", "1.1", "1.2")
        ambiguous = self._SyntheticMigration("other", "1.0", "1.2")
        stationary = self._SyntheticMigration("stationary", "1.0", "1.0")
        backward = self._SyntheticMigration("backward", "1.1", "1.0")

        with pytest.raises(ValueError, match="Duplicate migration identifier"):
            ExperimentConfigMigrationRegistry((first, duplicate_id))
        with pytest.raises(ValueError, match="Ambiguous migrations"):
            ExperimentConfigMigrationRegistry((first, ambiguous))
        with pytest.raises(ValueError, match="strictly forward"):
            ExperimentConfigMigrationRegistry((stationary,))
        with pytest.raises(ValueError, match="strictly forward"):
            ExperimentConfigMigrationRegistry((backward,))

    def test_registry_rejects_gaps_overshoots_and_downgrades(self) -> None:
        registry = ExperimentConfigMigrationRegistry(
            (self._SyntheticMigration("one_to_two", "1.0", "2.0"),)
        )

        with pytest.raises(ValueError, match="No migration path"):
            registry.path(ExperimentSchemaVersion("1.0"), ExperimentSchemaVersion("1.1"))
        with pytest.raises(ValueError, match="No migration path"):
            registry.path(ExperimentSchemaVersion("1.1"), ExperimentSchemaVersion("2.0"))
        with pytest.raises(ValueError, match="downgrade is not supported"):
            registry.path(ExperimentSchemaVersion("2.0"), ExperimentSchemaVersion("1.0"))

    def test_schema_catalog_is_version_aligned_valid_and_defensive(self) -> None:
        catalog = ExperimentSchemaCatalog()
        schema = catalog.schema()
        historical = catalog.schema("1.0")
        historical_resource = (
            files("lambdaforge").joinpath("schemas").joinpath("experiment-1.0.schema.json")
        )

        assert catalog.current_version == ExperimentSchemaVersion("1.1")
        assert catalog.supported_versions == (
            ExperimentSchemaVersion("1.0"),
            ExperimentSchemaVersion("1.1"),
        )
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["$id"].endswith("experiment-1.1.json")
        assert schema["properties"]["schema_version"]["const"] == "1.1"
        assert historical["$id"].endswith("experiment-1.0.json")
        assert historical["properties"]["schema_version"]["const"] == "1.0"
        assert "retention" not in historical["properties"]
        assert hashlib.sha256(historical_resource.read_bytes()).hexdigest() == (
            "273f435eb2fe02d6489b757528c53a92719473463d6f1470f2ebb33189605444"
        )
        assert "retention" in schema["properties"]
        assert "schema_version" in schema["required"]
        assert catalog.validation_errors(self.valid_config(versioned=True)) == ()
        assert (
            catalog.validation_errors(
                self.valid_config(versioned=True, schema_version="1.0"),
                "1.0",
            )
            == ()
        )
        cached_validator = catalog._validator_cache[ExperimentSchemaVersion.current()]
        assert catalog.validation_errors(self.valid_config(versioned=True)) == ()
        assert catalog._validator_cache[ExperimentSchemaVersion.current()] is cached_validator
        assert len(catalog._schema_cache) == 2
        assert len(catalog._validator_cache) == 2
        assert any(
            "schema_version" in error
            for error in catalog.validation_errors(self.valid_config(versioned=False))
        )

        schema["properties"]["schema_version"]["const"] = "corrupted"
        assert catalog.schema()["properties"]["schema_version"]["const"] == "1.1"
        with pytest.raises(ValueError, match="must contain the current version"):
            ExperimentSchemaCatalog({})

    def test_schema_1_1_accepts_the_complete_retention_contract(self) -> None:
        config = self.valid_config(versioned=True)
        config["retention"] = {
            "mode": "apply",
            "checkpoints": {
                "keep": "last_and_best",
                "prune_unselected": True,
            },
            "protect": ["reports/**", "predictions/final.json"],
            "rules": [
                {
                    "action": "compress",
                    "include": ["predictions/**/*.json"],
                    "exclude": ["predictions/final.json"],
                    "min_size_bytes": 1024,
                    "compression": {
                        "level": 9,
                        "only_if_smaller": True,
                    },
                },
                {
                    "action": "prune",
                    "include": ["temporary/**"],
                    "exclude": [],
                    "min_size_bytes": 0,
                },
            ],
            "archive": {
                "name": "retained_artifacts.zip",
                "compression_level": 6,
            },
            "lock_timeout_seconds": 30.5,
        }

        catalog = ExperimentSchemaCatalog()

        assert catalog.validation_errors(config) == ()
        retention = catalog.schema()["$defs"]["retention"]
        assert retention["additionalProperties"] is False
        assert retention["properties"]["mode"]["default"] == "disabled"
        assert retention["properties"]["checkpoints"]["default"] == {}
        assert retention["properties"]["protect"]["default"] == []
        assert retention["properties"]["rules"]["default"] == []
        assert retention["properties"]["archive"]["default"] == {}
        assert retention["properties"]["lock_timeout_seconds"]["default"] == 60
        checkpoints = catalog.schema()["$defs"]["retentionCheckpoints"]["properties"]
        assert checkpoints["keep"]["default"] == "all"
        assert checkpoints["prune_unselected"]["default"] is False

    @pytest.mark.parametrize(
        ("retention", "error_fragment"),
        [
            ({"mode": "delete"}, "mode"),
            ({"unknown": True}, "Additional properties"),
            ({"checkpoints": {"keep": "none"}}, "checkpoints"),
            ({"checkpoints": {"prune_unselected": 1}}, "checkpoints"),
            (
                {"rules": [{"action": "compress", "include": []}]},
                "rules",
            ),
            (
                {"rules": [{"action": "prune", "include": [""]}]},
                "rules",
            ),
            (
                {
                    "rules": [
                        {
                            "action": "prune",
                            "include": ["temporary/**"],
                            "compression": {"level": 1},
                        }
                    ]
                },
                "rules",
            ),
            (
                {
                    "rules": [
                        {
                            "action": "compress",
                            "include": ["data/**"],
                            "min_size_bytes": -1,
                        }
                    ]
                },
                "rules",
            ),
            (
                {
                    "rules": [
                        {
                            "action": "compress",
                            "include": ["data/**"],
                            "compression": {"level": 10},
                        }
                    ]
                },
                "rules",
            ),
            ({"archive": {"name": "../escape.zip"}}, "archive"),
            ({"archive": {"name": "artifacts.tar"}}, "archive"),
            ({"archive": {"compression_level": -1}}, "archive"),
            ({"lock_timeout_seconds": 0}, "lock_timeout_seconds"),
        ],
    )
    def test_schema_1_1_rejects_unsafe_or_malformed_retention(
        self,
        retention: dict[str, Any],
        error_fragment: str,
    ) -> None:
        config = self.valid_config(versioned=True)
        config["retention"] = retention

        errors = ExperimentSchemaCatalog().validation_errors(config)

        assert errors
        assert any(error_fragment in error for error in errors)

    def test_migrator_validates_each_result_and_rejects_unsupported_targets(self) -> None:
        invalid = {"experiment": {"name": "incomplete"}}

        with pytest.raises(ValueError, match="produced invalid Schema 1.0"):
            ExperimentConfigMigrator.default().preview_mapping(invalid)
        unchecked = ExperimentConfigMigrator.default().preview_mapping(invalid, validate=False)
        assert unchecked.config["schema_version"] == "1.1"
        with pytest.raises(ValueError, match="No packaged JSON Schema"):
            ExperimentConfigMigrator.default().preview_mapping(
                self.valid_config(versioned=False), target_version="2.0"
            )
        with pytest.raises(ValueError, match="explicit Schema version"):
            ExperimentConfigMigrator.default().preview_mapping(
                self.valid_config(versioned=False),
                target_version=ExperimentSchemaVersion.unversioned(),
            )

    def test_result_diff_json_and_defensive_payload_are_deterministic(self) -> None:
        migrator = ExperimentConfigMigrator.default()
        first = migrator.preview_mapping(self.valid_config(versioned=False))
        second = migrator.preview_mapping(self.valid_config(versioned=False))

        assert first.diff() == second.diff()
        assert first.to_dict() == second.to_dict()
        assert first.render(MigrationPreviewFormat.DIFF) == first.diff()
        assert first.render("yaml") == first.migrated_yaml
        payload = json.loads(first.render(MigrationPreviewFormat.JSON))
        assert payload == first.to_dict()
        assert payload["migration_result_version"] == 1
        assert payload["source_version"] is None
        assert payload["target_version"] == "1.1"
        assert payload["changed"] is True
        assert payload["steps"][0]["id"] == "unversioned_to_1_0"
        assert payload["steps"][1]["id"] == "1_0_to_1_1"
        assert "--- configuration.yaml" in payload["diff"]
        assert "+++ configuration.yaml [migrated to 1.1]" in payload["diff"]

        payload["config"]["experiment"]["name"] = "mutated"
        assert first.to_dict()["config"]["experiment"]["name"] == "migration_demo"
        semantic = first.config
        semantic["experiment"]["name"] = "mutated"
        assert first.config["experiment"]["name"] == "migration_demo"
        with pytest.raises(TypeError, match="FrozenJsonMapping is immutable"):
            first["config"]["experiment"]["name"] = "mutated"
        with pytest.raises(TypeError, match="FrozenJsonList is immutable"):
            first["steps"].append({})
        assert json.loads(json.dumps(first)) == first.to_dict()
        with pytest.raises(TypeError, match="does not support item assignment"):
            first["changed"] = False

    def test_mapping_boundary_preserves_non_yaml_python_value_types(self) -> None:
        source = self.valid_config(versioned=True)
        source["metadata"] = {
            "tuple": (1, 2),
            "path": Path("relative/data"),
            "opaque": object(),
        }

        config = ExperimentConfig(source)
        result = ExperimentConfigMigrator.default().preview_mapping(source)

        assert config["metadata"]["tuple"] == (1, 2)
        assert isinstance(config["metadata"]["tuple"], tuple)
        assert config["metadata"]["path"] == Path("relative/data")
        assert type(config["metadata"]["opaque"]) is object
        assert result.config["metadata"]["tuple"] == (1, 2)
        assert json.loads(result.render("json"))["config"]["metadata"] == {
            "tuple": [1, 2],
            "path": os.fspath(Path("relative/data")),
            "opaque": "<builtins.object>",
        }

    def test_json_preview_projects_dates_without_changing_semantic_config(self) -> None:
        source = self.valid_config(versioned=True)
        source["metadata"] = {"release_date": date(2026, 7, 17)}

        result = ExperimentConfigMigrator.default().preview_mapping(source)

        assert result.config["metadata"]["release_date"] == date(2026, 7, 17)
        assert json.loads(result.render("json"))["config"]["metadata"]["release_date"] == (
            "2026-07-17"
        )

    def test_result_writes_atomically_only_to_a_distinct_explicit_path(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "legacy.yaml"
        source.write_text(self.legacy_yaml(), encoding="utf-8", newline="")
        original = source.read_text(encoding="utf-8")
        result = ExperimentConfigMigrator.default().preview_file(source)
        destination = tmp_path / "nested" / "migrated.yaml"

        assert result.write_yaml(destination) == destination
        assert destination.read_text(encoding="utf-8") == result.migrated_yaml
        assert source.read_text(encoding="utf-8") == original
        assert not list(destination.parent.glob("*.tmp"))
        assert not list(destination.parent.glob(".*.tmp"))

        with pytest.raises(FileExistsError, match="already exists"):
            result.write_yaml(destination)
        destination.write_text("stale\n", encoding="utf-8")
        result.write_yaml(destination, overwrite=True)
        assert destination.read_text(encoding="utf-8") == result.migrated_yaml
        with pytest.raises(ValueError, match="must not overwrite the source"):
            result.write_yaml(source, overwrite=True)
        assert source.read_text(encoding="utf-8") == original

    def test_concurrent_non_overwrite_publication_has_exactly_one_winner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = ExperimentConfigMigrator.default().preview_mapping(
            self.valid_config(versioned=False)
        )
        destination = tmp_path / "race.yaml"
        barrier = threading.Barrier(2)
        original_link = os.link

        def synchronized_link(source: str | bytes, target: str | bytes) -> None:
            barrier.wait(timeout=5)
            original_link(source, target)

        monkeypatch.setattr(os, "link", synchronized_link)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(result.write_yaml, destination) for _ in range(2)]

        successes = 0
        failures = 0
        for future in futures:
            try:
                assert future.result() == destination
                successes += 1
            except FileExistsError:
                failures += 1
        assert (successes, failures) == (1, 1)
        assert destination.read_text(encoding="utf-8") == result.migrated_yaml
        assert not list(tmp_path.glob(".*.tmp"))

    def test_cli_default_preview_is_diff_only_and_has_no_side_effects(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "legacy.yaml"
        original = self.legacy_yaml()
        source.write_text(original, encoding="utf-8", newline="")
        before = set(tmp_path.iterdir())
        assert "unimportable_lambdaforge_test_target" not in sys.modules

        exit_code = CommandLineInterface.main(["migrate", str(source)])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.err == ""
        assert captured.out.startswith("--- ")
        assert "+schema_version:" in captured.out
        assert source.read_text(encoding="utf-8") == original
        assert set(tmp_path.iterdir()) == before
        assert "unimportable_lambdaforge_test_target" not in sys.modules

    def test_cli_json_is_a_pure_machine_readable_preview(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "legacy.yaml"
        source.write_text(self.legacy_yaml(), encoding="utf-8", newline="")

        exit_code = CommandLineInterface.main(["migrate", str(source), "--format", "json"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert exit_code == 0
        assert captured.err == ""
        assert payload["changed"] is True
        assert payload["source"] == str(source)
        assert payload["config"]["schema_version"] == "1.1"

    def test_cli_output_is_explicit_atomic_and_never_changes_the_source(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "legacy.yaml"
        destination = tmp_path / "migrated.yaml"
        original = self.legacy_yaml()
        source.write_text(original, encoding="utf-8", newline="")

        assert (
            CommandLineInterface.main(
                ["migrate", str(source), "--format", "yaml", "--output", str(destination)]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert captured.out == destination.read_text(encoding="utf-8")
        assert f"Wrote migrated YAML: {destination}" in captured.err
        assert source.read_text(encoding="utf-8") == original
        assert yaml.safe_load(destination.read_text(encoding="utf-8"))["schema_version"] == "1.1"
        assert not list(tmp_path.glob("*.tmp"))

        destination.write_text("occupied\n", encoding="utf-8")
        assert (
            CommandLineInterface.main(
                ["migrate", str(source), "--output", str(destination), "--force"]
            )
            == 0
        )
        capsys.readouterr()
        assert "schema_version" in destination.read_text(encoding="utf-8")

    def test_cli_check_reports_migration_need_without_writing(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        legacy = tmp_path / "legacy.yaml"
        current = tmp_path / "current.yaml"
        legacy.write_text(self.legacy_yaml(), encoding="utf-8", newline="")
        current.write_text(self.current_yaml(), encoding="utf-8", newline="")
        before = {path: path.read_bytes() for path in tmp_path.iterdir()}

        assert CommandLineInterface.main(["migrate", str(legacy), "--check"]) == 1
        legacy_output = capsys.readouterr()
        assert "+schema_version:" in legacy_output.out
        assert legacy_output.err == ""
        assert CommandLineInterface.main(["migrate", str(current), "--check"]) == 0
        current_output = capsys.readouterr()
        assert "no migration required" in current_output.out
        assert current_output.err == ""
        assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before

    def test_cli_reports_option_target_and_write_errors_on_stderr(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "legacy.yaml"
        source.write_text(self.legacy_yaml(), encoding="utf-8", newline="")
        destination = tmp_path / "existing.yaml"
        destination.write_text("occupied\n", encoding="utf-8")

        assert CommandLineInterface.main(["migrate", str(source), "--force"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "--force requires --output" in captured.err

        assert (
            CommandLineInterface.main(
                ["migrate", str(source), "--check", "--output", str(destination)]
            )
            == 1
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "--check cannot be combined with --output" in captured.err

        assert CommandLineInterface.main(["migrate", str(source), "--target-version", "2.0"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "No packaged JSON Schema" in captured.err

        assert (
            CommandLineInterface.main(["migrate", str(source), "--output", str(destination)]) == 1
        )
        captured = capsys.readouterr()
        assert captured.out.startswith("--- ")
        assert "already exists" in captured.err

        original = source.read_text(encoding="utf-8")
        assert (
            CommandLineInterface.main(["migrate", str(source), "--output", str(source), "--force"])
            == 1
        )
        captured = capsys.readouterr()
        assert captured.out.startswith("--- ")
        assert "must not overwrite the source" in captured.err
        assert source.read_text(encoding="utf-8") == original

    def test_cli_rejects_an_unquoted_numeric_schema_version(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        source = tmp_path / "numeric.yaml"
        source.write_text(
            self.current_yaml().replace('schema_version: "1.1"', "schema_version: 1.1"),
            encoding="utf-8",
        )

        assert CommandLineInterface.main(["migrate", str(source)]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "schema_version must be a quoted 'MAJOR.MINOR' string" in captured.err

    def test_experiment_config_normalizes_legacy_and_rejects_unknown_versions(self) -> None:
        source = self.valid_config(versioned=False)
        snapshot = copy.deepcopy(source)

        config = ExperimentConfig(source)

        assert source == snapshot
        assert config["schema_version"] == "1.1"
        assert config.migration_result.changed
        assert config.migration_result.source_version.is_unversioned
        assert ExperimentConfig(self.valid_config(versioned=True)).migration_result.changed is False

        for version in ("0.9", "2.0"):
            unsupported = self.valid_config(versioned=False)
            unsupported["schema_version"] = version
            with pytest.raises(ValueError, match="Cannot normalize.*lambdaforge migrate"):
                ExperimentConfig(unsupported)

        config.set("schema_version", "2.0")
        with pytest.raises(ValueError, match="requires current Schema 1.1"):
            config.expand()

    def test_experiment_config_from_yaml_retains_migration_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "legacy.yaml"
        original = self.legacy_yaml()
        source.write_text(original, encoding="utf-8", newline="")

        config = ExperimentConfig.from_yaml(source)

        assert config.source == source
        assert config["schema_version"] == "1.1"
        assert type(config["schema_version"]) is str
        assert type(config["experiment"]) is dict
        assert type(config["experiment"]["seeds"]) is list
        assert config.migration_result.source == str(source)
        assert config.migration_result.changed
        assert yaml.safe_load(yaml.safe_dump(config.as_dict()))["schema_version"] == "1.1"
        assert source.read_text(encoding="utf-8") == original

    def test_validator_reports_migration_provenance_for_legacy_and_current_configs(
        self,
    ) -> None:
        validator = ExperimentValidator()

        legacy = validator.validate(self.valid_config(versioned=False), check_imports=False)
        assert legacy.is_valid
        assert legacy.expanded_runs == 1
        assert legacy.source_schema_version is None
        assert legacy.target_schema_version == "1.1"
        assert [step["id"] for step in legacy.migration_steps] == [
            "unversioned_to_1_0",
            "1_0_to_1_1",
        ]
        assert any("normalized from unversioned to 1.1" in warning for warning in legacy.warnings)
        assert legacy.to_dict()["migration"] == {
            "source_version": None,
            "target_version": "1.1",
            "changed": True,
            "steps": [dict(step) for step in legacy.migration_steps],
        }
        assert "migrated to Schema 1.1" in legacy.summary()

        current = validator.validate(self.valid_config(versioned=True), check_imports=False)
        assert current.is_valid
        assert current.source_schema_version == "1.1"
        assert current.target_schema_version == "1.1"
        assert current.migration_steps == ()
        assert not any("normalized from" in warning for warning in current.warnings)

    def test_validator_file_and_run_loader_normalize_without_rewriting_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        config_path = run_dir / "config.yaml"
        original = yaml.safe_dump(self.valid_config(versioned=False), sort_keys=False)
        config_path.write_text(original, encoding="utf-8")

        report = ExperimentValidator().validate_file(config_path, check_imports=False)
        loaded = RunLoader.load_config(run_dir)

        assert report.is_valid
        assert report.source == str(config_path)
        assert report.source_schema_version is None
        assert report.target_schema_version == "1.1"
        assert loaded["schema_version"] == "1.1"
        assert config_path.read_text(encoding="utf-8") == original

    def test_public_facade_and_lazy_namespaces_export_migration_objects(
        self,
        tmp_path: Path,
    ) -> None:
        import lambdaforge.experiments as experiments_api
        import lambdaforge.experiments.migrations as migrations_api

        expected_experiment_exports = {
            "ExperimentConfigMigration": ExperimentConfigMigration,
            "ExperimentConfigMigrationRegistry": ExperimentConfigMigrationRegistry,
            "ExperimentConfigMigrator": ExperimentConfigMigrator,
            "ExperimentSchemaCatalog": ExperimentSchemaCatalog,
            "ExperimentSchemaVersion": ExperimentSchemaVersion,
            "MigrationPreviewFormat": MigrationPreviewFormat,
        }
        for name, expected in expected_experiment_exports.items():
            assert name in experiments_api.__all__
            assert getattr(experiments_api, name) is expected

        for name in migrations_api.__all__:
            assert isinstance(getattr(migrations_api, name), type)

        source = tmp_path / "legacy.yaml"
        source.write_text(self.legacy_yaml(), encoding="utf-8", newline="")
        result = LambdaForge.preview_migration(source)
        assert result.changed
        assert result.config["schema_version"] == "1.1"

    def test_root_import_does_not_eagerly_load_configuration_dependencies(self) -> None:
        script = (
            "import sys; import lambdaforge; "
            "assert 'ruamel.yaml' not in sys.modules; "
            "assert 'jsonschema' not in sys.modules; "
            "assert lambdaforge.__version__ == '0.7.0'"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr

    def test_facade_validation_returns_a_report_for_unsupported_versions(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "future.yaml"
        config = self.valid_config(versioned=True)
        config["schema_version"] = "2.0"
        source.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        report = LambdaForge.validate(source, check_imports=False)

        assert not report.is_valid
        assert any("Schema downgrade is not supported" in error for error in report.errors)
