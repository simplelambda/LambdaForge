"""Atomic commit receipt for a complete experiment aggregation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.FrozenJsonMapping import FrozenJsonMapping
from lambdaforge.experiments.JsonResult import JsonResult
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.experiments.RunStatus import RunStatus


class AggregationReceipt(JsonResult):
    """Prove that all expected runs and aggregate outputs were complete together."""

    VERSION = 1
    FILE_NAME = "aggregation_receipt.json"
    _RUN_INPUTS = (
        "config.yaml",
        "environment.json",
        "hparams.json",
        "metrics.csv",
        "result.json",
    )
    _AGGREGATE_OUTPUTS = (
        "summary.json",
        "run_status.csv",
        "seed_metrics.csv",
        "summary.csv",
        "summary_wide.csv",
    )

    def __init__(
        self,
        *,
        receipt_id: str,
        base_dir: str | Path,
        complete: bool,
        expected_runs: int,
        completed_runs: int,
        run_dirs: Sequence[str],
        required_artifacts: Mapping[str, Sequence[str]],
        input_fingerprints: Mapping[str, Mapping[str, Any]],
        output_fingerprints: Mapping[str, Mapping[str, Any]],
        config_fingerprint: str,
        variants: Mapping[str, Mapping[str, Any]],
        reasons: Sequence[str] = (),
        created_at: str | None = None,
    ) -> None:
        self.receipt_id = str(receipt_id)
        self.base_dir = str(base_dir)
        self.complete = bool(complete)
        self.expected_runs = int(expected_runs)
        self.completed_runs = int(completed_runs)
        self.run_dirs = tuple(str(item) for item in run_dirs)
        self.required_artifacts = {
            str(run): tuple(str(item) for item in paths)
            for run, paths in required_artifacts.items()
        }
        self.input_fingerprints = copy.deepcopy(dict(input_fingerprints))
        self.output_fingerprints = copy.deepcopy(dict(output_fingerprints))
        self.config_fingerprint = str(config_fingerprint)
        self.variants = copy.deepcopy(dict(variants))
        self.reasons = tuple(str(item) for item in reasons)
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        payload = FrozenJsonMapping(
            {
                "aggregation_receipt_version": self.VERSION,
                "receipt_id": self.receipt_id,
                "base_dir": self.base_dir,
                "complete": self.complete,
                "expected_runs": self.expected_runs,
                "completed_runs": self.completed_runs,
                "run_dirs": list(self.run_dirs),
                "required_artifacts": {
                    run: list(paths) for run, paths in self.required_artifacts.items()
                },
                "input_fingerprints": self.input_fingerprints,
                "output_fingerprints": self.output_fingerprints,
                "config_fingerprint": self.config_fingerprint,
                "variants": self.variants,
                "reasons": list(self.reasons),
                "created_at": self.created_at,
            }
        )
        self._freeze_mapping(dict(payload))

    @classmethod
    def build(cls, config: ExperimentConfig | Mapping[str, Any]) -> AggregationReceipt:
        """Inspect a just-published aggregate and produce its commit receipt."""
        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        base_dir = normalized.suite_dir
        aggregate_dir = base_dir / "aggregate"
        reasons: list[str] = []
        input_fingerprints: dict[str, dict[str, Any]] = {}
        output_fingerprints: dict[str, dict[str, Any]] = {}
        required: dict[str, tuple[str, ...]] = {}
        run_dirs: list[str] = []
        completed_runs = 0
        expanded = normalized.expand()

        expected_variants: dict[str, int] = {}
        for run_config in expanded:
            variant = str(ExperimentConfig.get_value(run_config, "experiment.variant", "base"))
            expected_variants[variant] = expected_variants.get(variant, 0) + 1
            seed = ExperimentConfig.get_value(run_config, "experiment.seed")
            run_dir = base_dir / variant / (f"seed={seed}" if seed is not None else "seed=none")
            run_relative = run_dir.relative_to(base_dir).as_posix()
            run_dirs.append(run_relative)
            result_path = run_dir / "result.json"
            try:
                result = RunResult.read_json(result_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                reasons.append(f"{run_relative}: unreadable result.json ({type(error).__name__}).")
                result = None
            if result is not None and result.status is RunStatus.OK:
                completed_runs += 1
            else:
                reasons.append(f"{run_relative}: run status is not ok.")

            for name in cls._RUN_INPUTS:
                path = run_dir / name
                if (
                    name == "metrics.csv"
                    and not bool(
                        ExperimentConfig.get_value(
                            run_config,
                            "trainer.write_epoch_metrics_csv",
                            True,
                        )
                    )
                    and not path.exists()
                ):
                    continue
                relative = path.relative_to(base_dir).as_posix()
                fingerprint = cls._fingerprint_regular(path)
                if fingerprint is None:
                    reasons.append(f"{run_relative}: missing or unsafe {name}.")
                else:
                    input_fingerprints[relative] = fingerprint

            raw_required = ExperimentConfig.get_value(
                run_config,
                "experiment.required_artifacts",
                [],
            )
            if not isinstance(raw_required, list):
                reasons.append(f"{run_relative}: required_artifacts is not a list.")
                raw_required = []
            safe_required: list[str] = []
            for value in raw_required:
                try:
                    relative_required = cls._validate_required_path(str(value))
                    required_path = run_dir.joinpath(*PurePosixPath(relative_required).parts)
                    if not cls._safe_existing(run_dir, required_path):
                        reasons.append(
                            f"{run_relative}: missing or unsafe required artifact "
                            f"{relative_required}."
                        )
                        continue
                    safe_required.append(relative_required)
                except (OSError, ValueError) as error:
                    reasons.append(f"{run_relative}: invalid required artifact ({error}).")
            required[run_relative] = tuple(sorted(safe_required))

        for name in cls._AGGREGATE_OUTPUTS:
            path = aggregate_dir / name
            relative = path.relative_to(base_dir).as_posix()
            fingerprint = cls._fingerprint_regular(path)
            if fingerprint is None:
                reasons.append(f"aggregate: missing or unsafe {name}.")
            else:
                output_fingerprints[relative] = fingerprint

        legacy_summary = base_dir / "summary.csv"
        legacy_fingerprint = cls._fingerprint_regular(legacy_summary)
        if legacy_fingerprint is None:
            reasons.append("aggregate: missing or unsafe legacy summary.csv.")
        else:
            output_fingerprints["summary.csv"] = legacy_fingerprint

        aggregate_files, unsafe_aggregate_paths = cls._regular_tree_files(
            aggregate_dir,
            excluded_roots={
                cls.FILE_NAME,
                "retention",
                ".retention-quarantine",
                ".retention-transaction.json",
            },
        )
        for path in aggregate_files:
            relative = path.relative_to(base_dir).as_posix()
            fingerprint = cls._fingerprint_regular(path)
            if fingerprint is None:
                reasons.append(f"aggregate: output changed while hashing {relative}.")
            else:
                output_fingerprints[relative] = fingerprint
        for path in unsafe_aggregate_paths:
            reasons.append(
                f"aggregate: unsafe committed output {path.relative_to(base_dir).as_posix()}."
            )

        for variant in sorted(expected_variants):
            variant_dir = base_dir / variant
            variant_aggregate = variant_dir / "aggregate.json"
            relative_aggregate = variant_aggregate.relative_to(base_dir).as_posix()
            variant_fingerprint = cls._fingerprint_regular(variant_aggregate)
            if variant_fingerprint is None:
                reasons.append(f"aggregate: missing or unsafe {relative_aggregate}.")
            else:
                output_fingerprints[relative_aggregate] = variant_fingerprint

            for name in ("epoch_metrics.csv", "epoch_metrics_wide.csv"):
                path = variant_dir / name
                if not path.exists():
                    continue
                relative = path.relative_to(base_dir).as_posix()
                fingerprint = cls._fingerprint_regular(path)
                if fingerprint is None:
                    reasons.append(f"aggregate: unsafe variant output {relative}.")
                else:
                    output_fingerprints[relative] = fingerprint

            plot_files, unsafe_plot_paths = cls._regular_tree_files(variant_dir / "plots")
            for path in plot_files:
                relative = path.relative_to(base_dir).as_posix()
                fingerprint = cls._fingerprint_regular(path)
                if fingerprint is None:
                    reasons.append(f"aggregate: plot changed while hashing {relative}.")
                else:
                    output_fingerprints[relative] = fingerprint
            for path in unsafe_plot_paths:
                reasons.append(
                    f"aggregate: unsafe variant plot {path.relative_to(base_dir).as_posix()}."
                )

        summary_path = aggregate_dir / "summary.json"
        summary: Mapping[str, Any] = {}
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, Mapping):
                raise TypeError("summary must be an object")
            summary = loaded
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            reasons.append(f"aggregate: unreadable summary.json ({type(error).__name__}).")

        raw_variants = summary.get("variants", {})
        variants: dict[str, dict[str, Any]] = {}
        if not isinstance(raw_variants, Mapping):
            reasons.append("aggregate: summary variants is not a mapping.")
        else:
            for name, expected_count in sorted(expected_variants.items()):
                value = raw_variants.get(name)
                if not isinstance(value, Mapping):
                    reasons.append(f"aggregate: missing variant {name!r}.")
                    continue
                projection = {
                    "complete": bool(value.get("complete", False)),
                    "terminal": bool(value.get("terminal", False)),
                    "expected_runs": int(value.get("expected_n", 0)),
                    "completed_runs": int(value.get("n_seeds", 0)),
                }
                variants[name] = projection
                if not (
                    projection["complete"]
                    and projection["terminal"]
                    and projection["expected_runs"] == expected_count
                    and projection["completed_runs"] == expected_count
                ):
                    reasons.append(f"aggregate: variant {name!r} is not complete.")
            extra_variants = sorted(set(raw_variants) - set(expected_variants))
            if extra_variants:
                reasons.append(f"aggregate: unexpected variants {extra_variants}.")

        config_fingerprint = cls._fingerprint_mapping(normalized.as_dict())
        identity_payload = {
            "version": cls.VERSION,
            "base_dir": str(base_dir),
            "expected_runs": len(expanded),
            "completed_runs": completed_runs,
            "run_dirs": sorted(run_dirs),
            "required_artifacts": required,
            "input_fingerprints": input_fingerprints,
            "output_fingerprints": output_fingerprints,
            "config_fingerprint": config_fingerprint,
            "variants": variants,
            "reasons": sorted(set(reasons)),
        }
        receipt_id = cls._fingerprint_mapping(identity_payload)
        return cls(
            receipt_id=receipt_id,
            base_dir=base_dir,
            complete=not reasons and bool(expanded) and completed_runs == len(expanded),
            expected_runs=len(expanded),
            completed_runs=completed_runs,
            run_dirs=tuple(sorted(run_dirs)),
            required_artifacts=required,
            input_fingerprints=input_fingerprints,
            output_fingerprints=output_fingerprints,
            config_fingerprint=config_fingerprint,
            variants=variants,
            reasons=tuple(sorted(set(reasons))),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> AggregationReceipt:
        """Load and validate a persisted receipt."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("Aggregation receipt JSON must contain an object.")
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AggregationReceipt:
        """Parse an aggregation receipt without trusting its completeness."""
        required = value.get("required_artifacts", {})
        inputs = value.get("input_fingerprints", {})
        outputs = value.get("output_fingerprints", {})
        variants = value.get("variants", {})
        if not all(isinstance(item, Mapping) for item in (required, inputs, outputs, variants)):
            raise TypeError("Aggregation receipt mappings are malformed.")
        return cls(
            receipt_id=str(value["receipt_id"]),
            base_dir=str(value["base_dir"]),
            complete=bool(value.get("complete", False)),
            expected_runs=int(value.get("expected_runs", 0)),
            completed_runs=int(value.get("completed_runs", 0)),
            run_dirs=tuple(str(item) for item in value.get("run_dirs", ())),
            required_artifacts={
                str(run): tuple(str(path) for path in paths)
                for run, paths in required.items()
                if isinstance(paths, Sequence)
            },
            input_fingerprints={
                str(path): dict(fingerprint)
                for path, fingerprint in inputs.items()
                if isinstance(fingerprint, Mapping)
            },
            output_fingerprints={
                str(path): dict(fingerprint)
                for path, fingerprint in outputs.items()
                if isinstance(fingerprint, Mapping)
            },
            config_fingerprint=str(value.get("config_fingerprint", "")),
            variants={
                str(name): dict(payload)
                for name, payload in variants.items()
                if isinstance(payload, Mapping)
            },
            reasons=tuple(str(item) for item in value.get("reasons", ())),
            created_at=str(value.get("created_at", "")) or None,
        )

    @classmethod
    def path_for(cls, config: ExperimentConfig | Mapping[str, Any]) -> Path:
        """Return the canonical receipt path for one suite."""
        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        return normalized.suite_dir / "aggregate" / cls.FILE_NAME

    def is_current(self, config: ExperimentConfig | Mapping[str, Any]) -> bool:
        """Re-hash every committed input/output and verify required artifacts."""
        normalized = config if isinstance(config, ExperimentConfig) else ExperimentConfig(config)
        base_dir = normalized.suite_dir
        if str(base_dir) != self.base_dir:
            return False
        if self.config_fingerprint != self._fingerprint_mapping(normalized.as_dict()):
            return False
        for relative, expected in {
            **self.input_fingerprints,
            **self.output_fingerprints,
        }.items():
            try:
                path = base_dir.joinpath(*PurePosixPath(relative).parts)
                actual = self._fingerprint_regular(path)
            except (OSError, ValueError):
                return False
            if actual != expected:
                return False
        for run_relative, paths in self.required_artifacts.items():
            run_dir = base_dir.joinpath(*PurePosixPath(run_relative).parts)
            for relative in paths:
                path = run_dir.joinpath(*PurePosixPath(relative).parts)
                if not self._safe_existing(run_dir, path):
                    return False
        return self.complete

    def to_dict(self) -> dict[str, Any]:
        """Return an independent ordinary JSON mapping."""
        return copy.deepcopy(dict(self))

    @staticmethod
    def _fingerprint_mapping(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _fingerprint_regular(path: Path) -> dict[str, Any] | None:
        try:
            metadata = path.lstat()
        except OSError:
            return None
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            return None
        if attributes & reparse:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        if (
            after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ino != metadata.st_ino
        ):
            return None
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }

    @staticmethod
    def _regular_tree_files(
        root: Path,
        *,
        excluded_roots: set[str] | None = None,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        """List regular framework outputs without following links or reparse points."""
        if not root.exists():
            return (), ()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        try:
            root_metadata = root.lstat()
        except OSError:
            return (), (root,)
        root_attributes = getattr(root_metadata, "st_file_attributes", 0)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_ISLNK(root_metadata.st_mode)
            or root_attributes & reparse
        ):
            return (), (root,)

        excluded = excluded_roots or set()
        regular: list[Path] = []
        unsafe: list[Path] = []
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                unsafe.append(directory)
                continue
            for entry in entries:
                path = Path(entry.path)
                if entry.name.startswith(".") or (directory == root and entry.name in excluded):
                    continue
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    unsafe.append(path)
                    continue
                attributes = getattr(metadata, "st_file_attributes", 0)
                if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
                    unsafe.append(path)
                elif stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    regular.append(path)
                else:
                    unsafe.append(path)
        return (
            tuple(sorted(regular, key=lambda path: path.as_posix())),
            tuple(sorted(unsafe, key=lambda path: path.as_posix())),
        )

    @staticmethod
    def _validate_required_path(value: str) -> str:
        if not value or "\0" in value or "\\" in value:
            raise ValueError("required artifact paths must be non-empty relative POSIX paths")
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts or value.startswith("//"):
            raise ValueError(f"required artifact escapes the run: {value!r}")
        if len(value) >= 2 and value[0].isalpha() and value[1] == ":":
            raise ValueError(f"required artifact uses a drive path: {value!r}")
        return pure.as_posix()

    @staticmethod
    def _safe_existing(root: Path, path: Path) -> bool:
        try:
            relative = path.relative_to(root)
            cursor = root
            for part in relative.parts:
                cursor = cursor / part
                metadata = cursor.lstat()
                attributes = getattr(metadata, "st_file_attributes", 0)
                reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                if stat.S_ISLNK(metadata.st_mode) or attributes & reparse:
                    return False
            return path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            return False
