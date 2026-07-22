"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.experiments.Experiment import Experiment
from lambdaforge.experiments.ExperimentValidator import ExperimentValidator
from lambdaforge.experiments.migrations.ExperimentConfigMigrator import (
    ExperimentConfigMigrator,
)
from lambdaforge.experiments.migrations.MigrationPreviewFormat import (
    MigrationPreviewFormat,
)
from lambdaforge.experiments.results.ResultCatalog import ResultCatalog
from lambdaforge.plugins.PluginKind import PluginKind
from lambdaforge.plugins.PluginRegistry import PluginRegistry


class CommandLineInterface:
    """Parse CLI arguments and dispatch them to the public object API."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the CLI and return a process exit code."""
        parser = cls._parser()
        arguments = parser.parse_args(argv)
        if arguments.command == "plugins":
            descriptors = PluginRegistry.default().discover(arguments.kind)
            payload = [descriptor.to_dict() for descriptor in descriptors]
            if arguments.json:
                print(json.dumps(payload, indent=2))
            elif not payload:
                print("No LambdaForge plugins found.")
            else:
                for plugin in payload:
                    provider = plugin["distribution"] or "unknown distribution"
                    version = f" {plugin['version']}" if plugin["version"] else ""
                    print(
                        f"{plugin['kind']}:{plugin['name']} -> {plugin['value']} "
                        f"[{provider}{version}]"
                    )
            return 0
        if arguments.command == "validate":
            report = ExperimentValidator().validate_file(
                arguments.config,
                check_imports=not arguments.no_imports,
            )
            print(json.dumps(report.to_dict(), indent=2) if arguments.json else report.summary())
            return 0 if report.is_valid else 1
        if arguments.command == "migrate":
            if arguments.force and arguments.output is None:
                print("ERROR: --force requires --output.", file=sys.stderr)
                return 1
            if arguments.check and arguments.output is not None:
                print("ERROR: --check cannot be combined with --output.", file=sys.stderr)
                return 1
            try:
                result = ExperimentConfigMigrator.default().preview_file(
                    arguments.config,
                    target_version=arguments.target_version,
                )
                rendered = result.render(arguments.format)
                if rendered:
                    print(rendered, end="" if rendered.endswith("\n") else "\n")
                elif arguments.format == MigrationPreviewFormat.DIFF.value:
                    print(
                        f"Already at experiment Schema {result.target_version}; "
                        "no migration required."
                    )
                if arguments.check:
                    return 1 if result.changed else 0
                if arguments.output is not None:
                    result.write_yaml(arguments.output, overwrite=arguments.force)
                    print(f"Wrote migrated YAML: {arguments.output}", file=sys.stderr)
                return 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        if arguments.command == "retain":
            try:
                experiment = Experiment.from_yaml(arguments.config)
                retention_result = (
                    experiment.apply_retention()
                    if arguments.apply
                    else experiment.preview_retention()
                )
                print(
                    json.dumps(retention_result.to_dict(), indent=2)
                    if arguments.json
                    else retention_result.summary()
                )
                successful = {"applied", "already_applied"} if arguments.apply else {"preview"}
                return 0 if retention_result.status.value in successful else 1
            except Exception as error:
                if arguments.json:
                    print(
                        json.dumps(
                            {
                                "status": "error",
                                "error_type": error.__class__.__name__,
                                "message": str(error),
                            },
                            indent=2,
                        )
                    )
                else:
                    print(
                        f"ERROR: {error.__class__.__name__}: {error}",
                        file=sys.stderr,
                    )
                return 1
        if arguments.command == "results":
            try:
                source = arguments.source
                catalog = (
                    Experiment.from_yaml(source).result_catalog()
                    if source.suffix.lower() in {".yaml", ".yml"}
                    else ResultCatalog(source)
                )
                records = catalog.records(
                    status=arguments.status,
                    include_archived=not arguments.no_archived,
                )
                if arguments.duplicates:
                    duplicate_ids = {
                        record.attempt_id
                        for group in catalog.duplicate_groups().values()
                        for record in group
                    }
                    records = tuple(
                        record for record in records if record.attempt_id in duplicate_ids
                    )
                index_path = catalog.write_index() if arguments.write_index else None
                if arguments.json:
                    print(
                        json.dumps(
                            {
                                "summary": catalog.summary(),
                                "index_path": str(index_path) if index_path else None,
                                "records": [record.to_dict() for record in records],
                            },
                            indent=2,
                        )
                    )
                else:
                    print(catalog.summary())
                    for record in records:
                        location = "archived" if record.archived else "current"
                        print(
                            f"{record.attempt_id}  {record.status:<11} {location:<8} "
                            f"{record.result_path}"
                        )
                    if index_path is not None:
                        print(f"Wrote result index: {index_path}")
                return 2 if arguments.fail_on_ambiguous and catalog.ambiguous_successes() else 0
            except Exception as error:
                print(f"ERROR: {error.__class__.__name__}: {error}", file=sys.stderr)
                return 1
        experiment = Experiment.from_yaml(arguments.config)
        if arguments.command == "inspect":
            print(json.dumps(experiment.expand(), indent=2, default=str))
            return 0
        if arguments.command == "aggregate":
            experiment.aggregate(make_plots=not arguments.no_plots)
            return 0
        overrides = {
            "mode": arguments.mode,
            "gpus": arguments.gpus,
            "jobs_per_gpu": arguments.jobs_per_gpu,
            "devices_per_job": arguments.devices_per_job,
            "grace_seconds": arguments.grace_seconds,
        }
        results = experiment.run(
            dry_run=arguments.dry_run,
            execution_overrides=overrides,
            aggregate_plots=not arguments.no_plots,
        )
        failed = sum(result.get("status") == "failed" for result in results)
        print(f"LambdaForge finished {len(results)} run(s); failed={failed}.")
        return 1 if failed else 0

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="lambdaforge",
            description="Configure, run and aggregate reproducible ML experiments.",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        run = subparsers.add_parser("run", help="Execute an experiment YAML file.")
        run.add_argument("config", type=Path)
        run.add_argument("--dry-run", action="store_true")
        run.add_argument("--no-plots", action="store_true")
        run.add_argument("--mode", choices=("sequential", "parallel", "ddp"))
        run.add_argument("--gpus", help="Comma-separated logical GPU indices.")
        run.add_argument("--jobs-per-gpu", type=int)
        run.add_argument("--devices-per-job", type=int)
        run.add_argument("--grace-seconds", type=float)
        inspect = subparsers.add_parser("inspect", help="Print all expanded runs as JSON.")
        inspect.add_argument("config", type=Path)
        validate = subparsers.add_parser(
            "validate",
            help="Validate schema, expansion, resources and import paths without running.",
        )
        validate.add_argument("config", type=Path)
        validate.add_argument(
            "--no-imports",
            action="store_true",
            help="Skip importing target/ref paths and resolving plugins.",
        )
        validate.add_argument(
            "--json", action="store_true", help="Print a machine-readable report."
        )
        migrate = subparsers.add_parser(
            "migrate",
            help="Preview a versioned YAML configuration migration.",
        )
        migrate.add_argument("config", type=Path)
        migrate.add_argument(
            "--target-version",
            help="Exact MAJOR.MINOR target; defaults to the current Schema.",
        )
        migrate.add_argument(
            "--format",
            choices=tuple(value.value for value in MigrationPreviewFormat),
            default=MigrationPreviewFormat.DIFF.value,
            help="Preview as a unified diff, full YAML or stable JSON.",
        )
        migrate.add_argument(
            "--output",
            type=Path,
            help="Atomically write to a different path; the source is never overwritten.",
        )
        migrate.add_argument(
            "--force",
            action="store_true",
            help="Allow --output to replace an existing destination.",
        )
        migrate.add_argument(
            "--check",
            action="store_true",
            help="Return 1 when a migration is required, otherwise 0; never write.",
        )
        plugins = subparsers.add_parser(
            "plugins",
            help="List installed entry-point plugins without importing their modules.",
        )
        plugins.add_argument(
            "--kind",
            choices=tuple(kind.value for kind in PluginKind),
            help="Restrict discovery to one plugin contract.",
        )
        plugins.add_argument("--json", action="store_true", help="Print metadata as JSON.")
        aggregate = subparsers.add_parser("aggregate", help="Rebuild suite aggregates from disk.")
        aggregate.add_argument("config", type=Path)
        aggregate.add_argument("--no-plots", action="store_true")
        retain = subparsers.add_parser(
            "retain",
            help="Preview or explicitly apply artifact retention after complete aggregation.",
        )
        retain.add_argument("config", type=Path)
        retain.add_argument(
            "--apply",
            action="store_true",
            help="Apply the transaction; omission is always read-only.",
        )
        retain.add_argument("--json", action="store_true", help="Print a machine-readable result.")
        results = subparsers.add_parser(
            "results",
            help="Audit current and archived attempts by scientific configuration identity.",
        )
        results.add_argument("source", type=Path, help="Experiment YAML or result-tree root.")
        results.add_argument("--status", choices=("ok", "failed", "interrupted", "dry_run"))
        results.add_argument(
            "--no-archived",
            action="store_true",
            help="Only show the canonical result.json for each run directory.",
        )
        results.add_argument(
            "--duplicates",
            action="store_true",
            help="Only show identities with more than one attempt.",
        )
        results.add_argument(
            "--write-index",
            action="store_true",
            help="Atomically write .lambdaforge/result-index.json under the suite root.",
        )
        results.add_argument(
            "--fail-on-ambiguous",
            action="store_true",
            help="Return exit code 2 when one identity has multiple successful attempts.",
        )
        results.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
        return parser
