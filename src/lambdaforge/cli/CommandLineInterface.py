"""Command-line entry object for LambdaForge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lambdaforge.experiments.Experiment import Experiment


class CommandLineInterface:
    """Parse CLI arguments and dispatch them to the public object API."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Run the CLI and return a process exit code."""
        parser = cls._parser()
        arguments = parser.parse_args(argv)
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
        aggregate = subparsers.add_parser("aggregate", help="Rebuild suite aggregates from disk.")
        aggregate.add_argument("config", type=Path)
        aggregate.add_argument("--no-plots", action="store_true")
        return parser
