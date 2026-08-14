"""Process entry point for one durable dataset build job."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from lambdaforge.data.DatasetBuildService import DatasetBuildService
from lambdaforge.data.DatasetRecipeConfig import DatasetRecipeConfig


class DatasetBuildWorker:
    """Execute a recipe inside the target job allocation and return a process exit code."""

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Parse the intentionally small internal worker command."""
        parser = argparse.ArgumentParser(prog="lambdaforge-dataset-build-worker")
        parser.add_argument("recipe")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--force-stage", action="append", default=[])
        arguments = parser.parse_args(argv)
        result = DatasetBuildService().build(
            DatasetRecipeConfig.from_yaml(arguments.recipe),
            force=arguments.force,
            force_stages=arguments.force_stage,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(DatasetBuildWorker.main())
