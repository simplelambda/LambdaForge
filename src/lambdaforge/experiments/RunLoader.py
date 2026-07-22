"""Reconstruct trained models and tasks from persisted run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from lambdaforge.experiments.CheckpointChoice import CheckpointChoice
from lambdaforge.experiments.ExperimentConfig import ExperimentConfig
from lambdaforge.experiments.ObjectFactory import ObjectFactory
from lambdaforge.experiments.retention.CheckpointResolver import CheckpointResolver
from lambdaforge.experiments.RunResult import RunResult
from lambdaforge.plugins.PluginRegistry import PluginRegistry


class RunLoader:
    """Locate a run, rebuild its objects and load a selected checkpoint.

    The materialized ``config.yaml`` is always the source of truth. A variant
    may be omitted only when the suite contains exactly one variant directory.
    """

    def __init__(self, output_root: str | Path = "runs/experiments") -> None:
        self.output_root = Path(output_root)

    def find_run_dir(
        self,
        experiment: str,
        seed: Any = None,
        variant: str | None = None,
    ) -> Path:
        """Resolve and validate a concrete run directory."""
        base_dir = self.output_root / experiment
        if not base_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {base_dir}")
        if variant is None:
            variants = sorted(
                child.name
                for child in base_dir.iterdir()
                if child.is_dir() and any(child.glob("seed=*"))
            )
            if len(variants) != 1:
                if not variants:
                    raise FileNotFoundError(f"No variants with runs under {base_dir}")
                raise ValueError(
                    f"Experiment {experiment!r} has several variants; choose one of {variants}."
                )
            variant = variants[0]
        seed_segment = f"seed={seed}" if seed is not None else "seed=none"
        run_dir = base_dir / variant / seed_segment
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    @staticmethod
    def load_config(run_dir: str | Path) -> dict[str, Any]:
        """Load and validate a run's materialized YAML mapping."""
        path = Path(run_dir) / "config.yaml"
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise TypeError(f"Materialized config must be a mapping: {path}")
        return ExperimentConfig(data, source=path).as_dict()

    @staticmethod
    def load_result(run_dir: str | Path) -> RunResult:
        """Load the typed terminal result stored beside a materialized run."""
        return RunResult.read_json(Path(run_dir) / "result.json")

    @staticmethod
    def resolve_checkpoint(
        run_dir: str | Path,
        which: CheckpointChoice | str = CheckpointChoice.BEST,
    ) -> Path:
        """Locate an exact or automatic local checkpoint in a moved run."""
        run_dir = Path(run_dir)
        choice = CheckpointChoice(which)
        result_path = run_dir / "result.json"
        result: RunResult | None = None
        if result_path.exists():
            try:
                result = RunResult.read_json(result_path)
            except (OSError, TypeError, ValueError):
                result = None
        resolver = CheckpointResolver(run_dir)
        if choice is CheckpointChoice.BEST:
            selected = resolver.best(result)
        elif choice is CheckpointChoice.LAST:
            selected = resolver.last(result)
        else:
            selected = resolver.best(result) or resolver.last(result) or resolver.latest(result)
        if selected is None:
            raise FileNotFoundError(
                f"No unambiguous {choice.value!r} checkpoint under {resolver.checkpoint_dir}"
            )
        return selected

    def load_model(
        self,
        experiment: str,
        seed: Any = None,
        variant: str | None = None,
        which: CheckpointChoice | str = CheckpointChoice.BEST,
        strict: bool = True,
    ) -> torch.nn.Module:
        """Rebuild the bare model, load its weights and switch it to eval mode."""
        run_dir = self.find_run_dir(experiment, seed, variant)
        config = self.load_config(run_dir)
        model = ObjectFactory.build(config["model"], plugins=PluginRegistry.default())
        state = self._load_state(self.resolve_checkpoint(run_dir, which))
        prefixed = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
        model.load_state_dict(prefixed or state, strict=strict)
        model.eval()
        return model

    def load_task(
        self,
        experiment: str,
        seed: Any = None,
        variant: str | None = None,
        which: CheckpointChoice | str = CheckpointChoice.BEST,
        strict: bool = True,
    ) -> torch.nn.Module:
        """Rebuild the complete training task and load its checkpoint state."""
        from lambdaforge.experiments.ExperimentRunner import ExperimentRunner

        run_dir = self.find_run_dir(experiment, seed, variant)
        config = self.load_config(run_dir)
        plugins = PluginRegistry.default()
        model = ObjectFactory.build(config["model"], plugins=plugins)
        task, _ = ExperimentRunner()._build_task(config, model, plugins)
        task.load_state_dict(
            self._load_state(self.resolve_checkpoint(run_dir, which)), strict=strict
        )
        task.eval()
        return task

    @staticmethod
    def _load_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint must contain a mapping: {checkpoint_path}")
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint state_dict must be a mapping: {checkpoint_path}")
        return state
