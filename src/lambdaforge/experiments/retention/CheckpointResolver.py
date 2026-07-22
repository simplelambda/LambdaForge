"""Portable and conservative checkpoint-role resolution."""

from __future__ import annotations

import re
from pathlib import Path

from lambdaforge.experiments.retention.ArtifactPathGuard import ArtifactPathGuard
from lambdaforge.experiments.retention.CheckpointRetention import CheckpointRetention
from lambdaforge.experiments.RunResult import RunResult


class CheckpointResolver:
    """Resolve LambdaForge best/last roles strictly inside one run directory."""

    _BEST_PATTERN = re.compile(r"^best-.+[.]ckpt$")
    _EPOCH_PATTERN = re.compile(r"^epoch-.+[.]ckpt$")
    _NAMED_EPOCH_PATTERN = re.compile(r"epoch[=_-]?(?P<epoch>[0-9]+)")
    _NUMBER_PATTERN = re.compile(r"(?P<epoch>[0-9]+)")

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"

    def candidates(self) -> tuple[Path, ...]:
        """Return safe direct checkpoint files in deterministic name order."""
        if not self.checkpoint_dir.is_dir() or self.checkpoint_dir.is_symlink():
            return ()
        candidates: list[Path] = []
        for path in sorted(self.checkpoint_dir.glob("*.ckpt"), key=lambda item: item.name):
            if ArtifactPathGuard.relative_regular_file(self.checkpoint_dir, path) is not None:
                candidates.append(path)
        return tuple(candidates)

    def best(self, result: RunResult | None = None) -> Path | None:
        """Resolve the recorded or uniquely generated best checkpoint."""
        recorded = self._recorded(result.best_model_path if result is not None else None)
        if recorded is not None:
            return recorded
        candidates = [path for path in self.candidates() if self._BEST_PATTERN.fullmatch(path.name)]
        if len(candidates) == 1:
            return candidates[0]
        epoch = self._best_epoch(result)
        if epoch is not None:
            selected = [path for path in candidates if self._epoch(path) == epoch]
            if len(selected) == 1:
                return selected[0]
            epoch_candidates = [
                path
                for path in self.candidates()
                if self._epoch(path) is not None and self._epoch(path) == epoch
            ]
            if len(epoch_candidates) == 1:
                return epoch_candidates[0]
        return None

    def last(self, result: RunResult | None = None) -> Path | None:
        """Resolve deterministic last.ckpt, its recorded path or latest epoch."""
        conventional = self.checkpoint_dir / "last.ckpt"
        if conventional in self.candidates():
            return conventional
        recorded = self._recorded(result.last_model_path if result is not None else None)
        if recorded is not None:
            return recorded
        epoch_candidates = [
            (epoch, path)
            for path in self.candidates()
            if self._EPOCH_PATTERN.fullmatch(path.name) and (epoch := self._epoch(path)) is not None
        ]
        if not epoch_candidates:
            return None
        latest_epoch = max(epoch for epoch, _ in epoch_candidates)
        selected = [path for epoch, path in epoch_candidates if epoch == latest_epoch]
        return selected[0] if len(selected) == 1 else None

    def latest(self, result: RunResult | None = None) -> Path | None:
        """Resolve last, then best, then the newest safe local checkpoint."""
        selected = self.last(result) or self.best(result)
        if selected is not None:
            return selected
        candidates = self.candidates()
        if not candidates:
            return None
        return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))

    def retained(
        self,
        policy: CheckpointRetention,
        result: RunResult | None,
    ) -> tuple[Path, ...] | None:
        """Return selected paths, or None when any requested role is ambiguous."""
        candidates = self.candidates()
        if policy is CheckpointRetention.ALL:
            return candidates
        selected: list[Path] = []
        for role in policy.roles:
            path = self.best(result) if role == "best" else self.last(result)
            if path is None:
                return None
            if path not in selected:
                selected.append(path)
        return tuple(selected)

    def _recorded(self, value: str | None) -> Path | None:
        if not value:
            return None
        recorded = Path(value)
        candidates = self.candidates()
        if recorded.exists():
            try:
                resolved = recorded.resolve()
                for candidate in candidates:
                    if candidate.resolve() == resolved:
                        return candidate
            except OSError:
                pass
        for candidate in candidates:
            if candidate.name == recorded.name:
                return candidate
        return None

    @classmethod
    def _epoch(cls, path: Path) -> int | None:
        if not (cls._BEST_PATTERN.fullmatch(path.name) or cls._EPOCH_PATTERN.fullmatch(path.name)):
            return None
        match = cls._NAMED_EPOCH_PATTERN.search(path.stem)
        if match is not None:
            return int(match.group("epoch"))
        values = cls._NUMBER_PATTERN.findall(path.stem)
        return int(values[-1]) if values else None

    @staticmethod
    def _best_epoch(result: RunResult | None) -> int | None:
        if result is None or result.best_epoch_metrics is None:
            return None
        value = result.best_epoch_metrics.get("epoch")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)
