"""Public contract for immediate, checkpoint-aware post-run work."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lambdaforge.experiments.postrun.PostRunContext import PostRunContext
from lambdaforge.experiments.postrun.PostRunResult import PostRunResult


class PostRunAction(ABC):
    """Perform bounded analysis after one training run succeeds."""

    @abstractmethod
    def run(self, context: PostRunContext) -> PostRunResult:
        """Produce structured outputs and artifact declarations for one run."""
