"""The single public YAML-executable LambdaForge contract."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from lambdaforge.work.cache import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection
from lambdaforge.work.models import WorkConfiguration, WorkInput, WorkResources, WorkTrial
from lambdaforge.work.outputs import OutputCollection
from lambdaforge.work.runtime import (
    MetricCollection,
    ProgressReporter,
    WorkRuntime,
)
from lambdaforge.work.tools import ToolService

T = TypeVar("T")
R = TypeVar("R")


class Work:
    """Base class for every scientific unit executable from LambdaForge YAML.

    Subclasses implement exactly one public lifecycle entry point, ``run``. LambdaForge creates
    the instance, binds immutable planning data and managed runtime services, and then invokes it.
    Scientific parameters belong to ``run`` rather than the constructor.
    """

    __slots__ = ("_runtime",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        constructor = cls.__dict__.get("__init__")
        if constructor is not None:
            signature = inspect.signature(constructor)
            required = [
                parameter.name
                for parameter in tuple(signature.parameters.values())[1:]
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            ]
            if required:
                raise TypeError(
                    f"{cls.__module__}.{cls.__qualname__}.__init__ has required parameters "
                    f"{required}. Work constructors are framework-owned; put scientific "
                    "parameters on run()."
                )

    def __init__(self) -> None:
        self._runtime: WorkRuntime | None = None

    def run(self, **parameters: Any) -> Any:
        """Perform the scientific operation; every executable Work must override this method."""
        del parameters
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented.")

    @property
    def name(self) -> str:
        """Human Work name selected before execution (read-only, active execution only)."""
        return self._bound().config.name

    @property
    def config(self) -> WorkConfiguration:
        """Immutable normalized class, parameters and requested resources."""
        return self._bound().config

    @property
    def inputs(self) -> Mapping[str, WorkInput]:
        """Immutable explicitly typed file/dataset inputs, excluding ordinary scalar parameters."""
        return self._bound().inputs

    @property
    def outputs(self) -> OutputCollection:
        """Mutable named output registrar, finalized when ``run`` returns."""
        return self._bound().outputs

    @property
    def metrics(self) -> MetricCollection:
        """Append-only scalar metric history for live status, results and HPO."""
        return self._bound().metrics

    @property
    def checkpoints(self) -> CheckpointCollection:
        """Run-owned resumable state shared by compatible Attempts."""
        return self._bound().checkpoints

    @property
    def cache(self) -> WorkCache:
        """Reconstructible identity-scoped storage eligible for ``lf clean``."""
        return self._bound().cache

    @property
    def tools(self) -> ToolService:
        """Resolve and execute external scientific tools with recorded provenance."""
        return self._bound().tools

    @property
    def progress(self) -> ProgressReporter:
        """Optional bounded progress snapshot consumed by monitoring."""
        return self._bound().progress

    def log(self, message: object, *, level: str = "info") -> None:
        """Emit a timestamped message visible in Attempt and scheduler logs immediately."""
        self._bound().log.emit(message, level=level)

    @property
    def resources(self) -> WorkResources:
        """Immutable requested resources for this Run."""
        return self._bound().resources

    @property
    def seed(self) -> int | None:
        """Current seed, or ``None`` for an unseeded Run."""
        return self._bound().seed

    @property
    def trial(self) -> WorkTrial | None:
        """Immutable parameter-study member metadata, without controller internals."""
        return self._bound().trial

    @property
    def run_dir(self) -> Path:
        """Durable filesystem root owned by this Attempt."""
        return self._bound().run_dir

    @property
    def temp_dir(self) -> Path:
        """Ephemeral Attempt storage removed after finalization."""
        return self._bound().temp_dir

    @property
    def source_dir(self) -> Path:
        """Read-only consumer project/source directory."""
        return self._bound().source_dir

    @property
    def resuming(self) -> bool:
        """Whether compatible checkpoints existed when this Attempt began."""
        return self._bound().resuming

    def map(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        key: Callable[[T], str] | str | None = None,
        workers: int = 1,
        executor: str = "thread",
        resume: bool | None = None,
        name: str | None = None,
        validate: Callable[[R], bool] | None = None,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> list[R]:
        """Apply ``function`` with ordered concurrency and no hidden persistence.

        ``key`` remains a compatibility bridge to the old resumable behavior.
        New code should call :meth:`resume_map` when per-item checkpoints are wanted.
        """
        if key is None:
            if resume is not None or validate is not None:
                raise ValueError("resume and validate require Work.resume_map(..., key=...).")
            return self._bound().map(
                items,
                function,
                workers=workers,
                executor=executor,
                name=name,
                retries=retries,
                retry_backoff=retry_backoff,
            )
        return self.resume_map(
            items,
            function,
            key=key,
            workers=workers,
            executor=executor,
            resume=True if resume is None else resume,
            name=name,
            validate=validate,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    def resume_map(
        self,
        items: Iterable[T],
        function: Callable[[T], R],
        *,
        key: Callable[[T], str] | str,
        workers: int = 1,
        executor: str = "thread",
        resume: bool = True,
        name: str | None = None,
        validate: Callable[[R], bool] | None = None,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> list[R]:
        """Apply with explicit stable keys and dependency-aware per-item resume."""
        return self._bound().resume_map(
            items,
            function,
            key=key,
            workers=workers,
            executor=executor,
            resume=resume,
            name=name,
            validate=validate,
            retries=retries,
            retry_backoff=retry_backoff,
        )

    def _bind(self, runtime: WorkRuntime) -> None:
        if getattr(self, "_runtime", None) is not None:
            raise RuntimeError("A Work instance cannot be rebound to another execution.")
        self._runtime = runtime

    def _bound(self) -> WorkRuntime:
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            raise RuntimeError(
                "Work runtime properties are available only while LambdaForge executes this Work."
            )
        return runtime
