"""Sequential bounded FGW blocks with optional native Work.resume_map persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from threadpoolctl import threadpool_limits

from .FGWMatrixBlock import FGWMatrixBlock
from .FGWMatrixResult import FGWMatrixResult
from .FusedGromovWassersteinDistance import FusedGromovWassersteinDistance
from .matrix_identity import json_digest, measure_digest, solver_descriptor
from .StructuredMeasure import StructuredMeasure

if TYPE_CHECKING:
    from lambdaforge.work.Work import Work


class FGWMatrixBuilder:
    """Enumerate row-major comparisons, optionally mirror an explicitly symmetric triangle.

    No hidden concurrency or filesystem protocol. Work blocks are native JSON
    resume_map items, keyed by exact numerical content/solver identity. Failed
    blocks are not reused. On interruption only the current block is repeated.
    Use blocks() to avoid a dense output; build() guards dense output plus workspace.
    Budgets are conservative allocation estimates, not hard process RSS limits.
    """

    def __init__(
        self,
        distance: FusedGromovWassersteinDistance,
        *,
        block_size: int = 32,
        max_memory_bytes: int = 512 * 1024**2,
        error_policy: str = "raise",
        feature_encoder: Callable[[Any], bytes] | None = None,
        metric_identity: str | None = None,
        threads: int = 1,
    ) -> None:
        if not isinstance(distance, FusedGromovWassersteinDistance):
            raise TypeError("FGWMatrixBuilder requires FusedGromovWassersteinDistance.")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        if (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes < 1
        ):
            raise ValueError("max_memory_bytes must be a positive integer.")
        if error_policy not in {"raise", "record"}:
            raise ValueError("error_policy must be 'raise' or explicit 'record'.")
        if feature_encoder is not None and not callable(feature_encoder):
            raise TypeError("feature_encoder must be callable or null.")
        if metric_identity is not None and (
            not isinstance(metric_identity, str) or not metric_identity.strip()
        ):
            raise ValueError(
                "metric_identity must be a nonempty authored implementation identifier."
            )
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        self.distance, self.block_size = distance, block_size
        self.max_memory_bytes, self.error_policy = max_memory_bytes, error_policy
        self.feature_encoder, self.metric_identity = feature_encoder, metric_identity
        self.threads = threads

    def _collections(
        self,
        objects: Sequence[StructuredMeasure],
        targets: Sequence[StructuredMeasure] | None,
        symmetric: bool,
    ) -> tuple[tuple[StructuredMeasure, ...], tuple[StructuredMeasure, ...]]:
        left = tuple(objects)
        right = left if targets is None else tuple(targets)
        if (
            not left
            or not right
            or any(not isinstance(obj, StructuredMeasure) for obj in (*left, *right))
        ):
            raise ValueError("FGW blocks require nonempty collections of StructuredMeasure.")
        if not isinstance(symmetric, bool) or (symmetric and targets is not None):
            raise ValueError("symmetric=True is only valid for one square collection.")
        return left, right

    def _workspace(
        self, left: Sequence[StructuredMeasure], right: Sequence[StructuredMeasure]
    ) -> int:
        n, m = max(len(obj.structure) for obj in left), max(len(obj.structure) for obj in right)

        def feature_size(obj: StructuredMeasure) -> int:
            if isinstance(obj.features, np.ndarray):
                return int(obj.features.size) * 8
            if isinstance(self.distance.feature_metric, str) and isinstance(obj.features, tuple):
                return 8 * sum(
                    len(row) if isinstance(row, (list, tuple, np.ndarray)) else 1
                    for row in obj.features
                )
            return 0  # Opaque callbacks/encoders have caller-owned, unconstrained workspace.

        feature_bytes = sum(
            max((feature_size(obj) for obj in items), default=0) for items in (left, right)
        )
        return (
            8 * (4 * (n * n + m * m) + 12 * n * m)
            + 2 * feature_bytes
            + self.block_size * 4096
            + (len(left) + len(right)) * 256
        )

    def build(
        self,
        objects: Sequence[StructuredMeasure],
        targets: Sequence[StructuredMeasure] | None = None,
        *,
        symmetric: bool = False,
        work: Work | None = None,
    ) -> FGWMatrixResult:
        """Build square or rectangular matrix; NaN always means an explicit failed comparison."""
        left, right = self._collections(objects, targets, symmetric)
        comparisons = len(left) * len(right) if not symmetric else len(left) * (len(left) + 1) // 2
        n_blocks = (comparisons + self.block_size - 1) // self.block_size
        summary_bytes = 16 * 1024 if self.error_policy == "record" else 4 * 1024
        estimate = (
            16 * len(left) * len(right) + summary_bytes * n_blocks + self._workspace(left, right)
        )
        if estimate > self.max_memory_bytes:
            raise MemoryError(
                "FGW matrix output/diagnostics plus workspace exceeds max_memory_bytes; "
                "use blocks()."
            )
        values = np.full((len(left), len(right)), np.nan)
        summaries = []
        identity = ""
        for block in self.blocks(
            left, None if targets is None else right, symmetric=symmetric, work=work
        ):
            identity = block.identity
            for (i, j), value in zip(block.pairs, block.values, strict=True):
                values[i, j] = value
                if symmetric:
                    values[j, i] = value
            summaries.append(block.diagnostics)
        return FGWMatrixResult(values, identity, symmetric, tuple(summaries))

    def blocks(
        self,
        objects: Sequence[StructuredMeasure],
        targets: Sequence[StructuredMeasure] | None = None,
        *,
        symmetric: bool = False,
        work: Work | None = None,
    ) -> Iterator[FGWMatrixBlock]:
        """Yield bounded row-major blocks; the caller owns their consumption/publishing."""
        left, right = self._collections(objects, targets, symmetric)
        if self._workspace(left, right) > self.max_memory_bytes:
            raise MemoryError("FGW block workspace exceeds max_memory_bytes.")
        descriptor = solver_descriptor(self.distance, self.metric_identity)
        descriptor["threads"] = self.threads
        descriptor["solver_class"] = (
            f"{type(self.distance).__module__}.{type(self.distance).__qualname__}"
        )
        identity = json_digest(
            {
                "solver": descriptor,
                "symmetric": symmetric,
                "left": [measure_digest(obj, self.feature_encoder) for obj in left],
                "right": [measure_digest(obj, self.feature_encoder) for obj in right],
            }
        )
        pairs = ((i, j) for i in range(len(left)) for j in range(i if symmetric else 0, len(right)))
        completed = 0
        total = len(left) * len(right) if not symmetric else len(left) * (len(left) + 1) // 2
        while chunk := tuple(islice(pairs, self.block_size)):
            expected = [[i, j] for i, j in chunk]
            key = json_digest({"identity": identity, "pairs": expected})

            def calculate(
                _item: str,
                _chunk: tuple[tuple[int, int], ...] = chunk,
                _expected: list[list[int]] = expected,
            ) -> dict[str, Any]:
                values: list[float | None] = []
                errors: list[dict[str, Any]] = []
                converged, iterations, marginal = 0, 0, 0.0
                for i, j in _chunk:
                    try:
                        with threadpool_limits(limits=self.threads):
                            result = self.distance.solve(left[i], right[j])
                    except (MemoryError, ImportError):
                        raise
                    except Exception as error:
                        if self.error_policy == "raise":
                            raise
                        values.append(None)
                        if len(errors) < 8:
                            errors.append(
                                {
                                    "pair": [i, j],
                                    "type": type(error).__name__,
                                    "message": str(error)[:256],
                                }
                            )
                        continue
                    values.append(result.value)
                    converged += int(result.converged)
                    iterations += result.iterations
                    marginal = max(marginal, result.marginal_error)
                record: dict[str, Any] = {
                    "identity": identity,
                    "pairs": _expected,
                    "values": values,
                    "diagnostics": {
                        "pairs": len(_chunk),
                        "first_pair": _expected[0],
                        "last_pair": _expected[-1],
                        "failed": values.count(None),
                        "converged": converged,
                        "iterations": iterations,
                        "max_marginal_error": marginal,
                        "errors": errors,
                    },
                }
                return {**record, "sha256": json_digest(record)}

            def valid(
                record: Any, _expected: list[list[int]] = expected, _size: int = len(chunk)
            ) -> bool:
                if not isinstance(record, dict):
                    return False
                try:
                    content = {k: v for k, v in record.items() if k != "sha256"}
                    return (
                        record["identity"] == identity
                        and record["pairs"] == _expected
                        and isinstance(record.get("diagnostics"), dict)
                        and len(record["values"]) == _size
                        and all(
                            isinstance(v, (int, float))
                            and not isinstance(v, bool)
                            and np.isfinite(v)
                            and v >= 0
                            for v in record["values"]
                        )
                        and record["sha256"] == json_digest(content)
                    )
                except (KeyError, TypeError, ValueError):
                    return False

            record = (
                calculate(key)
                if work is None
                else work.resume_map(
                    [key],
                    calculate,
                    key=lambda item: item,
                    name=f"fgw-blocks-{identity}",
                    workers=1,
                    retries=0,
                    validate=valid,
                )[0]
            )
            # A freshly recorded failure is intentional partial evidence, not reusable success.
            yield FGWMatrixBlock(
                chunk,
                np.array([np.nan if v is None else v for v in record["values"]]),
                identity,
                MappingProxyType(record["diagnostics"]),
            )
            completed += len(chunk)
            if work is not None:
                work.progress.update(completed=completed, total=total, message="FGW comparisons")
