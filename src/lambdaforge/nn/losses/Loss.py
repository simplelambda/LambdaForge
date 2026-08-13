"""Implementation of the Loss object."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn as nn


class Loss(nn.Module, ABC):
    r"""Base class for training losses.

    A loss receives the model outputs and the original batch, and returns a
    scalar tensor that can be used for backpropagation.

    Losses are usually used inside the training loop:

        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()

    Unlike metrics, losses must return ``torch.Tensor`` objects, not Python
    floats, because gradients must be preserved.

    Mixed precision
    ---------------
    When the model trains under automatic mixed precision (``"16-mixed"`` /
    ``"bf16-mixed"``) or true low precision (``"16-true"`` / ``"bf16-true"``),
    many loss computations become numerically unsafe: reductions over many
    surface points, ``log``/``exp`` and cross-entropy accumulate large relative
    error in ``float16``. To guarantee correctness, :meth:`__call__` evaluates
    the loss in :meth:`compute_dtype` (``float32`` by default) whenever the
    incoming precision is reduced and :attr:`supports_reduced_precision` is
    ``False`` (the conservative default). This costs almost nothing because the
    loss is a tiny fraction of a training step, and it mirrors how PyTorch AMP
    itself keeps losses in ``float32``.

    A subclass whose every operation is provably stable in reduced precision may
    set :attr:`supports_reduced_precision` to ``True`` to skip the upcast.

    Parameters
    ----------
    name : str
        Name used to identify the loss in logs, callbacks or reports.
    weight : float
        Multiplicative factor applied to the computed loss. This is useful
        when combining several losses.
    """

    #: Whether this loss is numerically safe to evaluate directly in reduced
    #: precision (``float16`` / ``bfloat16``). Defaults to ``False`` so that
    #: functionality is guaranteed out of the box; override to ``True`` only for
    #: losses that are provably stable in reduced precision.
    supports_reduced_precision: bool = False

    @staticmethod
    def _autocast_reduced_dtype() -> torch.dtype | None:
        """Return the reduced dtype of an active autocast context, if any."""
        for device in ("cuda", "cpu"):
            try:
                enabled = cast(Any, torch.is_autocast_enabled)(device)
            except TypeError:  # Torch < 2.4 has device-specific legacy functions.
                enabled = (
                    torch.is_autocast_enabled()
                    if device == "cuda"
                    else bool(torch.is_autocast_cpu_enabled())
                )
            if not enabled:
                continue
            getter = getattr(torch, "get_autocast_dtype", None)
            dtype = (
                getter(device)
                if getter is not None
                else torch.get_autocast_gpu_dtype()
                if device == "cuda"
                else torch.get_autocast_cpu_dtype()
            )
            if dtype in (torch.float16, torch.bfloat16):
                return dtype
        return None

    @staticmethod
    def _tensor_reduced_dtype(mapping: Mapping[str, Any]) -> torch.dtype | None:
        """Return a reduced floating dtype found in a mapping's tensors."""
        for value in mapping.values():
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and value.dtype in (torch.float16, torch.bfloat16)
            ):
                return value.dtype
        return None

    @staticmethod
    def _upcast_reduced_floats(mapping: Mapping[str, Any], dtype: torch.dtype) -> dict[str, Any]:
        """Copy a mapping while casting only reduced floating tensors."""
        return {
            key: (
                value.to(dtype)
                if torch.is_tensor(value)
                and value.is_floating_point()
                and value.dtype in (torch.float16, torch.bfloat16)
                else value
            )
            for key, value in mapping.items()
        }

    @staticmethod
    @contextlib.contextmanager
    def _autocast_disabled():
        """Disable autocast on relevant devices for the enclosed loss call."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(torch.autocast(device_type="cpu", enabled=False))
            if torch.cuda.is_available():
                stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
            yield

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.name = name
        self.weight = weight

    def supports_precision(self, dtype: torch.dtype) -> bool:
        r"""Return whether the loss can be evaluated directly in ``dtype``.

        Full precision (``float32`` / ``float64``) is always supported. Reduced
        precision is supported only when :attr:`supports_reduced_precision` is
        ``True``. This is the method to query a loss' precision compatibility.
        """
        if dtype in (torch.float32, torch.float64):
            return True
        return self.supports_reduced_precision

    def compute_dtype(self) -> torch.dtype:
        r"""Dtype used to evaluate the loss when the incoming precision is unsafe.

        ``float32`` is stable for every loss in the project. Override to request
        a different fallback (for example ``float64``) if a subclass ever needs
        it.
        """
        return torch.float32

    def __call__(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        reduced_dtype = self._autocast_reduced_dtype() or self._tensor_reduced_dtype(outputs)

        if reduced_dtype is None or self.supports_precision(reduced_dtype):
            result = super().__call__(outputs, batch, context)
            return self._require_scalar(result)

        with self._autocast_disabled():
            safe_outputs = self._upcast_reduced_floats(outputs, self.compute_dtype())
            result = super().__call__(safe_outputs, batch, context)
            return self._require_scalar(result)

    @staticmethod
    def _require_scalar(result: Any) -> torch.Tensor:
        """Fail early when a training loss violates the scalar contract."""
        if not torch.is_tensor(result):
            raise TypeError("Loss.forward() must return a torch.Tensor.")
        if result.ndim != 0:
            raise ValueError(
                "Loss.forward() must return a scalar tensor; apply a mean or sum reduction."
            )
        return result

    @abstractmethod
    def forward(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> torch.Tensor:
        r"""Compute the loss.

        Parameters
        ----------
        outputs : Mapping[str, Any]
            Model outputs for the current batch. For example, this may contain
            keys such as ``"logits"``, ``"prediction"`` or ``"reconstruction"``.
        batch : Mapping[str, Any]
            Original input batch. For supervised losses, this usually contains
            the target tensor.
        context : object | None
            Optional runtime context, such as a trainer, epoch state or custom
            metadata.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor. This tensor should keep the computation graph
            so that ``backward()`` can be called on it.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"
