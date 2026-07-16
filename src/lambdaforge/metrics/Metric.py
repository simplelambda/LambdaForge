"""Implementation of the Metric object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class Metric(ABC):
    r"""Base class for evaluation metrics.

    A metric accumulates information batch by batch through ``update`` and
    returns its final scalar value through ``compute``.

    Metrics are usually used during training, validation or testing loops to
    track values such as accuracy, precision, recall, F1-score, mean absolute
    error, mean squared error, etc.

    The expected lifecycle is:

        metric.reset()

        for batch in dataloader:
            outputs = model(batch)
            metric.update(outputs, batch)

        value = metric.compute()

    Parameters
    ----------
    name : str
        Name used to identify the metric in logs, progress bars, callbacks,
        checkpoints or reports.
    higher_is_better : bool
        Whether larger values of this metric are better. This is useful for
        model selection, early stopping and checkpointing.

        Examples:

            Accuracy: ``higher_is_better=True``

            Loss: ``higher_is_better=False``

    Notes
    -----
    ``update`` may be called multiple times before ``compute``. Therefore,
    metric implementations should store cumulative state internally.

    If tensors are stored inside the metric, they should usually be detached
    from the computation graph to avoid keeping gradients alive unnecessarily.

    ``state_dict`` and ``load_state_dict`` are optional. Stateless metrics or
    metrics that do not need checkpointing can keep the default empty
    implementation.
    """

    def __init__(self, name: str, higher_is_better: bool = True) -> None:
        self.name = name
        self.higher_is_better = higher_is_better

    def __call__(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        r"""Update the metric state.

        This method is an alias for ``update`` so metrics can be called like
        regular functions.

        Parameters
        ----------
        outputs : Mapping[str, Any]
            Model outputs for the current batch.
        batch : Mapping[str, Any]
            Original batch used to compute the outputs.
        context : object | None
            Optional external context, such as a trainer, evaluator, epoch
            state or runtime metadata.
        """

        self.update(outputs, batch, context)

    @property
    def direction(self) -> str:
        r"""Optimization direction for this metric.

        Returns
        -------
        str
            ``"max"`` if higher values are better, otherwise ``"min"``.
        """

        return "max" if self.higher_is_better else "min"

    @abstractmethod
    def update(
        self,
        outputs: Mapping[str, Any],
        batch: Mapping[str, Any],
        context: object | None = None,
    ) -> None:
        r"""Update the metric using one batch.

        Parameters
        ----------
        outputs : Mapping[str, Any]
            Model outputs for the current batch. For example, this may contain
            keys such as ``"logits"``, ``"predictions"`` or ``"loss"``.
        batch : Mapping[str, Any]
            Input batch. For supervised metrics, this usually contains the
            target values.
        context : object | None
            Optional runtime context. Implementations may ignore it.
        """

        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        r"""Compute the current metric value.

        Returns
        -------
        float
            Scalar metric value computed from the accumulated state.
        """

        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        r"""Reset the internal metric state."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        r"""Return the metric state.

        This method can be overridden by stateful metrics that need to be
        checkpointed or resumed.

        Returns
        -------
        dict[str, Any]
            Serializable metric state.
        """

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Load a previously saved metric state.

        Parameters
        ----------
        state : Mapping[str, Any]
            State returned by ``state_dict``.
        """

        del state

    def distributed_state(self) -> dict[str, Any]:
        """Return mergeable accumulated state for distributed evaluation.

        Custom metrics used with DDP must override this method together with
        :meth:`merge_distributed_state`. Single-process training does not call
        either method.
        """
        raise NotImplementedError

    def merge_distributed_state(self, state: Mapping[str, Any]) -> None:
        """Merge one worker's accumulated state into this metric."""
        del state
        raise NotImplementedError

    def synchronize(self) -> None:
        """Merge metric state across all initialized PyTorch DDP workers."""
        import torch.distributed as distributed

        if not distributed.is_available() or not distributed.is_initialized():
            return
        if distributed.get_world_size() <= 1:
            return
        try:
            local_state = self.distributed_state()
        except NotImplementedError as error:
            raise RuntimeError(
                f"Metric {self.__class__.__name__} does not implement the "
                "distributed-state contract required by DDP."
            ) from error
        states: list[dict[str, Any] | None] = [None] * distributed.get_world_size()
        distributed.all_gather_object(states, local_state)
        self.reset()
        for state in states:
            if state is not None:
                self.merge_distributed_state(state)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, higher_is_better={self.higher_is_better})"
        )
