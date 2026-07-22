"""Deterministic ordinal preprocessing for heterogeneous categorical columns."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from typing import Any

import torch


class CategoricalFeatureEncoder:
    """Fit stable zero-based categorical vocabularies with one unknown bucket.

    Index zero is always reserved for missing or unseen values. Known values start
    at one, so :attr:`cardinalities` can be passed directly to LambdaForge tabular
    models. Fitting is explicit and deterministic; inference never grows state.
    """

    def __init__(self) -> None:
        self._vocabularies: tuple[dict[Hashable, int], ...] = ()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Report whether category vocabularies have been learned."""
        return self._fitted

    @property
    def cardinalities(self) -> tuple[int, ...]:
        """Return per-column sizes including the unknown/missing bucket."""
        self._require_fitted()
        return tuple(len(vocabulary) + 1 for vocabulary in self._vocabularies)

    def fit(self, rows: Sequence[Sequence[Any]]) -> CategoricalFeatureEncoder:
        """Replace state with deterministic vocabularies learned from ``rows``."""
        width = self._width(rows)
        columns: list[set[Hashable]] = [set() for _ in range(width)]
        for row in rows:
            if len(row) != width:
                raise ValueError("Every categorical row must have the same number of columns.")
            for index, value in enumerate(row):
                if self._is_missing(value):
                    continue
                if not isinstance(value, Hashable):
                    raise TypeError("Categorical values must be hashable.")
                columns[index].add(value)
        self._vocabularies = tuple(
            {
                value: index
                for index, value in enumerate(
                    sorted(column, key=lambda item: (type(item).__qualname__, repr(item))),
                    start=1,
                )
            }
            for column in columns
        )
        self._fitted = True
        return self

    def transform(self, rows: Sequence[Sequence[Any]]) -> torch.Tensor:
        """Encode rows as ``torch.long`` without mutating fitted vocabularies."""
        self._require_fitted()
        width = len(self._vocabularies)
        encoded: list[list[int]] = []
        for row in rows:
            if len(row) != width:
                raise ValueError("Categorical row width differs from fitted state.")
            encoded.append(
                [
                    0 if self._is_missing(value) else vocabulary.get(value, 0)
                    for value, vocabulary in zip(row, self._vocabularies, strict=True)
                ]
            )
        return torch.tensor(encoded, dtype=torch.long).reshape(len(rows), width)

    def fit_transform(self, rows: Sequence[Sequence[Any]]) -> torch.Tensor:
        """Fit and encode the same training rows."""
        return self.fit(rows).transform(rows)

    def state_dict(self) -> dict[str, Any]:
        """Return explicit serializable state for experiment artifacts."""
        self._require_fitted()
        return {
            "version": 1,
            "categories": [
                [value for value, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
                for vocabulary in self._vocabularies
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore vocabularies produced by :meth:`state_dict`."""
        if state.get("version") != 1 or not isinstance(state.get("categories"), list):
            raise ValueError("Unsupported categorical encoder state.")
        vocabularies: list[dict[Hashable, int]] = []
        for categories in state["categories"]:
            if not isinstance(categories, list):
                raise TypeError("Each categorical vocabulary must be a list.")
            if any(not isinstance(value, Hashable) for value in categories):
                raise TypeError("Categorical values must be hashable.")
            vocabulary = {value: index for index, value in enumerate(categories, start=1)}
            if len(vocabulary) != len(categories):
                raise ValueError("Categorical vocabularies cannot contain duplicates.")
            vocabularies.append(vocabulary)
        self._vocabularies = tuple(vocabularies)
        self._fitted = True

    @staticmethod
    def _width(rows: Sequence[Sequence[Any]]) -> int:
        if not rows:
            raise ValueError("At least one categorical row is required for fitting.")
        width = len(rows[0])
        if width < 1:
            raise ValueError("Categorical rows must contain at least one column.")
        return width

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CategoricalFeatureEncoder must be fitted before use.")
