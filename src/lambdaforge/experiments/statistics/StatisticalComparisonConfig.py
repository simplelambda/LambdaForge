"""Validated YAML configuration for cross-seed statistical comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from lambdaforge.experiments.statistics.intervals.ConfidenceIntervalMethod import (
    ConfidenceIntervalMethod,
)
from lambdaforge.experiments.statistics.paired.PairedAlternative import PairedAlternative
from lambdaforge.experiments.statistics.paired.PairedTestMethod import PairedTestMethod
from lambdaforge.experiments.statistics.paired.WilcoxonCalculation import (
    WilcoxonCalculation,
)
from lambdaforge.experiments.statistics.paired.WilcoxonZeroMethod import WilcoxonZeroMethod


@dataclass(frozen=True, slots=True)
class StatisticalComparisonConfig:
    """Materialize comparison defaults and reject unsafe direct-call values."""

    alpha: float = 0.05
    target_power: float = 0.80
    min_pairs_for_verdict: int = 3
    confidence_interval_method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.NORMAL
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 0
    bootstrap_batch_size: int = 1_024
    bootstrap_max_batch_elements: int = 1_000_000
    paired_test_method: PairedTestMethod = PairedTestMethod.SIGN
    paired_alternative: PairedAlternative = PairedAlternative.OBSERVED_DIRECTION
    wilcoxon_calculation: WilcoxonCalculation = WilcoxonCalculation.AUTO
    wilcoxon_zero_method: WilcoxonZeroMethod = WilcoxonZeroMethod.WILCOX
    wilcoxon_continuity_correction: bool = False
    wilcoxon_exact_max_pairs: int = 50
    zero_tolerance: float = 1e-12
    round_decimals: int | None = 12

    AGGREGATION_KEYS: ClassVar[frozenset[str]] = frozenset({"comparisons"})
    COMPARISON_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "alpha",
            "target_power",
            "min_pairs_for_verdict",
            "confidence_interval",
            "paired_test",
        }
    )
    INTERVAL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "method",
            "confidence_level",
            "resamples",
            "seed",
            "batch_size",
            "max_batch_elements",
        }
    )
    PAIRED_TEST_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "method",
            "alternative",
            "calculation",
            "zero_method",
            "continuity_correction",
            "exact_max_pairs",
            "zero_tolerance",
            "round_decimals",
        }
    )

    def __post_init__(self) -> None:
        """Normalize direct construction and apply the same safety bounds as YAML."""
        object.__setattr__(self, "alpha", self._number(self.alpha, "alpha"))
        object.__setattr__(
            self,
            "target_power",
            self._number(self.target_power, "target_power"),
        )
        object.__setattr__(
            self,
            "min_pairs_for_verdict",
            self._integer(self.min_pairs_for_verdict, "min_pairs_for_verdict"),
        )
        object.__setattr__(
            self,
            "confidence_interval_method",
            ConfidenceIntervalMethod(self.confidence_interval_method),
        )
        object.__setattr__(
            self,
            "confidence_level",
            self._number(self.confidence_level, "confidence_level"),
        )
        object.__setattr__(
            self,
            "bootstrap_resamples",
            self._integer(self.bootstrap_resamples, "bootstrap_resamples"),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            self._integer(self.bootstrap_seed, "bootstrap_seed"),
        )
        object.__setattr__(
            self,
            "bootstrap_batch_size",
            self._integer(self.bootstrap_batch_size, "bootstrap_batch_size"),
        )
        object.__setattr__(
            self,
            "bootstrap_max_batch_elements",
            self._integer(
                self.bootstrap_max_batch_elements,
                "bootstrap_max_batch_elements",
            ),
        )
        object.__setattr__(
            self,
            "paired_test_method",
            PairedTestMethod(self.paired_test_method),
        )
        object.__setattr__(
            self,
            "paired_alternative",
            PairedAlternative(self.paired_alternative),
        )
        object.__setattr__(
            self,
            "wilcoxon_calculation",
            WilcoxonCalculation(self.wilcoxon_calculation),
        )
        object.__setattr__(
            self,
            "wilcoxon_zero_method",
            WilcoxonZeroMethod(self.wilcoxon_zero_method),
        )
        object.__setattr__(
            self,
            "wilcoxon_continuity_correction",
            self._boolean(
                self.wilcoxon_continuity_correction,
                "wilcoxon_continuity_correction",
            ),
        )
        object.__setattr__(
            self,
            "wilcoxon_exact_max_pairs",
            self._integer(self.wilcoxon_exact_max_pairs, "wilcoxon_exact_max_pairs"),
        )
        object.__setattr__(
            self,
            "zero_tolerance",
            self._number(self.zero_tolerance, "zero_tolerance"),
        )
        object.__setattr__(
            self,
            "round_decimals",
            self._optional_integer(self.round_decimals, "round_decimals"),
        )
        self._validate()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> StatisticalComparisonConfig:
        """Parse the optional `aggregation.comparisons` YAML block."""
        aggregation = cls._mapping(config.get("aggregation", {}), "aggregation")
        cls._reject_unknown(aggregation, cls.AGGREGATION_KEYS, "aggregation")
        comparisons = cls._mapping(
            aggregation.get("comparisons", {}),
            "aggregation.comparisons",
        )
        cls._reject_unknown(comparisons, cls.COMPARISON_KEYS, "aggregation.comparisons")
        interval = cls._mapping(
            comparisons.get("confidence_interval", {}),
            "aggregation.comparisons.confidence_interval",
        )
        cls._reject_unknown(
            interval,
            cls.INTERVAL_KEYS,
            "aggregation.comparisons.confidence_interval",
        )
        paired = cls._mapping(
            comparisons.get("paired_test", {}),
            "aggregation.comparisons.paired_test",
        )
        cls._reject_unknown(
            paired,
            cls.PAIRED_TEST_KEYS,
            "aggregation.comparisons.paired_test",
        )

        instance = cls(
            alpha=cls._number(comparisons.get("alpha", 0.05), "alpha"),
            target_power=cls._number(
                comparisons.get("target_power", 0.80),
                "target_power",
            ),
            min_pairs_for_verdict=cls._integer(
                comparisons.get("min_pairs_for_verdict", 3),
                "min_pairs_for_verdict",
            ),
            confidence_interval_method=cls._enum(
                ConfidenceIntervalMethod,
                interval.get("method", ConfidenceIntervalMethod.NORMAL.value),
                "confidence_interval.method",
            ),
            confidence_level=cls._number(
                interval.get("confidence_level", 0.95),
                "confidence_interval.confidence_level",
            ),
            bootstrap_resamples=cls._integer(
                interval.get("resamples", 10_000),
                "confidence_interval.resamples",
            ),
            bootstrap_seed=cls._integer(
                interval.get("seed", 0),
                "confidence_interval.seed",
            ),
            bootstrap_batch_size=cls._integer(
                interval.get("batch_size", 1_024),
                "confidence_interval.batch_size",
            ),
            bootstrap_max_batch_elements=cls._integer(
                interval.get("max_batch_elements", 1_000_000),
                "confidence_interval.max_batch_elements",
            ),
            paired_test_method=cls._enum(
                PairedTestMethod,
                paired.get("method", PairedTestMethod.SIGN.value),
                "paired_test.method",
            ),
            paired_alternative=cls._enum(
                PairedAlternative,
                paired.get("alternative", PairedAlternative.OBSERVED_DIRECTION.value),
                "paired_test.alternative",
            ),
            wilcoxon_calculation=cls._enum(
                WilcoxonCalculation,
                paired.get("calculation", WilcoxonCalculation.AUTO.value),
                "paired_test.calculation",
            ),
            wilcoxon_zero_method=cls._enum(
                WilcoxonZeroMethod,
                paired.get("zero_method", WilcoxonZeroMethod.WILCOX.value),
                "paired_test.zero_method",
            ),
            wilcoxon_continuity_correction=cls._boolean(
                paired.get("continuity_correction", False),
                "paired_test.continuity_correction",
            ),
            wilcoxon_exact_max_pairs=cls._integer(
                paired.get("exact_max_pairs", 50),
                "paired_test.exact_max_pairs",
            ),
            zero_tolerance=cls._number(
                paired.get("zero_tolerance", 1e-12),
                "paired_test.zero_tolerance",
            ),
            round_decimals=cls._optional_integer(
                paired.get("round_decimals", 12),
                "paired_test.round_decimals",
            ),
        )
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Return the fully materialized, JSON-compatible protocol."""
        return {
            "alpha": self.alpha,
            "target_power": self.target_power,
            "min_pairs_for_verdict": self.min_pairs_for_verdict,
            "confidence_interval": {
                "method": self.confidence_interval_method.value,
                "confidence_level": self.confidence_level,
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "batch_size": self.bootstrap_batch_size,
                "max_batch_elements": self.bootstrap_max_batch_elements,
            },
            "paired_test": {
                "method": self.paired_test_method.value,
                "alternative": self.paired_alternative.value,
                "calculation": self.wilcoxon_calculation.value,
                "zero_method": self.wilcoxon_zero_method.value,
                "continuity_correction": self.wilcoxon_continuity_correction,
                "exact_max_pairs": self.wilcoxon_exact_max_pairs,
                "zero_tolerance": self.zero_tolerance,
                "round_decimals": self.round_decimals,
            },
        }

    def _validate(self) -> None:
        for name, value in (
            ("alpha", self.alpha),
            ("target_power", self.target_power),
            ("confidence_level", self.confidence_level),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"aggregation.comparisons.{name} must be between 0 and 1.")
        if self.min_pairs_for_verdict < 1:
            raise ValueError("min_pairs_for_verdict must be at least 1.")
        if not 1 <= self.bootstrap_resamples <= 10_000_000:
            raise ValueError("confidence_interval.resamples must be between 1 and 10000000.")
        if self.bootstrap_seed < 0:
            raise ValueError("confidence_interval.seed must be non-negative.")
        if not 1 <= self.bootstrap_batch_size <= 1_000_000:
            raise ValueError("confidence_interval.batch_size must be between 1 and 1000000.")
        if not 1 <= self.bootstrap_max_batch_elements <= 100_000_000:
            raise ValueError(
                "confidence_interval.max_batch_elements must be between 1 and 100000000."
            )
        if not 0 <= self.wilcoxon_exact_max_pairs <= 200:
            raise ValueError("paired_test.exact_max_pairs must be between 0 and 200.")
        if not math.isfinite(self.zero_tolerance) or self.zero_tolerance < 0.0:
            raise ValueError("paired_test.zero_tolerance must be finite and non-negative.")
        if self.round_decimals is not None and not 0 <= self.round_decimals <= 15:
            raise ValueError("paired_test.round_decimals must be null or between 0 and 15.")

    @staticmethod
    def _mapping(value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping.")
        return dict(value)

    @staticmethod
    def _reject_unknown(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"{path} contains unknown keys: {unknown}.")

    @staticmethod
    def _number(value: Any, path: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path} must be a number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite.")
        return number

    @staticmethod
    def _integer(value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer.")
        return int(value)

    @staticmethod
    def _optional_integer(value: Any, path: str) -> int | None:
        if value is None:
            return None
        return StatisticalComparisonConfig._integer(value, path)

    @staticmethod
    def _boolean(value: Any, path: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean.")
        return value

    @staticmethod
    def _enum(enum_type: Any, value: Any, path: str) -> Any:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a string.")
        try:
            return enum_type(value)
        except ValueError as exc:
            options = ", ".join(member.value for member in enum_type)
            raise ValueError(f"{path} must be one of: {options}.") from exc
