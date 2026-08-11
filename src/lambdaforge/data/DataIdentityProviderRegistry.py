"""Registry for built-in dataset identity strategies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DatasetIdIdentityProvider import DatasetIdIdentityProvider
from lambdaforge.data.ExplicitVersionIdentityProvider import ExplicitVersionIdentityProvider
from lambdaforge.data.ManifestIdentityProvider import ManifestIdentityProvider
from lambdaforge.data.StrictContentHashIdentityProvider import StrictContentHashIdentityProvider


class DataIdentityProviderRegistry:
    """Resolve documented provider names without importing project code implicitly."""

    def create(self, descriptor: Mapping[str, Any]) -> DataIdentityProvider:
        """Construct the selected built-in provider."""
        strategy = str(descriptor.get("strategy", "strict"))
        providers: dict[str, type[DataIdentityProvider]] = {
            "strict": StrictContentHashIdentityProvider,
            "manifest": ManifestIdentityProvider,
            "dataset_id": DatasetIdIdentityProvider,
            "version": ExplicitVersionIdentityProvider,
        }
        try:
            return providers[strategy]()
        except KeyError as error:
            raise ValueError(
                f"Unknown data identity strategy {strategy!r}; choose {sorted(providers)}."
            ) from error
