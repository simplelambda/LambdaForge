"""Backward-compatible name for the feature-aware memory model."""

from lambdaforge.hpo.FeatureAwareMemoryModel import FeatureAwareMemoryModel


class EmpiricalMemoryModel(FeatureAwareMemoryModel):
    """Compatibility alias; new code should use ``FeatureAwareMemoryModel``."""
