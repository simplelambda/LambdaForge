"""Cluster-operational policy remains explicit and never changes scientific YAML."""

import json

import pytest

from lambdaforge.controlplane.ClusterProfile import ClusterProfile
from lambdaforge.runtime.GPUObservation import GPUObservationCache
from lambdaforge.runtime.ResourceSafetyPolicy import ResourceSafetyPolicy


@pytest.mark.parametrize("profile", ["balanced", "throughput", "shared-host", "conservative"])
def test_native_cluster_policy_round_trip(profile):
    value = ClusterProfile.from_mapping("test", {"resource_safety": profile})
    restored = ClusterProfile.from_mapping("test", value.to_dict())
    assert restored.resource_safety == value.resource_safety
    assert restored.resource_safety.profile == profile


@pytest.mark.parametrize(
    "value",
    [
        {"wat": 1},
        {"priority": 0},
        {"priority": True},
        {"profile": "single-run"},
        {"probe_timeout": 60},
        {"admission_ttl": 100},
        {"waiter_ttl_seconds": 1},
        {"startup_seconds": float("nan")},
    ],
)
def test_invalid_policy_rejected(value):
    with pytest.raises(ValueError):
        ResourceSafetyPolicy.from_value(value)


def test_default_keeps_observation_intervals_and_packing_ceilings_untouched():
    value = ResourceSafetyPolicy()
    assert value.profile == "balanced" and value.admission_ttl == 0.5
    assert not {"runs_per_gpu", "max_parallel", "seeds", "objective"}.intersection(value.to_dict())


def test_explicit_inherited_policy_reaches_shared_observer(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "LAMBDAFORGE_RESOURCE_SAFETY", json.dumps({"profile": "shared-host", "probe_timeout": 2.5})
    )
    cache = GPUObservationCache(tmp_path / "cache", probe=lambda: {})
    assert cache.policy.timeout == 2.5 and cache.policy.telemetry_ttl == 10
