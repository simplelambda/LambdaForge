"""Mandatory real-provider tests for the optional BoTorch adaptive sampler."""

from __future__ import annotations

from lambdaforge.hpo.AdaptiveSampler import CandidateObservation
from lambdaforge.hpo.BayesianSampler import BayesianSampler
from lambdaforge.hpo.SurvivalModel import SurvivalAcquisitionPolicy, SurvivalEstimate


def test_botorch_provider_is_installed_in_the_provider_job() -> None:
    """This assertion intentionally fails instead of skipping a misconfigured CI job."""
    assert BayesianSampler.available() is True


def test_real_numerical_multifidelity_gp_uses_noise_pending_and_target_fidelity() -> None:
    candidates = {trial: {"x": float(trial - 1) / 7.0} for trial in range(1, 9)}
    observations = tuple(
        CandidateObservation(
            trial=trial,
            value=(float(trial) / 10.0) * fidelity,
            standard_error=0.01 + trial / 1000,
            fidelity=fidelity,
        )
        for trial in range(1, 6)
        for fidelity in (0.25, 1.0)
    )
    sampler = BayesianSampler(candidates, mode="max")

    proposed = sampler.propose(
        observations,
        selected=(1, 2, 3, 4, 5),
        pending_observations=((5, 0.5),),
        count=1,
    )

    assert len(proposed) == 1
    assert proposed[0] in {6, 7, 8}
    assert sampler.last_diagnostics["backend"] == "SingleTaskMultiFidelityGP"
    assert sampler.last_diagnostics["target_fidelity"] == 1.0
    assert sampler.last_diagnostics["pending_observations"] == 1
    assert sampler.last_diagnostics["ranked_candidates"]
    assert all(
        value["prediction_standard_deviation"] >= 0
        for value in sampler.last_diagnostics["ranked_candidates"]
    )


def test_real_mixed_gp_preserves_categorical_conditional_and_survival_evidence() -> None:
    candidates = {
        1: {"model": "linear", "rate": 0.01},
        2: {"model": "linear", "rate": 0.03},
        3: {"model": "tree", "depth": 2, "rate": 0.01},
        4: {"model": "tree", "depth": 4, "rate": 0.02},
        5: {"model": "tree", "depth": 6, "rate": 0.03},
        6: {"model": "linear", "rate": 0.05},
        7: {"model": "tree", "depth": 3, "rate": 0.04},
        8: {"model": "linear", "rate": 0.07},
        9: {"model": "tree", "depth": 7, "rate": 0.06},
    }
    observations = tuple(
        CandidateObservation(
            trial=trial,
            value=0.2 + trial / 20,
            standard_error=0.02,
            fidelity=0.5 if trial % 2 else 1.0,
        )
        for trial in range(1, 7)
    )
    survival = {trial: SurvivalEstimate(trial, 0.6, 0.3, 0.9, 1.0) for trial in (7, 8, 9)}
    sampler = BayesianSampler(candidates, mode="max")

    proposed = sampler.propose(
        observations,
        selected=(1, 2, 3, 4, 5, 6),
        pending_observations=((6, 0.75),),
        survival=survival,
        count=1,
    )

    assert len(proposed) == 1
    assert proposed[0] in {7, 8, 9}
    assert sampler.last_diagnostics["backend"] == "MixedSingleTaskGP"
    assert sampler.last_diagnostics["survival_adjustment"] == (
        SurvivalAcquisitionPolicy().to_dict()
    )
    assert sampler.last_diagnostics["conditional_dependence"].startswith("encoded jointly")
