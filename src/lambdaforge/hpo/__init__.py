"""Search helpers used by Work study expansion."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.hpo.AdaptiveResources import (
        ActiveResourceEvidence,
        AdmissionDecision,
        GPUPlacementPlanner,
        ResourceDemandModel,
        ResourcePrediction,
        ResourceProfileObservation,
        ResourceTrajectoryAnalyzer,
    )
    from lambdaforge.hpo.AdaptiveSampler import AdaptiveSampler, CandidateObservation
    from lambdaforge.hpo.AdaptiveStatistics import AdaptiveSeedRacer, CandidateEstimate
    from lambdaforge.hpo.BayesianSampler import BayesianSampler
    from lambdaforge.hpo.RandomSearch import RandomSearch
    from lambdaforge.hpo.SobolSearch import SobolSearch
    from lambdaforge.hpo.StudyInsights import StudyInsightAnalyzer
    from lambdaforge.hpo.Trial import Trial

LazyExports.install(
    __name__,
    {
        name: (f"lambdaforge.hpo.{module}", name)
        for module, names in {
            "AdaptiveResources": (
                "ActiveResourceEvidence",
                "AdmissionDecision",
                "GPUPlacementPlanner",
                "ResourceDemandModel",
                "ResourcePrediction",
                "ResourceProfileObservation",
                "ResourceTrajectoryAnalyzer",
            ),
            "AdaptiveSampler": ("AdaptiveSampler", "CandidateObservation"),
            "AdaptiveStatistics": ("AdaptiveSeedRacer", "CandidateEstimate"),
            "BayesianSampler": ("BayesianSampler",),
            "RandomSearch": ("RandomSearch",),
            "SobolSearch": ("SobolSearch",),
            "StudyInsights": ("StudyInsightAnalyzer",),
            "Trial": ("Trial",),
        }.items()
        for name in names
    },
)

__all__ = [
    "AdaptiveSampler",
    "AdaptiveSeedRacer",
    "ActiveResourceEvidence",
    "AdmissionDecision",
    "BayesianSampler",
    "CandidateEstimate",
    "CandidateObservation",
    "GPUPlacementPlanner",
    "RandomSearch",
    "ResourceDemandModel",
    "ResourcePrediction",
    "ResourceProfileObservation",
    "ResourceTrajectoryAnalyzer",
    "SobolSearch",
    "StudyInsightAnalyzer",
    "Trial",
]
