"""Public safe artifact inspection, validation and visualization toolkit."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.artifacts.ArtifactInspection import ArtifactInspection
    from lambdaforge.artifacts.ArtifactInspector import ArtifactInspector
    from lambdaforge.artifacts.ArtifactPluginRegistry import ArtifactPluginRegistry
    from lambdaforge.artifacts.ArtifactSchema import ArtifactSchema
    from lambdaforge.artifacts.ArtifactService import ArtifactService
    from lambdaforge.artifacts.ArtifactValidationResult import ArtifactValidationResult
    from lambdaforge.artifacts.ArtifactValidator import ArtifactValidator
    from lambdaforge.artifacts.ArtifactVisualizer import ArtifactVisualizer
    from lambdaforge.artifacts.GenericArtifactVisualizer import GenericArtifactVisualizer
    from lambdaforge.artifacts.NumpyArtifactInspector import NumpyArtifactInspector
    from lambdaforge.artifacts.NumpyArtifactValidator import NumpyArtifactValidator
    from lambdaforge.artifacts.RemoteArtifactService import RemoteArtifactService
    from lambdaforge.artifacts.TabularArtifactInspector import TabularArtifactInspector

_NAMES = (
    "ArtifactInspection",
    "ArtifactInspector",
    "ArtifactPluginRegistry",
    "ArtifactSchema",
    "ArtifactService",
    "ArtifactValidationResult",
    "ArtifactValidator",
    "ArtifactVisualizer",
    "GenericArtifactVisualizer",
    "NumpyArtifactInspector",
    "NumpyArtifactValidator",
    "RemoteArtifactService",
    "TabularArtifactInspector",
)
LazyExports.install(__name__, {name: (f"lambdaforge.artifacts.{name}", name) for name in _NAMES})

__all__ = [
    "ArtifactInspection",
    "ArtifactInspector",
    "ArtifactPluginRegistry",
    "ArtifactSchema",
    "ArtifactService",
    "ArtifactValidationResult",
    "ArtifactValidator",
    "ArtifactVisualizer",
    "GenericArtifactVisualizer",
    "NumpyArtifactInspector",
    "NumpyArtifactValidator",
    "RemoteArtifactService",
    "TabularArtifactInspector",
]
