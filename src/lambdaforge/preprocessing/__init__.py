"""Public composable preprocessing and dataset-artifact contracts."""

from typing import TYPE_CHECKING

from lambdaforge.LazyExports import LazyExports

if TYPE_CHECKING:
    from lambdaforge.preprocessing.CallableTransform import CallableTransform
    from lambdaforge.preprocessing.DatasetArtifact import DatasetArtifact
    from lambdaforge.preprocessing.FileTreeSource import FileTreeSource
    from lambdaforge.preprocessing.JsonDirectorySink import JsonDirectorySink
    from lambdaforge.preprocessing.JsonLinesSource import JsonLinesSource
    from lambdaforge.preprocessing.PreprocessingErrorPolicy import PreprocessingErrorPolicy
    from lambdaforge.preprocessing.PreprocessingManifest import PreprocessingManifest
    from lambdaforge.preprocessing.PreprocessingRecord import PreprocessingRecord
    from lambdaforge.preprocessing.PreprocessingSink import PreprocessingSink
    from lambdaforge.preprocessing.PreprocessingSource import PreprocessingSource
    from lambdaforge.preprocessing.PreprocessingTask import PreprocessingTask
    from lambdaforge.preprocessing.PreprocessingTransform import PreprocessingTransform
    from lambdaforge.preprocessing.PreprocessingWorkload import PreprocessingWorkload

LazyExports.install(
    __name__,
    {
        "CallableTransform": (
            "lambdaforge.preprocessing.CallableTransform",
            "CallableTransform",
        ),
        "DatasetArtifact": (
            "lambdaforge.preprocessing.DatasetArtifact",
            "DatasetArtifact",
        ),
        "FileTreeSource": (
            "lambdaforge.preprocessing.FileTreeSource",
            "FileTreeSource",
        ),
        "JsonDirectorySink": (
            "lambdaforge.preprocessing.JsonDirectorySink",
            "JsonDirectorySink",
        ),
        "JsonLinesSource": (
            "lambdaforge.preprocessing.JsonLinesSource",
            "JsonLinesSource",
        ),
        "PreprocessingErrorPolicy": (
            "lambdaforge.preprocessing.PreprocessingErrorPolicy",
            "PreprocessingErrorPolicy",
        ),
        "PreprocessingManifest": (
            "lambdaforge.preprocessing.PreprocessingManifest",
            "PreprocessingManifest",
        ),
        "PreprocessingRecord": (
            "lambdaforge.preprocessing.PreprocessingRecord",
            "PreprocessingRecord",
        ),
        "PreprocessingSink": (
            "lambdaforge.preprocessing.PreprocessingSink",
            "PreprocessingSink",
        ),
        "PreprocessingSource": (
            "lambdaforge.preprocessing.PreprocessingSource",
            "PreprocessingSource",
        ),
        "PreprocessingTask": (
            "lambdaforge.preprocessing.PreprocessingTask",
            "PreprocessingTask",
        ),
        "PreprocessingTransform": (
            "lambdaforge.preprocessing.PreprocessingTransform",
            "PreprocessingTransform",
        ),
        "PreprocessingWorkload": (
            "lambdaforge.preprocessing.PreprocessingWorkload",
            "PreprocessingWorkload",
        ),
    },
)

__all__ = [
    "CallableTransform",
    "DatasetArtifact",
    "FileTreeSource",
    "JsonDirectorySink",
    "JsonLinesSource",
    "PreprocessingErrorPolicy",
    "PreprocessingManifest",
    "PreprocessingRecord",
    "PreprocessingSink",
    "PreprocessingSource",
    "PreprocessingTask",
    "PreprocessingTransform",
    "PreprocessingWorkload",
]
