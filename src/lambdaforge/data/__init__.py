"""Task-agnostic dataset wrappers and bounded cache backends."""

from lambdaforge.LazyExports import LazyExports

_CACHE_NAMES = (
    "CacheBackend",
    "CacheIntegrityError",
    "CacheIntegrityMode",
    "CacheNamespaceManifest",
    "CacheRecord",
    "CacheRecordCodec",
    "CacheStats",
    "CacheUsage",
    "DatasetCache",
    "DatasetFingerprint",
    "DatasetSerializer",
    "DiskCacheBackend",
    "MemoryMappedCacheBackend",
    "NumpyDatasetSerializer",
    "PickleDatasetSerializer",
)
_BUILD_MODEL_NAMES = (
    "DatasetBuildPlan",
    "DatasetBuildResult",
    "DatasetRecipeValidationReport",
    "DatasetStagePlan",
)
_ERROR_NAMES = (
    "AmbiguousDatasetVersionError",
    "DatasetResolutionError",
    "InvalidDatasetBuildError",
    "MissingDatasetPlacementError",
    "MissingDatasetRecipeError",
    "MissingManagedEnvironmentError",
    "OfflineClusterError",
    "UnknownDatasetError",
    "UnsafeDatasetOperationError",
)
_RECIPE_CONFIG_NAMES = ("DatasetRecipeConfig", "DatasetRecipeStage")
_INDEX_NAMES = ("DatasetAsset", "DatasetIndex", "DatasetMember")
_DATA_NAMES = (
    "CategoricalFeatureEncoder",
    "ClassificationDatasetProfiler",
    "DataCatalog",
    "DataIdentityProvider",
    "DataIdentityProviderRegistry",
    "DataReplicationResult",
    "DataService",
    "DataTransferProvider",
    "DatasetDeletionPlan",
    "DatasetBuildService",
    "DatasetBuildWorker",
    "DatasetIdIdentityProvider",
    "DatasetIdentity",
    "DatasetLocation",
    "DatasetMaterializationPlan",
    "DatasetPlacement",
    "DatasetProfiler",
    "DatasetPublisher",
    "DatasetRecipe",
    "DatasetRecipeSchemaCatalog",
    "DatasetRecord",
    "DatasetReference",
    "DatasetReferenceResolver",
    "DatasetRegistry",
    "DatasetResolution",
    "DatasetResolver",
    "DatasetService",
    "ExplicitVersionIdentityProvider",
    "FileDataset",
    "ManifestIdentityProvider",
    "NumpyMemmapDataset",
    "RsyncDataTransferProvider",
    "StrictContentHashIdentityProvider",
)

LazyExports.install(
    __name__,
    {
        **{name: (f"lambdaforge.data.cache.{name}", name) for name in _CACHE_NAMES},
        **{name: ("lambdaforge.data.build_models", name) for name in _BUILD_MODEL_NAMES},
        **{name: ("lambdaforge.data.errors", name) for name in _ERROR_NAMES},
        **{name: ("lambdaforge.data.index", name) for name in _INDEX_NAMES},
        **{name: ("lambdaforge.data.recipe_config", name) for name in _RECIPE_CONFIG_NAMES},
        **{name: (f"lambdaforge.data.{name}", name) for name in _DATA_NAMES},
    },
)

__all__ = [
    *_CACHE_NAMES,
    *_BUILD_MODEL_NAMES,
    *_ERROR_NAMES,
    *_INDEX_NAMES,
    *_RECIPE_CONFIG_NAMES,
    *_DATA_NAMES,
]
