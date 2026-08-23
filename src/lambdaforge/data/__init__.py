"""Immutable datasets, placements, profiling and bounded cache utilities."""

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
_NAMES = (
    "CategoricalFeatureEncoder",
    "ClassificationDatasetProfiler",
    "DataCatalog",
    "DataIdentityProvider",
    "DataIdentityProviderRegistry",
    "DatasetArtifact",
    "DatasetDeletionPlan",
    "DatasetIdIdentityProvider",
    "DatasetIdentity",
    "DatasetLocation",
    "DatasetMaterializationPlan",
    "DatasetOperations",
    "DatasetPlacement",
    "DatasetProfiler",
    "DatasetPublisher",
    "DatasetRecord",
    "DatasetReference",
    "DatasetRegistry",
    "DatasetResolver",
    "DatasetService",
    "ExplicitVersionIdentityProvider",
    "FileDataset",
    "ManifestIdentityProvider",
    "NumpyMemmapDataset",
    "StoredArtifact",
    "StrictContentHashIdentityProvider",
)

LazyExports.install(
    __name__,
    {
        **{name: (f"lambdaforge.data.cache.{name}", name) for name in _CACHE_NAMES},
        **{name: (f"lambdaforge.data.{name}", name) for name in _NAMES},
        "DatasetAsset": ("lambdaforge.data.index", "DatasetAsset"),
        "DatasetIndex": ("lambdaforge.data.index", "DatasetIndex"),
        "DatasetMember": ("lambdaforge.data.index", "DatasetMember"),
        "DatasetPlacementResolution": (
            "lambdaforge.data.DatasetResolution",
            "DatasetPlacementResolution",
        ),
        "DatasetPlacementState": (
            "lambdaforge.data.DatasetResolution",
            "DatasetPlacementState",
        ),
        "AmbiguousDatasetVersionError": (
            "lambdaforge.data.errors",
            "AmbiguousDatasetVersionError",
        ),
    },
)

__all__ = [
    *_CACHE_NAMES,
    *_NAMES,
    "DatasetAsset",
    "DatasetIndex",
    "DatasetMember",
    "DatasetPlacementResolution",
    "DatasetPlacementState",
    "AmbiguousDatasetVersionError",
]
