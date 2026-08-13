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
    "DatasetIdIdentityProvider",
    "DatasetIdentity",
    "DatasetLocation",
    "DatasetMaterializationPlan",
    "DatasetPlacement",
    "DatasetProfiler",
    "DatasetRecord",
    "DatasetReference",
    "DatasetReferenceResolver",
    "DatasetRegistry",
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
        **{name: (f"lambdaforge.data.{name}", name) for name in _DATA_NAMES},
    },
)

__all__ = [*_CACHE_NAMES, *_DATA_NAMES]
