"""Task-agnostic dataset wrappers and bounded cache backends."""

from lambdaforge.data.cache import (
    CacheBackend,
    CacheIntegrityError,
    CacheIntegrityMode,
    CacheNamespaceManifest,
    CacheRecord,
    CacheRecordCodec,
    CacheStats,
    CacheUsage,
    DatasetCache,
    DatasetFingerprint,
    DatasetSerializer,
    DiskCacheBackend,
    MemoryMappedCacheBackend,
    NumpyDatasetSerializer,
    PickleDatasetSerializer,
)
from lambdaforge.data.CategoricalFeatureEncoder import CategoricalFeatureEncoder
from lambdaforge.data.FileDataset import FileDataset
from lambdaforge.data.NumpyMemmapDataset import NumpyMemmapDataset

__all__ = [
    "CacheBackend",
    "CacheIntegrityError",
    "CacheIntegrityMode",
    "CacheNamespaceManifest",
    "CacheRecord",
    "CacheRecordCodec",
    "CacheStats",
    "CacheUsage",
    "CategoricalFeatureEncoder",
    "DatasetCache",
    "DatasetFingerprint",
    "DatasetSerializer",
    "DiskCacheBackend",
    "FileDataset",
    "MemoryMappedCacheBackend",
    "NumpyDatasetSerializer",
    "NumpyMemmapDataset",
    "PickleDatasetSerializer",
]
