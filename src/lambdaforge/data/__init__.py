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
from lambdaforge.data.DataCatalog import DataCatalog
from lambdaforge.data.DataIdentityProvider import DataIdentityProvider
from lambdaforge.data.DataIdentityProviderRegistry import DataIdentityProviderRegistry
from lambdaforge.data.DataReplicationResult import DataReplicationResult
from lambdaforge.data.DataService import DataService
from lambdaforge.data.DatasetIdentity import DatasetIdentity
from lambdaforge.data.DatasetIdIdentityProvider import DatasetIdIdentityProvider
from lambdaforge.data.DatasetLocation import DatasetLocation
from lambdaforge.data.DatasetReference import DatasetReference
from lambdaforge.data.DatasetReferenceResolver import DatasetReferenceResolver
from lambdaforge.data.DataTransferProvider import DataTransferProvider
from lambdaforge.data.ExplicitVersionIdentityProvider import ExplicitVersionIdentityProvider
from lambdaforge.data.FileDataset import FileDataset
from lambdaforge.data.ManifestIdentityProvider import ManifestIdentityProvider
from lambdaforge.data.NumpyMemmapDataset import NumpyMemmapDataset
from lambdaforge.data.RsyncDataTransferProvider import RsyncDataTransferProvider
from lambdaforge.data.StrictContentHashIdentityProvider import StrictContentHashIdentityProvider

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
    "DataCatalog",
    "DataIdentityProvider",
    "DataIdentityProviderRegistry",
    "DataReplicationResult",
    "DataService",
    "DataTransferProvider",
    "DatasetCache",
    "DatasetFingerprint",
    "DatasetIdIdentityProvider",
    "DatasetIdentity",
    "DatasetLocation",
    "DatasetReference",
    "DatasetReferenceResolver",
    "DatasetSerializer",
    "DiskCacheBackend",
    "ExplicitVersionIdentityProvider",
    "FileDataset",
    "MemoryMappedCacheBackend",
    "ManifestIdentityProvider",
    "NumpyDatasetSerializer",
    "NumpyMemmapDataset",
    "PickleDatasetSerializer",
    "RsyncDataTransferProvider",
    "StrictContentHashIdentityProvider",
]
