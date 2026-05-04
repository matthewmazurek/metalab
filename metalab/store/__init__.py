"""Filesystem-only storage for records, events, logs, and artifacts."""

from metalab.store.base import Store
from metalab.store.capabilities import (
    SupportsArtifactOpen,
    SupportsExperimentManifests,
    SupportsLogListing,
    SupportsLogPath,
    SupportsStructuredResults,
    SupportsWorkingDirectory,
)
from metalab.store.config import ConfigRegistry, StoreConfig
from metalab.store.events import FileEventSink, PersistentEvent
from metalab.store.file import FileStore, FileStoreConfig
from metalab.store.layout import FileStoreLayout, safe_experiment_id
from metalab.store.locator import (
    DEFAULT_STORE_ROOT,
    LocatorInfo,
    create_store,
    parse_locator,
    parse_to_config,
)

__all__ = [
    "Store",
    "StoreConfig",
    "ConfigRegistry",
    "FileStoreConfig",
    "FileStore",
    "FileStoreLayout",
    "FileEventSink",
    "PersistentEvent",
    "SupportsWorkingDirectory",
    "SupportsExperimentManifests",
    "SupportsArtifactOpen",
    "SupportsLogPath",
    "SupportsStructuredResults",
    "SupportsLogListing",
    "DEFAULT_STORE_ROOT",
    "LocatorInfo",
    "create_store",
    "parse_locator",
    "parse_to_config",
    "safe_experiment_id",
]
