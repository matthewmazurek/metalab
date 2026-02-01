"""
Store module: Backend-agnostic persistence for records and artifacts.

Provides:
- Store: Protocol for storage backends
- FileStore: Filesystem-based storage with defined layout
- StoreFactory: Factory for creating stores from locator URIs
- create_store: Convenience function for creating stores
- parse_locator: Parse store locator URIs
- to_locator: Convert store instances to locator strings
- export_store, FallbackStore: Transfer and fallback utilities

Capability protocols (for optional store features):
- SupportsWorkingDirectory: stores with local filesystem roots
- SupportsExperimentManifests: stores that persist experiment manifests
- SupportsArtifactOpen: stores that can open artifacts for reading
- SupportsLogPath: stores that provide log file paths for streaming
- SupportsStructuredResults: stores that support inline structured data
- SupportsLogListing: stores that can list and retrieve logs
"""

from metalab.store.base import Store
from metalab.store.capabilities import (
    SupportsArtifactOpen,
    SupportsExperimentManifests,
    SupportsLogListing,
    SupportsLogPath,
    SupportsStructuredResults,
    SupportsWorkingDirectory,
)
from metalab.store.file import FileStore
from metalab.store.locator import (
    LocatorInfo,
    StoreFactory,
    create_store,
    parse_locator,
    to_locator,
)
from metalab.store.transfer import (
    FallbackStore,
    export_store,
    export_to_filestore,
    import_from_filestore,
)

__all__ = [
    # Base protocol
    "Store",
    # Capability protocols
    "SupportsWorkingDirectory",
    "SupportsExperimentManifests",
    "SupportsArtifactOpen",
    "SupportsLogPath",
    "SupportsStructuredResults",
    "SupportsLogListing",
    # Implementations
    "FileStore",
    # Factory/locator utilities
    "LocatorInfo",
    "StoreFactory",
    "create_store",
    "parse_locator",
    "to_locator",
    # Transfer utilities
    "export_store",
    "export_to_filestore",
    "import_from_filestore",
    "FallbackStore",
]
