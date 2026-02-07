"""
Store locator: URI-based store identification and parsing.

Supported locator schemes:
- file:///path/to/store           → FileStoreConfig
- /path/to/store                  → FileStoreConfig (implicit file://)
- postgresql://user@host:port/db  → PostgresStoreConfig (requires file_root)
- discover                        → auto-detect from nearest ServiceBundle

The locator abstraction allows stores to be passed as strings across
process/cluster boundaries while supporting multiple storage backends.

All locator parsing goes through parse_to_config(), which delegates
to the appropriate StoreConfig subclass registered in ConfigRegistry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from metalab.store.base import Store
    from metalab.store.config import StoreConfig

logger = logging.getLogger(__name__)

# Default root directory for stores when not specified
DEFAULT_STORE_ROOT = "./experiments"


@dataclass
class LocatorInfo:
    """
    Parsed store locator information.

    Attributes:
        scheme: The locator scheme (file, postgresql).
        path: Path component (filesystem path or database name).
        host: Hostname for network stores.
        port: Port number for network stores.
        user: Username for authenticated stores.
        password: Password for authenticated stores.
        params: Additional query parameters.
        raw: The original locator string.
    """

    scheme: str
    path: str
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    params: dict[str, str] = None  # type: ignore[assignment]
    raw: str = ""

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


def _resolve_discover_locator() -> LocatorInfo:
    """Resolve a 'discover' locator by finding the nearest service bundle.

    Looks for a running service bundle (services/bundle.json) by walking
    up from the current directory. If found, uses its store_locator field.

    Returns:
        LocatorInfo from the discovered store locator.

    Raises:
        ValueError: If no service bundle found or it has no store_locator.
    """
    try:
        from metalab.environment.bundle import ServiceBundle
    except ImportError:
        raise ValueError(
            "Store discovery requires the metalab environment module. "
            "Install metalab with environment support."
        )

    bundle = ServiceBundle.find_nearest()
    if bundle is None:
        raise ValueError(
            "No active service bundle found. "
            "Start services with 'metalab atlas up' first, "
            "or specify an explicit store locator."
        )
    if not bundle.store_locator:
        raise ValueError(
            "Service bundle found but has no store_locator. "
            "The bundle may not have a database service configured."
        )
    # Recursively parse the discovered locator (it will be a real URI)
    return parse_locator(bundle.store_locator)


def parse_locator(locator: str) -> LocatorInfo:
    """
    Parse a store locator string into its components.

    Args:
        locator: Store locator URI or path.

    Returns:
        Parsed locator information.

    Examples:
        >>> parse_locator("/path/to/store")
        LocatorInfo(scheme='file', path='/path/to/store', ...)

        >>> parse_locator("file:///path/to/store")
        LocatorInfo(scheme='file', path='/path/to/store', ...)

        >>> parse_locator("postgresql://user@localhost:5432/metalab")
        LocatorInfo(scheme='postgresql', path='/metalab', host='localhost', ...)
    """
    # Handle "discover" special locator — auto-detect from service bundle
    if locator.strip().lower() == "discover":
        return _resolve_discover_locator()

    # Handle plain filesystem paths
    if locator.startswith("/") or locator.startswith("./") or locator.startswith(".."):
        return LocatorInfo(
            scheme="file",
            path=str(Path(locator).resolve()),
            raw=locator,
        )

    # Handle Windows-style absolute paths (C:\... or C:/...)
    if len(locator) >= 2 and locator[1] == ":" and locator[0].isalpha():
        return LocatorInfo(
            scheme="file",
            path=str(Path(locator).resolve()),
            raw=locator,
        )

    # Parse as URI
    parsed = urlparse(locator)

    # Extract query parameters
    params = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            params[key] = values[0] if values else ""

    # Determine path
    path = parsed.path
    if parsed.scheme == "file":
        if parsed.netloc:
            path = parsed.path
        path = str(Path(path).resolve()) if path else ""

    return LocatorInfo(
        scheme=parsed.scheme or "file",
        path=path,
        host=parsed.hostname,
        port=parsed.port,
        user=parsed.username,
        password=parsed.password,
        params=params,
        raw=locator,
    )


def parse_to_config(locator: str, **kwargs: Any) -> "StoreConfig":
    """
    Parse a locator string into a StoreConfig.

    Delegates to the config class registered for the scheme.
    This is the primary way to create configs from URI strings.

    Args:
        locator: Store locator URI or path.
        **kwargs: Additional arguments passed to the config's from_locator().

    Returns:
        A StoreConfig instance.

    Raises:
        ValueError: If the scheme is not supported or required params missing.

    Examples:
        # FileStoreConfig from path
        config = parse_to_config("/path/to/store")

        # FileStoreConfig from URI
        config = parse_to_config("file:///path/to/store")

        # PostgresStoreConfig (requires file_root)
        config = parse_to_config(
            "postgresql://localhost/db",
            file_root="/path/to/files",
        )

        # PostgresStoreConfig with file_root in URI
        config = parse_to_config(
            "postgresql://localhost/db?file_root=/path/to/files"
        )
    """
    from metalab.store.config import ConfigRegistry

    info = parse_locator(locator)

    # Look up config class from registry
    config_class = ConfigRegistry.get(info.scheme)
    if config_class is None:
        raise ValueError(f"Unknown store scheme: {info.scheme}")

    # Each config class knows how to parse its own locator
    return config_class.from_locator(info, **kwargs)


def create_store(locator: str, **kwargs: Any) -> "Store":
    """
    Create a Store instance from a locator string.

    Convenience function that parses the locator to a config and connects.
    Equivalent to: parse_to_config(locator, **kwargs).connect()

    Args:
        locator: Store locator URI or path.
        **kwargs: Additional arguments passed to the config's from_locator().

    Returns:
        A Store instance.

    Raises:
        ValueError: If the scheme is not supported or required params missing.

    Examples:
        # FileStore from path
        store = create_store("/path/to/store")

        # FileStore from URI
        store = create_store("file:///path/to/store")

        # PostgresStore (requires file_root)
        store = create_store(
            "postgresql://localhost/db",
            file_root="/path/to/files",
        )
    """
    return parse_to_config(locator, **kwargs).connect()
