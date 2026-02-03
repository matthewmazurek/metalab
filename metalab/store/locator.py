"""
Store locator: URI-based store identification and factory.

Supported locator schemes:
- file:///path/to/store           → FileStore
- /path/to/store                  → FileStore (implicit file://)
- postgresql://user@host:port/db  → PostgresStore (when implemented)
- auto://?primary=...&fallback=...→ Fallback chain

The locator abstraction allows stores to be passed as strings across
process/cluster boundaries while supporting multiple storage backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from metalab.store.base import Store

logger = logging.getLogger(__name__)


@dataclass
class LocatorInfo:
    """
    Parsed store locator information.

    Attributes:
        scheme: The locator scheme (file, postgresql, auto).
        path: Path component (filesystem path or database name).
        host: Hostname for network stores.
        port: Port number for network stores.
        user: Username for authenticated stores.
        password: Password for authenticated stores (use with care).
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
        LocatorInfo(scheme='postgresql', path='/metalab', host='localhost', port=5432, user='user', ...)
    """
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
        # Handle file:///path and file://host/path
        if parsed.netloc:
            # file://host/path → /path (ignore host for local files)
            path = parsed.path
        # Ensure absolute path
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


def to_locator(store: "Store") -> str:
    """
    Convert a Store instance to its locator string.

    This enables serialization of store references for cross-process
    communication (e.g., SLURM payloads).

    Args:
        store: A Store instance.

    Returns:
        Locator string that can recreate the store.

    Raises:
        ValueError: If the store type is not supported.
    """
    # Both FileStore and PostgresStore expose a `locator` property
    if hasattr(store, "locator"):
        return store.locator  # type: ignore[attr-defined]

    raise ValueError(f"Cannot determine locator for store type: {type(store).__name__}")


class StoreFactory:
    """
    Factory for creating Store instances from locator strings.

    Supports pluggable backends via registration. The default backends are:
    - file:// → FileStore
    - postgresql:// → PostgresStore (when available)
    - auto:// → Fallback chain
    """

    _backends: dict[str, type] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Register default backends on first use."""
        if cls._initialized:
            return

        # Register FileStore
        from metalab.store.file import FileStore

        cls.register("file", FileStore)

        # Register PostgresStore (optional dependency)
        try:
            from metalab.store.postgres import PostgresStore

            cls.register("postgresql", PostgresStore)
            cls.register("postgres", PostgresStore)
        except ImportError:
            pass  # psycopg not installed

        cls._initialized = True

    @classmethod
    def register(cls, scheme: str, backend_class: type) -> None:
        """
        Register a storage backend for a scheme.

        Args:
            scheme: The URI scheme (e.g., "file", "postgresql").
            backend_class: The Store implementation class.
        """
        cls._backends[scheme] = backend_class
        logger.debug(f"Registered store backend: {scheme} -> {backend_class.__name__}")

    @classmethod
    def from_locator(
        cls,
        locator: str,
        *,
        connect_timeout: float = 5.0,
        fallback: str | None = None,
        **kwargs: Any,
    ) -> "Store":
        """
        Create a Store instance from a locator string.

        Args:
            locator: Store locator URI or path.
            connect_timeout: Timeout for network stores (seconds).
            fallback: Fallback locator if primary fails.
            **kwargs: Additional arguments passed to the backend constructor.

        Returns:
            A Store instance.

        Raises:
            ValueError: If the scheme is not supported.
            ConnectionError: If connection fails and no fallback is available.

        Examples:
            # FileStore from path
            store = StoreFactory.from_locator("/path/to/store")

            # FileStore from URI
            store = StoreFactory.from_locator("file:///path/to/store")

            # With fallback
            store = StoreFactory.from_locator(
                "postgresql://localhost/db",
                fallback="file:///path/to/store",
            )
        """
        cls._ensure_initialized()

        info = parse_locator(locator)

        # Handle auto:// scheme with fallback chain
        if info.scheme == "auto":
            primary = info.params.get("primary")
            auto_fallback = info.params.get("fallback")
            if not primary:
                raise ValueError("auto:// locator requires 'primary' parameter")
            return cls.from_locator(
                primary,
                connect_timeout=connect_timeout,
                fallback=auto_fallback or fallback,
                **kwargs,
            )

        # Get backend class
        backend_class = cls._backends.get(info.scheme)
        if backend_class is None:
            # Check if it's a known but not-yet-implemented backend
            if info.scheme in ("postgresql", "postgres"):
                raise ValueError(
                    f"PostgreSQL store not yet implemented. "
                    f"Install with: pip install metalab[postgres]"
                )
            raise ValueError(f"Unknown store scheme: {info.scheme}")

        # Create store instance
        try:
            if info.scheme == "file":
                # FileStore doesn't use experiment_id (only PostgresStore does)
                kwargs.pop("experiment_id", None)
                return backend_class(info.path, **kwargs)
            elif info.scheme in ("postgresql", "postgres"):
                # PostgresStore accepts the full connection string
                # Extract experiments_root from kwargs or params
                experiments_root = kwargs.pop(
                    "experiments_root", None
                ) or info.params.get("experiments_root")
                if experiments_root is None:
                    raise ValueError(
                        "PostgresStore requires 'experiments_root' parameter. "
                        "Example: postgresql://host/db?experiments_root=/shared/experiments"
                    )
                # Extract experiment_id for nested FileStore directory
                experiment_id = kwargs.pop("experiment_id", None)
                return backend_class(
                    locator,
                    experiments_root=experiments_root,
                    experiment_id=experiment_id,
                    connect_timeout=connect_timeout,
                    **kwargs,
                )
            else:
                # Generic: pass locator info
                return backend_class(info, **kwargs)
        except Exception as e:
            if fallback:
                logger.warning(
                    f"Failed to create store from {locator}: {e}. "
                    f"Falling back to {fallback}"
                )
                return cls.from_locator(fallback, **kwargs)
            raise

    @classmethod
    def supports(cls, scheme: str) -> bool:
        """
        Check if a scheme is supported.

        Args:
            scheme: The URI scheme to check.

        Returns:
            True if the scheme has a registered backend.
        """
        cls._ensure_initialized()
        return scheme in cls._backends


# Convenience function
def create_store(
    locator: str,
    *,
    fallback: str | None = None,
    **kwargs: Any,
) -> "Store":
    """
    Create a Store instance from a locator string.

    This is a convenience wrapper around StoreFactory.from_locator().

    Args:
        locator: Store locator URI or path.
        fallback: Fallback locator if primary fails.
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        A Store instance.

    Examples:
        # Simple usage
        store = create_store("/path/to/store")

        # With fallback
        store = create_store(
            "postgresql://localhost/db",
            fallback="/path/to/store",
        )
    """
    return StoreFactory.from_locator(locator, fallback=fallback, **kwargs)
