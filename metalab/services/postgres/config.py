"""
PostgreSQL service configuration and service-info dataclasses.

Contains:
- ``PostgresServiceConfig``: Configuration for starting a PostgreSQL service.
- ``PostgresService``: Runtime information about a running service.
- ``resolve_password``: Stable password resolution (config → file → generate).
- ``build_store_locator``: Construct a PostgresStore locator URI.
- Default constants (``DEFAULT_PORT``, ``DEFAULT_DATABASE``, etc.).
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Default locations
DEFAULT_LOCAL_SERVICE_DIR = Path.home() / ".metalab" / "services" / "postgres"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "metalab"


def resolve_password(
    service_dir: Path,
    auth_method: str,
    explicit_password: str | None = None,
) -> str | None:
    """Resolve the PostgreSQL password.

    Resolution order: explicit value > persisted ``.pgpass`` > generate new.

    The password must be stable across restarts because PGDATA may already
    be initialised with a previous password.  This function is the single
    source of truth for that invariant, used by both SLURM and local paths.

    Args:
        service_dir: Directory where ``.pgpass`` is stored.
        auth_method: PostgreSQL authentication method.
        explicit_password: Password provided explicitly in config.

    Returns:
        The resolved password, or ``None`` when *auth_method* does not
        require a password (e.g. ``"trust"``).
    """
    if explicit_password:
        return explicit_password
    if auth_method != "scram-sha-256":
        return None
    password_file = service_dir / ".pgpass"
    if password_file.exists():
        return password_file.read_text().strip()
    password = secrets.token_urlsafe(16)
    password_file.write_text(password)
    os.chmod(password_file, 0o600)
    return password


@dataclass
class PostgresServiceConfig:
    """
    Configuration for PostgreSQL service.

    Attributes:
        data_dir: Directory for PGDATA (database files).
        port: Port to listen on.
        database: Database name to create/use.
        user: Database user (defaults to current user).
        password: Password for authentication (generated if not provided).
        auth_method: Authentication method ('trust', 'scram-sha-256').
        listen_addresses: Addresses to listen on ('*' for all, 'localhost' for local).
        max_connections: Maximum concurrent connections.
    """

    data_dir: Path | None = None
    port: int = DEFAULT_PORT
    database: str = DEFAULT_DATABASE
    user: str = field(default_factory=lambda: os.environ.get("USER", "postgres"))
    password: str | None = None
    auth_method: str = "trust"  # 'trust' for dev, 'scram-sha-256' for production
    listen_addresses: str = "localhost"
    max_connections: int = 200

    def __post_init__(self) -> None:
        if self.data_dir is not None:
            self.data_dir = Path(self.data_dir)


@dataclass
class PostgresService:
    """
    Information about a running PostgreSQL service.

    Can be serialized to/from a service.json file.
    """

    host: str
    port: int
    database: str
    user: str
    password: str | None = None
    pgdata: str | None = None
    pid: int | None = None
    slurm_job_id: str | None = None
    started_at: str | None = None

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        auth = f"{self.user}"
        if self.password:
            auth = f"{self.user}:{self.password}"
        return f"postgresql://{auth}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "pgdata": self.pgdata,
            "pid": self.pid,
            "slurm_job_id": self.slurm_job_id,
            "started_at": self.started_at,
            "connection_string": self.connection_string,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostgresService:
        """Create from dictionary."""
        return cls(
            host=data["host"],
            port=data["port"],
            database=data["database"],
            user=data["user"],
            password=data.get("password"),
            pgdata=data.get("pgdata"),
            pid=data.get("pid"),
            slurm_job_id=data.get("slurm_job_id"),
            started_at=data.get("started_at"),
        )

    def save(self, service_file: Path) -> None:
        """Save service info to file."""
        service_file.parent.mkdir(parents=True, exist_ok=True)
        service_file.write_text(json.dumps(self.to_dict(), indent=2))
        # Restrict permissions (contains password)
        os.chmod(service_file, 0o600)

    @classmethod
    def load(cls, service_file: Path) -> PostgresService:
        """Load service info from file."""
        data = json.loads(service_file.read_text())
        return cls.from_dict(data)


def build_store_locator(
    service: PostgresService | str,
    *,
    file_root: Path | str,
    schema: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str:
    """
    Build a PostgresStore locator from a running service or connection string.

    Ensures required file_root is included and preserves existing params.
    """
    conn_str = service if isinstance(service, str) else service.connection_string
    parsed = urlparse(conn_str)
    params = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            params[key] = values[-1] if values else ""

    params["file_root"] = str(Path(file_root))
    if schema:
        params["schema"] = schema
    if extra_params:
        params.update(extra_params)

    return urlunparse(parsed._replace(query=urlencode(params)))
