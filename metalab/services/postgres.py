"""
PostgreSQL service manager for metalab.

Provides utilities for starting, stopping, and monitoring PostgreSQL
services for local development and HPC/SLURM environments.

Service discovery:
    Services write a JSON file with connection info to a known location:
    - Local: ~/.metalab/services/postgres/<service_id>/service.json
    - SLURM: <store_root>/services/postgres/service.json

Service file format:
    {
        "host": "hostname or IP",
        "port": 5432,
        "database": "metalab",
        "user": "username",
        "password": "generated_password",  # Optional, for scram auth
        "pgdata": "/path/to/data",
        "pid": 12345,
        "slurm_job_id": "123456",  # Only for SLURM
        "started_at": "2024-01-01T00:00:00",
        "connection_string": "postgresql://..."
    }
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# Default locations
DEFAULT_LOCAL_SERVICE_DIR = Path.home() / ".metalab" / "services" / "postgres"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "metalab"


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
    max_connections: int = 100

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
    service: PostgresService,
    *,
    experiments_root: Path | str,
    schema: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str:
    """
    Build a PostgresStore locator from a running service.

    Ensures required experiments_root is included and preserves existing params.
    """
    parsed = urlparse(service.connection_string)
    params = {}
    if parsed.query:
        for key, values in parse_qs(parsed.query).items():
            params[key] = values[-1] if values else ""

    params["experiments_root"] = str(Path(experiments_root))
    if schema:
        params["schema"] = schema
    if extra_params:
        params.update(extra_params)

    return urlunparse(parsed._replace(query=urlencode(params)))


def _find_postgres_binaries() -> dict[str, Path | None]:
    """Find PostgreSQL binaries in PATH or common locations."""
    binaries = ["initdb", "pg_ctl", "psql", "createdb"]
    found = {}

    # Check PATH first
    for name in binaries:
        path = shutil.which(name)
        found[name] = Path(path) if path else None

    # Check common locations if not in PATH
    common_paths = [
        Path("/usr/lib/postgresql/15/bin"),
        Path("/usr/lib/postgresql/14/bin"),
        Path("/usr/lib/postgresql/13/bin"),
        Path("/usr/local/pgsql/bin"),
        Path("/opt/homebrew/opt/postgresql/bin"),
        Path("/usr/local/opt/postgresql/bin"),
    ]

    for bin_dir in common_paths:
        if bin_dir.exists():
            for name in binaries:
                if found[name] is None:
                    candidate = bin_dir / name
                    if candidate.exists():
                        found[name] = candidate

    return found


def _run_cmd(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command with optional environment."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return result


def start_postgres_local(
    config: PostgresServiceConfig | None = None,
    *,
    service_id: str = "default",
    service_dir: Path | None = None,
) -> PostgresService:
    """
    Start a local PostgreSQL service.

    Uses Docker/Podman if available, otherwise falls back to local binaries.

    Args:
        config: Service configuration.
        service_id: Unique identifier for this service instance.
        service_dir: Directory for service files.

    Returns:
        PostgresService with connection info.

    Raises:
        RuntimeError: If PostgreSQL cannot be started.
    """
    if config is None:
        config = PostgresServiceConfig()

    if service_dir is None:
        service_dir = DEFAULT_LOCAL_SERVICE_DIR / service_id

    service_dir.mkdir(parents=True, exist_ok=True)
    service_file = service_dir / "service.json"

    # Check if already running
    if service_file.exists():
        existing = PostgresService.load(service_file)
        if _is_postgres_running(existing):
            logger.info(f"PostgreSQL already running at {existing.connection_string}")
            return existing

    # Generate password if using scram auth
    password = config.password
    if config.auth_method == "scram-sha-256" and password is None:
        password = secrets.token_urlsafe(16)

    # Try Docker/Podman first
    docker = shutil.which("docker") or shutil.which("podman")
    if docker:
        return _start_postgres_container(
            config, service_dir, service_file, docker, password
        )

    # Fall back to local binaries
    binaries = _find_postgres_binaries()
    if all(binaries.values()):
        resolved_binaries = cast(dict[str, Path], binaries)
        return _start_postgres_native(
            config, service_dir, service_file, resolved_binaries, password
        )

    raise RuntimeError(
        "PostgreSQL not found. Install PostgreSQL or Docker/Podman, "
        "or load the appropriate module (e.g., 'module load postgresql')"
    )


def _start_postgres_container(
    config: PostgresServiceConfig,
    service_dir: Path,
    service_file: Path,
    docker: str,
    password: str | None,
) -> PostgresService:
    """Start PostgreSQL in a Docker/Podman container."""
    container_name = f"metalab-postgres-{service_dir.name}"
    data_dir = config.data_dir or (service_dir / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Stop any existing container
    _run_cmd([docker, "rm", "-f", container_name], check=False)

    # Build run command
    cmd = [
        docker,
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{config.port}:5432",
        "-v",
        f"{data_dir}:/var/lib/postgresql/data",
        "-e",
        f"POSTGRES_USER={config.user}",
        "-e",
        f"POSTGRES_DB={config.database}",
    ]

    if password:
        cmd.extend(["-e", f"POSTGRES_PASSWORD={password}"])
    else:
        cmd.extend(["-e", "POSTGRES_HOST_AUTH_METHOD=trust"])

    cmd.append("postgres:15-alpine")

    _run_cmd(cmd)

    # Wait for startup
    time.sleep(2)

    service = PostgresService(
        host="localhost",
        port=config.port,
        database=config.database,
        user=config.user,
        password=password,
        pgdata=str(data_dir),
        started_at=datetime.now().isoformat(),
    )

    # Wait for ready
    _wait_for_postgres(service, timeout=30)

    service.save(service_file)
    logger.info(f"PostgreSQL started: {service.connection_string}")

    return service


def _start_postgres_native(
    config: PostgresServiceConfig,
    service_dir: Path,
    service_file: Path,
    binaries: dict[str, Path],
    password: str | None,
) -> PostgresService:
    """Start PostgreSQL using native binaries."""
    data_dir = config.data_dir or (service_dir / "data")

    # Initialize data directory if needed
    if not (data_dir / "PG_VERSION").exists():
        logger.info(f"Initializing PostgreSQL data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

        cmd = [str(binaries["initdb"]), "-D", str(data_dir)]
        if password:
            # Write password to temp file for pwfile
            pwfile = service_dir / ".pgpass"
            pwfile.write_text(password)
            os.chmod(pwfile, 0o600)
            cmd.extend(["--pwfile", str(pwfile)])
            cmd.extend(["-A", "scram-sha-256"])
        else:
            cmd.extend(["-A", "trust"])

        _run_cmd(cmd)

        # Configure pg_hba.conf for network access if needed
        if config.listen_addresses != "localhost":
            _configure_pg_hba(data_dir, config)

        # Configure postgresql.conf
        _configure_postgresql(data_dir, config)

    # Start server
    log_file = service_dir / "postgres.log"

    cmd = [
        str(binaries["pg_ctl"]),
        "-D",
        str(data_dir),
        "-l",
        str(log_file),
        "-o",
        f"-p {config.port}",
        "start",
    ]

    _run_cmd(cmd)

    # Get PID
    pid = None
    pid_file = data_dir / "postmaster.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().split("\n")[0])

    service = PostgresService(
        host="localhost",
        port=config.port,
        database=config.database,
        user=config.user,
        password=password,
        pgdata=str(data_dir),
        pid=pid,
        started_at=datetime.now().isoformat(),
    )

    # Wait for ready
    _wait_for_postgres(service, timeout=30)

    # Create database if needed
    _ensure_database(service, binaries)

    service.save(service_file)
    logger.info(f"PostgreSQL started: {service.connection_string}")

    return service


def _configure_pg_hba(data_dir: Path, config: PostgresServiceConfig) -> None:
    """Configure pg_hba.conf for network access."""
    hba_file = data_dir / "pg_hba.conf"

    # Read existing content
    content = hba_file.read_text()

    # Add line for network access
    auth = config.auth_method
    new_line = f"host    all    all    0.0.0.0/0    {auth}\n"

    if new_line not in content:
        with hba_file.open("a") as f:
            f.write(f"\n# Added by metalab\n{new_line}")


def _configure_postgresql(data_dir: Path, config: PostgresServiceConfig) -> None:
    """Configure postgresql.conf."""
    conf_file = data_dir / "postgresql.conf"

    settings = {
        "listen_addresses": f"'{config.listen_addresses}'",
        "port": str(config.port),
        "max_connections": str(config.max_connections),
    }

    content = conf_file.read_text()

    for key, value in settings.items():
        # Comment out existing setting
        import re

        content = re.sub(
            rf"^{key}\s*=.*$",
            f"# {key} = (overridden by metalab)",
            content,
            flags=re.MULTILINE,
        )

    # Add our settings
    content += "\n# metalab settings\n"
    for key, value in settings.items():
        content += f"{key} = {value}\n"

    conf_file.write_text(content)


def _wait_for_postgres(service: PostgresService, timeout: float = 30) -> None:
    """Wait for PostgreSQL to be ready."""
    import socket

    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((service.host, service.port))
            sock.close()
            if result == 0:
                return
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"PostgreSQL failed to start within {timeout}s")


def _ensure_database(service: PostgresService, binaries: dict[str, Path]) -> None:
    """Ensure the database exists."""
    # Try to create database (ignore error if exists)
    cmd = [
        str(binaries["createdb"]),
        "-h",
        service.host,
        "-p",
        str(service.port),
        "-U",
        service.user,
        service.database,
    ]
    _run_cmd(cmd, check=False)


def _is_postgres_running(service: PostgresService) -> bool:
    """Check if a PostgreSQL service is running."""
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((service.host, service.port))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_postgres_slurm(
    config: PostgresServiceConfig | None = None,
    *,
    store_root: Path,
    slurm_partition: str = "default",
    slurm_time: str = "24:00:00",
    slurm_memory: str = "4G",
) -> PostgresService:
    """
    Start a PostgreSQL service via SLURM job.

    Submits a SLURM job that runs PostgreSQL on a compute node.
    Service discovery file is written to {store_root}/services/postgres/service.json.

    Args:
        config: Service configuration.
        store_root: Root directory for the store (on shared filesystem).
        slurm_partition: SLURM partition to submit to.
        slurm_time: Maximum walltime for the service job.
        slurm_memory: Memory allocation.

    Returns:
        PostgresService with connection info.
    """
    if config is None:
        config = PostgresServiceConfig(
            listen_addresses="*",  # Need network access from other nodes
            auth_method="trust",  # For simplicity on internal network
        )

    service_dir = store_root / "services" / "postgres"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_file = service_dir / "service.json"

    # Check if already running
    if service_file.exists():
        existing = PostgresService.load(service_file)
        if existing.slurm_job_id:
            # Check if job is still running
            result = _run_cmd(
                ["squeue", "-j", existing.slurm_job_id, "-h"],
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                if _is_postgres_running(existing):
                    logger.info(
                        f"PostgreSQL already running: {existing.connection_string}"
                    )
                    return existing

    # Generate password
    password = config.password
    if config.auth_method == "scram-sha-256" and password is None:
        password = secrets.token_urlsafe(16)

    # PGDATA on shared filesystem
    data_dir = config.data_dir or (service_dir / "data")

    # Create SLURM job script
    password_literal = password or ""
    auth_prefix = config.user
    if password:
        auth_prefix = f"{config.user}:{password}"
    script_content = f"""#!/bin/bash
#SBATCH --job-name=metalab-postgres
#SBATCH --partition={slurm_partition}
#SBATCH --time={slurm_time}
#SBATCH --mem={slurm_memory}
#SBATCH --cpus-per-task=2
#SBATCH --output={service_dir}/slurm-%j.out
#SBATCH --error={service_dir}/slurm-%j.err

# Get hostname
HOSTNAME=$(hostname)
echo "Starting PostgreSQL on $HOSTNAME"

# Ensure PostgreSQL binaries are available
if [ -n "${{METALAB_PG_BIN_DIR:-}}" ] && [ -d "${{METALAB_PG_BIN_DIR:-}}" ]; then
    export PATH="$METALAB_PG_BIN_DIR:$PATH"
fi
if ! command -v initdb >/dev/null 2>&1; then
    for d in /usr/lib/postgresql/15/bin /usr/lib/postgresql/14/bin /usr/lib/postgresql/13/bin /usr/local/pgsql/bin; do
        if [ -x "$d/initdb" ]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi
for bin in initdb pg_ctl pg_isready createdb; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "PostgreSQL binary not found: $bin. Set METALAB_PG_BIN_DIR or load a Postgres module." >&2
        exit 1
    fi
done

# Setup PGDATA
export PGDATA_BASE="{data_dir}"
export PGDATA="$PGDATA_BASE"
if [ -d "$PGDATA_BASE" ] && [ ! -f "$PGDATA_BASE/PG_VERSION" ]; then
    if [ -n "$(ls -A "$PGDATA_BASE" 2>/dev/null)" ]; then
        # Avoid initdb on a mount point with dotfiles.
        export PGDATA="$PGDATA_BASE/pgdata"
    fi
fi
mkdir -p "$PGDATA"
PASSWORD={json.dumps(password_literal)}

# Initialize if needed
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL data directory..."
    if [ -n "$PASSWORD" ]; then
        PWFILE="{service_dir}/.pgpass_init_$SLURM_JOB_ID"
        printf "%s" "$PASSWORD" > "$PWFILE"
        chmod 600 "$PWFILE"
        initdb -D "$PGDATA" -A {config.auth_method} --pwfile="$PWFILE"
        rm -f "$PWFILE"
    else
        initdb -D "$PGDATA" -A {config.auth_method}
    fi
    
    # Configure for network access
    echo "host all all 0.0.0.0/0 {config.auth_method}" >> "$PGDATA/pg_hba.conf"
    echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
    echo "port = {config.port}" >> "$PGDATA/postgresql.conf"
fi

# Start PostgreSQL
pg_ctl -D "$PGDATA" -l "{service_dir}/postgres.log" start

# Wait for startup
sleep 5

# Create database if needed
createdb -h localhost -p {config.port} {config.database} 2>/dev/null || true

# Write service file
cat > "{service_file}" << EOF
{{
    "host": "$HOSTNAME",
    "port": {config.port},
    "database": "{config.database}",
    "user": "{config.user}",
    "password": {json.dumps(password)},
    "pgdata": "$PGDATA",
    "slurm_job_id": "$SLURM_JOB_ID",
    "started_at": "$(date -Iseconds)",
    "connection_string": "postgresql://{auth_prefix}@$HOSTNAME:{config.port}/{config.database}"
}}
EOF
chmod 600 "{service_file}"

echo "PostgreSQL ready on $HOSTNAME:{config.port}"

# Keep job running (PostgreSQL runs in background)
while true; do
    if ! pg_isready -h localhost -p {config.port} -q; then
        echo "PostgreSQL stopped, exiting"
        exit 1
    fi
    sleep 60
done
"""

    script_path = service_dir / "start_postgres.sh"
    script_path.write_text(script_content)
    os.chmod(script_path, 0o700)

    # Submit job
    result = _run_cmd(["sbatch", str(script_path)])

    # Parse job ID
    # Output: "Submitted batch job 12345"
    job_id = None
    for line in result.stdout.split("\n"):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break

    if not job_id:
        raise RuntimeError(f"Failed to parse SLURM job ID: {result.stdout}")

    logger.info(f"Submitted SLURM job {job_id} for PostgreSQL service")

    # Wait for service file to appear
    timeout = 120  # 2 minutes for job to start
    start = time.time()
    while time.time() - start < timeout:
        if service_file.exists():
            try:
                service = PostgresService.load(service_file)
                if _is_postgres_running(service):
                    logger.info(f"PostgreSQL ready: {service.connection_string}")
                    return service
            except Exception:
                pass
        time.sleep(5)

        # Check job status
        result = _run_cmd(["squeue", "-j", job_id, "-h"], check=False)
        if result.returncode != 0 or not result.stdout.strip():
            # Job finished (possibly failed)
            err_file = service_dir / f"slurm-{job_id}.err"
            if err_file.exists():
                logger.error(f"SLURM job failed: {err_file.read_text()}")
            raise RuntimeError(f"SLURM job {job_id} failed to start PostgreSQL")

    raise RuntimeError(f"PostgreSQL service did not start within {timeout}s")


def get_service_info(
    service_path: Path | str | None = None,
    *,
    store_root: Path | None = None,
    service_id: str = "default",
) -> PostgresService | None:
    """
    Get information about a running PostgreSQL service.

    Args:
        service_path: Direct path to service.json file.
        store_root: Store root for SLURM service discovery.
        service_id: Service ID for local services.

    Returns:
        PostgresService if found and running, None otherwise.
    """
    # Try direct path
    if service_path:
        service_file = Path(service_path)
        if service_file.exists():
            service = PostgresService.load(service_file)
            if _is_postgres_running(service):
                return service
            return None

    # Try store root (SLURM)
    if store_root:
        service_file = Path(store_root) / "services" / "postgres" / "service.json"
        if service_file.exists():
            service = PostgresService.load(service_file)
            if _is_postgres_running(service):
                return service

    # Try local service
    service_file = DEFAULT_LOCAL_SERVICE_DIR / service_id / "service.json"
    if service_file.exists():
        service = PostgresService.load(service_file)
        if _is_postgres_running(service):
            return service

    return None


def stop_postgres(
    service_path: Path | str | None = None,
    *,
    store_root: Path | None = None,
    service_id: str = "default",
) -> bool:
    """
    Stop a running PostgreSQL service.

    Args:
        service_path: Direct path to service.json file.
        store_root: Store root for SLURM service discovery.
        service_id: Service ID for local services.

    Returns:
        True if stopped successfully, False if not running.
    """
    service = get_service_info(
        service_path, store_root=store_root, service_id=service_id
    )

    if service is None:
        logger.info("PostgreSQL service not running")
        return False

    # If SLURM job, cancel it
    if service.slurm_job_id:
        _run_cmd(["scancel", service.slurm_job_id], check=False)
        logger.info(f"Cancelled SLURM job {service.slurm_job_id}")

    # Try pg_ctl stop
    if service.pgdata:
        binaries = _find_postgres_binaries()
        if binaries.get("pg_ctl"):
            _run_cmd(
                [str(binaries["pg_ctl"]), "-D", service.pgdata, "stop", "-m", "fast"],
                check=False,
            )

    # Try Docker stop
    docker = shutil.which("docker") or shutil.which("podman")
    if docker:
        # Try to find and stop container
        for prefix in ["metalab-postgres-", "metalab_postgres_"]:
            _run_cmd([docker, "stop", f"{prefix}{service_id}"], check=False)
            _run_cmd([docker, "rm", f"{prefix}{service_id}"], check=False)

    # Remove service file
    if service_path:
        Path(service_path).unlink(missing_ok=True)
    elif store_root:
        (Path(store_root) / "services" / "postgres" / "service.json").unlink(
            missing_ok=True
        )
    else:
        (DEFAULT_LOCAL_SERVICE_DIR / service_id / "service.json").unlink(
            missing_ok=True
        )

    logger.info("PostgreSQL service stopped")
    return True
