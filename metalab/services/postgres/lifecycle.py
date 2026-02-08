"""
PostgreSQL service lifecycle management.

Start, stop, and discover PostgreSQL instances for local development
and SLURM/HPC environments.  These functions are consumed directly by
the CLI and by ``PostgresPlugin.plan_local``.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import cast

from metalab.services.postgres._bash import PgBashParams, render_slurm_job_script
from metalab.services.postgres.config import (
    DEFAULT_LOCAL_SERVICE_DIR,
    DEFAULT_PORT,
    PostgresService,
    PostgresServiceConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _ensure_database(service: PostgresService, binaries: dict[str, Path]) -> None:
    """Ensure the database exists."""
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


def _configure_pg_hba(data_dir: Path, config: PostgresServiceConfig) -> None:
    """Configure pg_hba.conf for network access."""
    hba_file = data_dir / "pg_hba.conf"
    content = hba_file.read_text()
    auth = config.auth_method
    new_line = f"host    all    all    0.0.0.0/0    {auth}\n"

    if new_line not in content:
        with hba_file.open("a") as f:
            f.write(f"\n# Added by metalab\n{new_line}")


def _configure_postgresql(data_dir: Path, config: PostgresServiceConfig) -> None:
    """Configure postgresql.conf."""
    import re

    conf_file = data_dir / "postgresql.conf"

    settings = {
        "listen_addresses": f"'{config.listen_addresses}'",
        "port": str(config.port),
        "max_connections": str(config.max_connections),
    }

    content = conf_file.read_text()

    for key in settings:
        content = re.sub(
            rf"^{key}\s*=.*$",
            f"# {key} = (overridden by metalab)",
            content,
            flags=re.MULTILINE,
        )

    content += "\n# metalab settings\n"
    for key, value in settings.items():
        content += f"{key} = {value}\n"

    conf_file.write_text(content)


# ---------------------------------------------------------------------------
# Container start
# ---------------------------------------------------------------------------


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

    _run_cmd([docker, "rm", "-f", container_name], check=False)

    cmd = [
        docker, "run", "-d",
        "--name", container_name,
        "-p", f"{config.port}:5432",
        "-v", f"{data_dir}:/var/lib/postgresql/data",
        "-e", f"POSTGRES_USER={config.user}",
        "-e", f"POSTGRES_DB={config.database}",
    ]

    if password:
        cmd.extend(["-e", f"POSTGRES_PASSWORD={password}"])
    else:
        cmd.extend(["-e", "POSTGRES_HOST_AUTH_METHOD=trust"])

    cmd.append("postgres:15-alpine")
    _run_cmd(cmd)

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

    _wait_for_postgres(service, timeout=30)
    service.save(service_file)
    logger.info(f"PostgreSQL started: {service.connection_string}")
    return service


# ---------------------------------------------------------------------------
# Native start
# ---------------------------------------------------------------------------


def _start_postgres_native(
    config: PostgresServiceConfig,
    service_dir: Path,
    service_file: Path,
    binaries: dict[str, Path],
    password: str | None,
) -> PostgresService:
    """Start PostgreSQL using native binaries."""
    data_dir = config.data_dir or (service_dir / "data")

    if not (data_dir / "PG_VERSION").exists():
        logger.info(f"Initializing PostgreSQL data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

        cmd = [str(binaries["initdb"]), "-D", str(data_dir)]
        if password:
            pwfile = service_dir / ".pgpass"
            pwfile.write_text(password)
            os.chmod(pwfile, 0o600)
            cmd.extend(["--pwfile", str(pwfile)])
            cmd.extend(["-A", "scram-sha-256"])
        else:
            cmd.extend(["-A", "trust"])

        _run_cmd(cmd)

        if config.listen_addresses != "localhost":
            _configure_pg_hba(data_dir, config)
        _configure_postgresql(data_dir, config)

    log_file = service_dir / "postgres.log"
    cmd = [
        str(binaries["pg_ctl"]),
        "-D", str(data_dir),
        "-l", str(log_file),
        "-o", f"-p {config.port}",
        "start",
    ]
    _run_cmd(cmd)

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

    _wait_for_postgres(service, timeout=30)
    _ensure_database(service, binaries)
    service.save(service_file)
    logger.info(f"PostgreSQL started: {service.connection_string}")
    return service


# ---------------------------------------------------------------------------
# Public lifecycle functions
# ---------------------------------------------------------------------------


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

    if service_file.exists():
        existing = PostgresService.load(service_file)
        if _is_postgres_running(existing):
            logger.info(f"PostgreSQL already running at {existing.connection_string}")
            return existing

    password = config.password
    if config.auth_method == "scram-sha-256" and password is None:
        password = secrets.token_urlsafe(16)

    docker = shutil.which("docker") or shutil.which("podman")
    if docker:
        return _start_postgres_container(
            config, service_dir, service_file, docker, password
        )

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
            listen_addresses="*",
            auth_method="trust",
        )

    service_dir = store_root / "services" / "postgres"
    service_dir.mkdir(parents=True, exist_ok=True)
    service_file = service_dir / "service.json"

    # Check if already running
    if service_file.exists():
        existing = PostgresService.load(service_file)
        if existing.slurm_job_id:
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

    password = config.password
    if config.auth_method == "scram-sha-256" and password is None:
        password = secrets.token_urlsafe(16)

    data_dir = config.data_dir or (service_dir / "data")

    params = PgBashParams(
        user=config.user,
        password=password,
        port=config.port or DEFAULT_PORT,
        database=config.database,
        auth_method=config.auth_method,
        data_dir=data_dir,
        service_dir=service_dir,
        service_file=service_file,
    )

    script_content = render_slurm_job_script(
        params,
        slurm_partition=slurm_partition,
        slurm_time=slurm_time,
        slurm_memory=slurm_memory,
    )

    script_path = service_dir / "start_postgres.sh"
    script_path.write_text(script_content)
    os.chmod(script_path, 0o700)

    result = _run_cmd(["sbatch", str(script_path)])

    job_id = None
    for line in result.stdout.split("\n"):
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break

    if not job_id:
        raise RuntimeError(f"Failed to parse SLURM job ID: {result.stdout}")

    logger.info(f"Submitted SLURM job {job_id} for PostgreSQL service")

    timeout = 120
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

        result = _run_cmd(["squeue", "-j", job_id, "-h"], check=False)
        if result.returncode != 0 or not result.stdout.strip():
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
    if service_path:
        service_file = Path(service_path)
        if service_file.exists():
            service = PostgresService.load(service_file)
            if _is_postgres_running(service):
                return service
            return None

    if store_root:
        service_file = Path(store_root) / "services" / "postgres" / "service.json"
        if service_file.exists():
            service = PostgresService.load(service_file)
            if _is_postgres_running(service):
                return service

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

    if service.slurm_job_id:
        _run_cmd(["scancel", service.slurm_job_id], check=False)
        logger.info(f"Cancelled SLURM job {service.slurm_job_id}")

    if service.pgdata:
        binaries = _find_postgres_binaries()
        if binaries.get("pg_ctl"):
            _run_cmd(
                [str(binaries["pg_ctl"]), "-D", service.pgdata, "stop", "-m", "fast"],
                check=False,
            )

    docker = shutil.which("docker") or shutil.which("podman")
    if docker:
        for prefix in ["metalab-postgres-", "metalab_postgres_"]:
            _run_cmd([docker, "stop", f"{prefix}{service_id}"], check=False)
            _run_cmd([docker, "rm", f"{prefix}{service_id}"], check=False)

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
