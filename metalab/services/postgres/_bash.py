"""
Bash template generation for PostgreSQL service scripts.

Single source of truth for the bash fragments used by the
SLURM provider (``plugin.plan_slurm``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PgBashParams:
    """Parameters consumed by the bash templates."""

    user: str
    password: str | None
    port: int
    database: str
    auth_method: str
    data_dir: Path
    service_dir: Path
    service_file: Path
    max_connections: int = 200

    @property
    def password_literal(self) -> str:
        return self.password or ""

    @property
    def auth_prefix(self) -> str:
        if self.password:
            return f"{self.user}:{self.password}"
        return self.user


def render_setup_bash(p: PgBashParams) -> str:
    """Render the PostgreSQL setup bash fragment.

    This is the canonical bash template for initialising and starting
    PostgreSQL on a remote node (SLURM, etc.).  It:

    1. Locates PostgreSQL binaries.
    2. Initialises PGDATA if needed.
    3. Starts the server via ``pg_ctl``.
    4. Waits for readiness via ``pg_isready``.
    5. Creates the database.
    6. Writes ``service.json`` for discovery.
    """
    return f"""
echo "Starting PostgreSQL..."

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
export PGDATA_BASE="{p.data_dir}"
export PGDATA="$PGDATA_BASE"
if [ -d "$PGDATA_BASE" ] && [ ! -f "$PGDATA_BASE/PG_VERSION" ]; then
    if [ -n "$(ls -A "$PGDATA_BASE" 2>/dev/null)" ]; then
        export PGDATA="$PGDATA_BASE/pgdata"
    fi
fi
mkdir -p "$PGDATA"
PASSWORD={json.dumps(p.password_literal)}

# Initialize if needed
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing PostgreSQL data directory..."
    if [ -n "$PASSWORD" ]; then
        PWFILE="{p.service_dir}/.pgpass_init_$SLURM_JOB_ID"
        printf "%s" "$PASSWORD" > "$PWFILE"
        chmod 600 "$PWFILE"
        initdb -D "$PGDATA" -A {p.auth_method} --pwfile="$PWFILE"
        rm -f "$PWFILE"
    else
        initdb -D "$PGDATA" -A {p.auth_method}
    fi

    # Configure for network access
    echo "host all all 0.0.0.0/0 {p.auth_method}" >> "$PGDATA/pg_hba.conf"
    echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
    echo "port = {p.port}" >> "$PGDATA/postgresql.conf"
    echo "max_connections = {p.max_connections}" >> "$PGDATA/postgresql.conf"
fi

# Start PostgreSQL (background)
pg_ctl -D "$PGDATA" -l "{p.service_dir}/postgres.log" start

# Wait for PostgreSQL to be ready
for i in $(seq 1 30); do
    if pg_isready -h localhost -p {p.port} -q; then
        break
    fi
    sleep 1
done

if ! pg_isready -h localhost -p {p.port} -q; then
    echo "PostgreSQL failed to start" >&2
    exit 1
fi

# Create database if needed
if [ -n "$PASSWORD" ]; then
    PGPASSWORD="$PASSWORD" createdb -h localhost -p {p.port} -U "{p.user}" -w "{p.database}" 2>/dev/null || true
else
    createdb -h localhost -p {p.port} -U "{p.user}" "{p.database}" 2>/dev/null || true
fi

# Write postgres service file
cat > "{p.service_file}" << EOF
{{
    "host": "$HOSTNAME",
    "port": {p.port},
    "database": "{p.database}",
    "user": "{p.user}",
    "password": {json.dumps(p.password)},
    "pgdata": "$PGDATA",
    "slurm_job_id": "$SLURM_JOB_ID",
    "started_at": "$(date -Iseconds)",
    "connection_string": "postgresql://{p.auth_prefix}@$HOSTNAME:{p.port}/{p.database}"
}}
EOF
chmod 600 "{p.service_file}"

echo "PostgreSQL ready on $HOSTNAME:{p.port}"
"""


def render_cleanup_bash() -> str:
    """Render the PostgreSQL cleanup bash fragment."""
    return (
        'if [ -n "${PGDATA:-}" ] && command -v pg_ctl >/dev/null 2>&1; then '
        'pg_ctl -D "$PGDATA" stop -m fast 2>/dev/null || true; fi'
    )
