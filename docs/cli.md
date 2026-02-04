# CLI Reference

The `metalab` CLI provides commands for managing PostgreSQL services and transferring data between stores.

## Global Options

```
metalab [-v | --verbose] <command>
```

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Enable verbose logging |

---

## PostgreSQL Commands

Manage PostgreSQL services for experiment storage.

### metalab postgres start

Start a PostgreSQL service either locally or as a SLURM job.

**Usage:**

```
metalab postgres start [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--store`, `-s` | — | Store root directory (for SLURM service discovery). Defaults to `--experiments-root` |
| `--port`, `-p` | `5432` | PostgreSQL port |
| `--database`, `-d` | `metalab` | Database name |
| `--slurm` | — | Submit as SLURM job instead of running locally |
| `--slurm-partition` | `default` | SLURM partition |
| `--slurm-time` | `24:00:00` | SLURM walltime |
| `--slurm-memory` | `4G` | SLURM memory allocation |
| `--data-dir` | — | PostgreSQL data directory (PGDATA) |
| `--auth-method` | `trust` | Authentication method (`trust` or `scram-sha-256`) |
| `--password` | — | PostgreSQL password (used with `scram-sha-256`) |
| `--experiments-root` | — | Shared experiments root (also used for SLURM service discovery if `--store` is omitted) |
| `--schema` | `public` | PostgreSQL schema for PostgresStore |
| `--print-store-locator` | — | Print PostgresStore locator for this service |

**Examples:**

Start PostgreSQL locally with default settings:

```bash
metalab postgres start
```

Start PostgreSQL on a custom port with a specific database:

```bash
metalab postgres start --port 5433 --database experiments
```

Start PostgreSQL as a SLURM job with custom resources:

```bash
metalab postgres start \
  --slurm \
  --store /shared/experiments \
  --slurm-partition gpu \
  --slurm-time 48:00:00 \
  --slurm-memory 8G
```

Start with password authentication and print the store locator:

```bash
metalab postgres start \
  --auth-method scram-sha-256 \
  --password mypassword \
  --experiments-root /path/to/experiments \
  --print-store-locator
```

---

### metalab postgres status

Check the status of a running PostgreSQL service.

**Usage:**

```
metalab postgres status [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--store`, `-s` | — | Store root directory (defaults to `--experiments-root`) |
| `--json` | — | Output as JSON |
| `--experiments-root` | — | Shared experiments root (also used for SLURM service discovery if `--store` is omitted) |
| `--schema` | `public` | PostgreSQL schema for PostgresStore |
| `--store-locator` | — | Include PostgresStore locator in output |

**Examples:**

Check status of local PostgreSQL service:

```bash
metalab postgres status
```

Check status with JSON output:

```bash
metalab postgres status --json
```

Check status for a SLURM-managed service and include the store locator:

```bash
metalab postgres status \
  --store /shared/experiments \
  --store-locator
```

**Sample Output:**

```
PostgreSQL service is running:
  Connection: postgresql://localhost:5432/metalab
  Host: localhost
  Port: 5432
  Database: metalab
  SLURM Job: 12345
  Started: 2024-01-15T10:30:00
```

---

### metalab postgres stop

Stop a running PostgreSQL service.

**Usage:**

```
metalab postgres stop [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--store`, `-s` | — | Store root directory (defaults to `--experiments-root`) |
| `--experiments-root` | — | Shared experiments root (also used for SLURM service discovery if `--store` is omitted) |

**Examples:**

Stop local PostgreSQL service:

```bash
metalab postgres stop
```

Stop a SLURM-managed PostgreSQL service:

```bash
metalab postgres stop --store /shared/experiments
```

---

## Store Commands

Transfer data between different storage backends.

### metalab store export

Export data from a source store to a destination store.

**Usage:**

```
metalab store export --from <source> --to <destination> [options]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--from`, `-f` | Yes | Source store locator (e.g., `postgresql://localhost/db`) |
| `--to`, `-t` | Yes | Destination store locator (e.g., `file:///path/to/store`) |
| `--experiment`, `-e` | No | Filter by experiment ID |
| `--include-artifacts` | No | Include artifact files (usually skipped for performance) |

**Examples:**

Export all data from PostgreSQL to a file store:

```bash
metalab store export \
  --from postgresql://localhost:5432/metalab \
  --to file:///home/user/experiments
```

Export a specific experiment:

```bash
metalab store export \
  --from postgresql://localhost/metalab \
  --to file:///backup/experiments \
  --experiment my_experiment_id
```

Export including artifact files:

```bash
metalab store export \
  --from postgresql://localhost/metalab \
  --to file:///full-backup \
  --include-artifacts
```

**Sample Output:**

```
Exporting: 150/150
Export complete:
  runs: 150
  artifacts: 450
```

---

### metalab store import

Import data from a source store to a destination store.

**Usage:**

```
metalab store import --from <source> --to <destination> [options]
```

**Options:**

| Option | Required | Description |
|--------|----------|-------------|
| `--from`, `-f` | Yes | Source store locator (e.g., `file:///path/to/store`) |
| `--to`, `-t` | Yes | Destination store locator (e.g., `postgresql://localhost/db`) |
| `--experiment`, `-e` | No | Filter by experiment ID |

**Examples:**

Import all data from a file store to PostgreSQL:

```bash
metalab store import \
  --from file:///home/user/experiments \
  --to postgresql://localhost:5432/metalab
```

Import a specific experiment:

```bash
metalab store import \
  --from file:///backup/experiments \
  --to postgresql://localhost/metalab \
  --experiment my_experiment_id
```

**Sample Output:**

```
Importing: 150/150
Import complete:
  runs: 150
  artifacts: 450
```

---

## Store Locators

Store locators are URIs that identify storage backends:

| Type | Format | Example |
|------|--------|---------|
| File Store | `file://<path>` | `file:///home/user/experiments` |
| PostgreSQL | `postgresql://<host>:<port>/<database>` | `postgresql://localhost:5432/metalab` |

For PostgreSQL stores with additional options, query parameters can be used:

```
postgresql://localhost:5432/metalab?schema=experiments
```
