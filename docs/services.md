# Services and Environments

metalab includes an environment system for provisioning and managing infrastructure services -- such as PostgreSQL and Atlas -- across different deployment targets. Whether you are developing locally or running on an HPC cluster, the same configuration drives service lifecycle, connectivity, and teardown.

## Overview

The environment system handles three concerns:

1. **Configuration** -- `.metalab.toml` defines project-level settings, named environment profiles, and service declarations.
2. **Provisioning** -- `metalab services up` starts the right services for the selected environment (subprocess locally, SLURM jobs on a cluster).
3. **Connectivity** -- `metalab tunnel` opens SSH tunnels so remote services appear on `localhost`.

Supported deployment targets:

| Target | Environment type | How services run |
|--------|-----------------|-----------------|
| Local workstation | `local` | Subprocesses |
| SLURM / HPC | `slurm` | `sbatch` jobs on compute nodes |
| *(future)* | `kubernetes`, cloud | Pods, managed services |

---

## Project Configuration (`.metalab.toml`)

Project configuration lives in `.metalab.toml` at the project root. metalab walks up from the current working directory to find it, so you can run commands from any subdirectory.

### File format

The config uses [TOML](https://toml.io) and has four top-level sections:

| Section | Purpose |
|---------|---------|
| `[project]` | Project name and default environment |
| `[services.*]` | Service declarations (Postgres, Atlas) |
| `[environments.*]` | Named deployment profiles |

### Full example

```toml
[project]
name = "my-project"
default_env = "slurm"

[services.postgres]
auth_method = "scram-sha-256"
database = "metalab"

[services.atlas]
port = 8000

[environments.local]
type = "local"
file_root = "./runs"

[environments.slurm]
type = "slurm"
gateway = "hpc.university.edu"
user = "researcher"
partition = "cpu2019"
time = "1-00:00:00"
memory = "1G"
file_root = "/shared/experiments"
```

### Local overrides (`.metalab.local.toml`)

Personal or sensitive values go in `.metalab.local.toml`, which sits next to `.metalab.toml` and should be gitignored. It uses the same structure and is deep-merged on top of the base config.

```toml
# .metalab.local.toml -- gitignored
[environments.slurm]
user = "jsmith"
ssh_key = "~/.ssh/cluster_key"

[services.postgres]
password = "my-secret-password"
```

### Resolution order

When a config is resolved for a specific environment, values are merged in this order (last wins):

1. Base `[services]` and `[environments]` sections
2. Named environment profile (e.g., `[environments.slurm]`)
3. Local overrides from `.metalab.local.toml`
4. CLI flags (e.g., `--env`)
5. Environment variables (e.g., `METALAB_ENV`)

---

## Environment Profiles

An environment profile is a named deployment target defined under `[environments]` in your config. Each profile specifies a `type` (the backend) and backend-specific settings.

### Listing profiles

```bash
metalab env list
```

```
  local                local
  slurm                slurm *

  * = default (set via [project] default_env)
```

### Inspecting a profile

```bash
metalab env show slurm
```

```
Environment: slurm
  Type: slurm
  File root: /shared/experiments
  Config:
    gateway: hpc.university.edu
    user: researcher
    partition: cpu2019
    time: 1-00:00:00
    memory: 1G
  Services:
    postgres: {'auth_method': 'scram-sha-256', 'database': 'metalab'}
    atlas: {'port': 8000}
```

### Selecting an environment

The active environment is determined by (in priority order):

1. `--env <name>` flag on any command
2. `METALAB_ENV` environment variable
3. `default_env` in `[project]`

```bash
# Explicit flag
metalab services up --env local

# Environment variable
export METALAB_ENV=slurm
metalab services up

# Falls back to default_env in .metalab.toml
metalab services up
```

---

## Service Provisioning

The `metalab services` commands manage the full service lifecycle. The orchestrator reads your resolved config and provisions only the services you have declared.

### `metalab services up`

Provisions services for the selected environment.

```bash
metalab services up --env slurm
```

What happens:

1. Checks for an existing service bundle -- if all services are still alive, reuses it.
2. Creates the appropriate `ServiceEnvironment` (local subprocess manager or SLURM job submitter).
3. Starts **PostgreSQL** if `[services.postgres]` is present in your config.
4. Starts **Atlas** if `[services.atlas]` is present, or if a store locator or `file_root` exists.
5. Saves a `bundle.json` with connection details for all running services.

The `--tunnel` flag opens an SSH tunnel immediately after provisioning:

```bash
metalab services up --env slurm --tunnel
```

### Three provisioning scenarios

| Scenario | Config needed | What starts |
|----------|--------------|-------------|
| **Postgres-backed** | `[services.postgres]` + `[services.atlas]` | PostgreSQL + Atlas (Atlas reads from PG) |
| **File-only** | `file_root` set, no `[services.postgres]` | Atlas only (reads shared filesystem directly) |
| **Reuse existing** | Same as above | Nothing new -- existing bundle is reused if healthy |

### `metalab services status`

Check health of running services:

```bash
metalab services status --env slurm
```

```
  ✓ postgres: cn001:5432 (running)
  ✓ atlas: cn001:8000 (running)
```

Use `--json` for machine-readable output.

### `metalab services down`

Stop all services and clean up:

```bash
metalab services down --env slurm
```

Services are stopped in reverse dependency order (Atlas first, then PostgreSQL). On SLURM, jobs are cancelled with `scancel`. The `bundle.json` file is removed.

---

## Store Discovery

When services are running, experiment configs can use `store: "discover"` to automatically locate the active store without hardcoding URIs.

### How it works

1. `metalab services up` provisions services and saves a `bundle.json` containing the store locator.
2. When metalab encounters `store: "discover"`, it walks up from the current directory looking for `services/bundle.json`.
3. The `store_locator` field from the bundle is used as the store URI.

### Before and after

Without discovery, you must embed connection details:

```python
metalab.run(
    exp,
    store="postgresql://researcher@cn001:5432/metalab?file_root=/shared/experiments",
)
```

With discovery:

```python
metalab.run(exp, store="discover")
```

The locator is resolved at runtime from the running service bundle. This also works in experiment YAML configs:

```yaml
store: discover
```

---

## SSH Authentication and Tunneling

When services run on remote hosts (e.g., SLURM compute nodes), metalab establishes SSH tunnels to make them accessible on your local machine.

### Default behavior

metalab uses your existing SSH configuration. If you can run `ssh user@gateway` without a password prompt, `metalab tunnel` works with zero additional config.

Under the hood, it spawns:

```
ssh -N -L local_port:127.0.0.1:remote_port [-J user@gateway] [user@]remote_host
```

This leverages your `~/.ssh/config`, SSH agent, and any keys already loaded.

### Setting up SSH key authentication

If you have not already configured key-based authentication for your cluster:

1. **Generate a key** (if you don't have one):

    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```

2. **Copy it to the remote host**:

    ```bash
    ssh-copy-id researcher@hpc.university.edu
    ```

3. **Verify** (should not prompt for a password):

    ```bash
    ssh researcher@hpc.university.edu hostname
    ```

On macOS, the Keychain-backed SSH agent handles passphrase caching automatically. On Linux, ensure `ssh-agent` is running and your key is added (`ssh-add`).

### Explicit key overrides

If a specific environment requires a different key, set it in `.metalab.local.toml`:

```toml
[environments.slurm]
ssh_key = "~/.ssh/special_cluster_key"
```

This adds `-i ~/.ssh/special_cluster_key` to the SSH command.

### The `metalab tunnel` command

Open a tunnel to running services:

```bash
metalab tunnel --env slurm
```

```
Tunnel established: http://127.0.0.1:8000
Press Ctrl+C to close.
```

The tunnel reads `bundle.json` to determine the remote host and port, then forwards them to `localhost`. The process runs in the foreground until you press Ctrl+C.

!!! note
    Always prefer key-based authentication over passwords. Password-based SSH is not supported by `metalab tunnel`.

---

## Workflow Guides

### Local Development

The simplest setup -- services run as local subprocesses.

```bash
metalab services up --env local
```

```
Services started (local):
  atlas: 127.0.0.1:8000
```

Atlas is immediately available at [http://localhost:8000](http://localhost:8000), reading from the `file_root` defined in your local environment profile (e.g., `./runs`). No tunnel is needed.

### SLURM / HPC with PostgreSQL

Full-featured setup with a Postgres query index.

```bash
# Provision PostgreSQL and Atlas on a compute node
metalab services up --env slurm

# Open an SSH tunnel to access Atlas locally
metalab tunnel

# Atlas is now available at http://localhost:8000
# Run experiments -- they write to the shared filesystem and PG index

# When finished, clean up
metalab services down --env slurm
```

Or provision and tunnel in one step:

```bash
metalab services up --env slurm --tunnel
```

### SLURM / HPC with File Store Only

If you don't need PostgreSQL (e.g., small experiments where filesystem-only is sufficient), omit the `[services.postgres]` section from your config. Atlas will read the shared filesystem directly.

```bash
metalab services up --env slurm
metalab tunnel
# Atlas reads from file_root on the shared filesystem
```

---

## Executor Configuration

Executors (which run experiment tasks) can be created from configuration dicts using the `executor_from_config()` factory. This enables a clean split:

- **TOML** (`.metalab.toml`) defines infrastructure defaults -- partitions, walltime, memory.
- **YAML** (per-experiment) specifies only what varies per experiment -- worker counts, GPUs.

### `executor_from_config()`

```python
from metalab.executor.config import executor_from_config

# Create from type name and config dict
executor = executor_from_config("slurm", {
    "partition": "gpu",
    "time": "2:00:00",
    "memory": "16G",
    "gpus": 1,
})

# Local executor with multiple workers
executor = executor_from_config("local", {"workers": 4})

# Single-threaded (returns None, metalab runs in-process)
executor = executor_from_config("local", {"workers": 1})
```

### Supported executor types

| Type | Config class | What it creates |
|------|-------------|----------------|
| `local` | `LocalExecutorConfig` | `ProcessExecutor` (or `None` for serial) |
| `slurm` | `SlurmExecutorConfig` | `SlurmExecutor` with job array support |

### SLURM executor options

| Field | Default | Description |
|-------|---------|-------------|
| `partition` | `"default"` | SLURM partition |
| `time` | `"1:00:00"` | Walltime limit |
| `cpus` | `1` | CPUs per task |
| `memory` | `"4G"` | Memory per task |
| `gpus` | `0` | GPUs per task |
| `max_concurrent` | `None` | Max simultaneous array tasks |
| `modules` | `[]` | `module load` commands |
| `conda_env` | `None` | Conda environment to activate |
| `setup` | `[]` | Extra shell commands before execution |

### Simplified experiment config

With `executor_from_config`, experiment YAML files become concise:

```yaml
# experiment.yaml
executor:
  type: slurm
  gpus: 1
  time: "4:00:00"

store: discover
```

The executor inherits partition, memory, and other defaults from `.metalab.toml`, while the experiment only overrides what it needs.

---

## How Configuration Flows into `metalab.run()`

When you call `metalab.run()`, multiple configuration layers merge transparently:

```
.metalab.toml               shared infrastructure defaults
  └─ .metalab.local.toml    machine-specific overrides (gitignored)
      └─ experiment config   per-experiment overrides
          └─ metalab.run()   final resolution at runtime
```

### Store resolution

The `store` argument accepts several forms:

| Value | What happens |
|-------|-------------|
| `None` (default) | FileStore at `./experiments/{experiment_id}/` |
| `"runs/"` | FileStore at the given path |
| `"postgresql://..."` | PostgresStore (pass `file_root` for artifacts) |
| `"discover"` | Auto-detect from the nearest running service bundle |
| `StoreConfig` object | Used directly (auto-scoped to experiment) |

### Executor resolution

`metalab.resolve_executor(platform, overrides)` merges TOML defaults with per-experiment overrides:

1. Auto-discovers `.metalab.toml` (walks up from cwd)
2. Resolves the environment profile matching `platform`
3. Deep-merges TOML defaults with the `overrides` dict
4. Creates the executor via the plugin registry

```python
# .metalab.toml provides partition, time, memory, modules, conda_env...
# Your experiment only overrides what differs:
executor = metalab.resolve_executor("slurm", {"gpus": 1, "time": "4:00:00"})
```

If no `.metalab.toml` exists, only the overrides dict is used — your code still works without a project config file.

### Inspecting the resolved config

Use the CLI to verify what metalab will use before running experiments:

```bash
metalab env list           # List available environment profiles
metalab env show slurm     # Show fully merged config for a profile
```

When `metalab.run()` executes, it logs the resolved store, executor, and run counts at INFO level so you can confirm what was picked up.

---

## Teardown and Cleanup

### `metalab services down`

Stops all services tracked in the service bundle:

- **SLURM jobs** are cancelled via `scancel`.
- **Local processes** receive `SIGTERM`, then `SIGKILL` if they don't exit within 5 seconds.
- The `bundle.json` file is removed.

Services are stopped in reverse dependency order (Atlas before PostgreSQL) so that dependents are torn down before the services they rely on.

### Orphan detection

If a previous session was not cleanly shut down, `metalab services status` will detect stale bundles. If the services referenced in the bundle are unreachable, `metalab services up` will discard the stale bundle and provision fresh services.

```bash
# Check for orphaned services
metalab services status --env slurm

# If services show as unreachable, re-provision
metalab services up --env slurm
```

### Bundle location

The service bundle is stored at:

- `{file_root}/services/bundle.json` -- if `file_root` is set in the environment config.
- `~/.metalab/services/{env_name}/bundle.json` -- otherwise.

The bundle file has `0o600` permissions since it may contain credentials.
