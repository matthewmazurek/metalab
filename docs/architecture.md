# Architecture

An overview of how metalab is structured internally: the core abstractions, plugin system, orchestration flow, and design patterns that tie everything together.

## Central Contract

Everything in metalab revolves around a single idea:

```
(ContextSpec, Params, SeedBundle) → RunRecord + Artifacts
```

Domain logic lives entirely in user-defined **Operations**. Everything else—execution, storage, services—is pluggable.

## Layers at a Glance

| Layer | Key Types | Purpose |
| --- | --- | --- |
| **Definition** | `Experiment`, `ContextSpec`, `ParamSource`, `SeedPlan` | Declarative experiment specification |
| **Execution** | `Executor`, `RunPayload`, `RunHandle` | Where and how work runs (threads, processes, SLURM) |
| **Operation** | `OperationWrapper`, `Capture`, `Runtime` | User computation + injected plumbing |
| **Storage** | `Store`, `StoreConfig`, `ArtifactDescriptor` | Persisting run records, artifacts, and logs |
| **Results** | `Results`, `Run`, `DerivedMetricFn` | Querying and post-processing completed runs |
| **Services** | `ServicePlugin`, `ServiceEnvironment`, `ServiceOrchestrator` | Managing infrastructure (Postgres, Atlas) |

## Orchestration Flow

The entry point is `metalab.run(experiment, store=..., executor=...)`. Here is what happens under the hood:

### 1. Resolve store

The `store` argument (a path string, URI, or `StoreConfig`) is parsed via `parse_to_config()`, scoped to the experiment, and connected:

```
"./experiments" → FileStoreConfig(root="./experiments")
                    → .scoped("my_exp:1.0")
                    → .connect() → FileStore
```

### 2. Resolve executor

Defaults to `ThreadExecutor(max_workers=1)`. When a SLURM environment is configured, the executor is resolved from project config instead.

### 3. Generate payloads

The runner iterates the cartesian product of **params × seeds** and builds a `RunPayload` for each combination:

- Resolves context (computes lazy file hashes for `FilePath`/`DirPath`)
- Computes a deterministic `run_id` from `sha256(experiment_id + context_fp + params_fp + seed_fp + code_hash)`
- Skips runs that already exist as `SUCCESS` in the store (resume / dedupe)
- Writes an experiment manifest for Atlas

### 4. Submit

```python
executor.submit(payloads, store, operation) → RunHandle
```

### 5. Execute (inside worker)

Each payload is executed in `executor/core.py`:

1. Create `Runtime` (logger, scratch directory, cancel token)
2. Create `Capture` (metric / artifact / log emission)
3. Write a `RUNNING` record to the store (crash resilience)
4. Call `operation.run(context, params, seeds, runtime, capture)`
5. On success — build final `RunRecord`, compute derived metrics, persist
6. On failure — build failed `RunRecord`, persist error

### 6. Collect results

```python
handle.result() → Results
```

Blocks until all runs complete, then returns a queryable `Results` object.

## Plugin System

metalab uses **three independent plugin registries**, all discovered via Python entry points in `pyproject.toml` and loaded lazily on first access.

### Executor plugins (`metalab.executors`)

| Entry point | Config class |
| --- | --- |
| `local` | `LocalExecutorConfig` |
| `slurm` | `SlurmExecutorConfig` |

Each `ExecutorConfig` subclass provides:

- `create() → Executor` — instantiate the executor
- `from_dict()` / `to_dict()` — serialization
- `handle_class()` — optional, for reconnection support

### Store plugins (`metalab.stores`)

| Entry point | Config class |
| --- | --- |
| `file` | `FileStoreConfig` |
| `postgresql` | `PostgresStoreConfig` |

Each `StoreConfig` subclass provides:

- `connect() → Store` — create a connected store instance
- `scoped(experiment_id)` — return a new config scoped to an experiment
- `from_locator(info)` — parse from a URI

### Service plugins (`metalab.service_plugins`)

| Entry point | Plugin class |
| --- | --- |
| `postgres` | `PostgresPlugin` |
| `atlas` | `AtlasPlugin` |

Each `ServicePlugin` subclass dispatches via `plan(spec, env_type)` to environment-specific methods (`plan_slurm`, `plan_local`), returning platform-specific fragments the environment knows how to execute.

### Environment registry

A fourth registry maps environment types to `ServiceEnvironment` implementations via import-time registration:

| Type | Implementation |
| --- | --- |
| `local` | `LocalEnvironment` |
| `slurm` | `SlurmEnvironment` |

### Adding a plugin

All entry-point registries follow the same pattern: **Config (pure data, serializable) → Factory method → Instance**.

```toml
# pyproject.toml
[project.entry-points."metalab.stores"]
myscheme = "my_package.store:MyStoreConfig"
```

Once the entry point is installed, `create_store("myscheme://...")` discovers and loads it automatically.

## Protocol-Based Design

All major abstractions are defined as [`typing.Protocol`][typing.Protocol] classes—structural subtyping with no inheritance required. This keeps implementations fully decoupled from the core:

- `Executor` — submit payloads, get handles
- `Store` — persist and retrieve run records and artifacts
- `RunHandle` — track running work, cancel, collect results
- `ParamSource` — iterate parameter cases
- `Serializer` — encode and decode artifacts
- `ServiceEnvironment` — start, stop, inspect services
- `Connector` — establish tunnels to remote services

### Capability protocols

Not every store supports every feature. Rather than `hasattr()` checks, metalab uses `@runtime_checkable` capability protocols defined in `metalab.store.capabilities`:

| Protocol | Meaning |
| --- | --- |
| `SupportsWorkingDirectory` | Has a local filesystem root |
| `SupportsArtifactOpen` | Can open artifacts for reading |
| `SupportsLogPath` | Can provide filesystem paths for streaming logs |
| `SupportsStructuredResults` | Supports inline structured data queries |
| `SupportsLogListing` | Can list and retrieve log files |
| `SupportsExperimentManifests` | Stores versioned experiment manifests |

Code checks capabilities with `isinstance()`:

```python
if isinstance(store, SupportsWorkingDirectory):
    path = store.get_working_directory()
```

## Services Layer

Services like Postgres and Atlas are infrastructure that experiments depend on. They are managed through a layered stack:

```
CLI (metalab services up / down / status)
  └── ServiceOrchestrator          config-driven, service-agnostic
        ├── ServiceEnvironment     LocalEnvironment | SlurmEnvironment
        ├── ServicePlugin.plan()   returns platform-specific fragments
        └── ServiceBundle          persisted state (~/.metalab/services/)
```

1. The **orchestrator** reads project config to determine which services are needed
2. Each **plugin** produces platform-specific fragments (bash scripts, subprocess commands)
3. The **environment** composes fragments into jobs and manages their lifecycle
4. Service handles are persisted in a **bundle** on disk

The bundle's `store_locator` auto-wires the Postgres connection as the default store via a `"discover"` URI scheme, so experiments connect without hardcoded connection strings.

## Configuration

Project configuration lives in `.metalab.toml`:

```toml
[project]
name = "myproject"
default_env = "local"

[environments.local]
type = "local"

[environments.slurm]
type = "slurm"
gateway = "cluster.example.com"
[environments.slurm.executor]
partition = "gpu"
time = "2:00:00"

[services.postgres]
database = "metalab"

[services.atlas]
port = 8000
```

`ProjectConfig.resolve(env_name) → ResolvedConfig` merges base sections, named profiles, and `.metalab.local.toml` overrides into a single flat config that drives both the runner and the service orchestrator.

## Design Patterns

**Fingerprint-based identity**
:   `run_id = sha256(experiment_id + context + params + seed + code)` enables deterministic resume and deduplication. Derived metrics are excluded from fingerprints.

**Config → Instance separation**
:   Configs are pure frozen dataclasses, serializable across process boundaries. Instances are stateful connections created via `.connect()` or `.create()`.

**Lazy imports**
:   Optional dependencies (submitit, psycopg, rich) are deferred via module-level `__getattr__`, so the core stays lightweight.

**Signature inspection**
:   The `@operation` decorator wraps user functions and only injects the arguments they declare (`context`, `params`, `seeds`, `runtime`, `capture`).

**Index-addressed SLURM arrays**
:   Instead of pickling per task, a single manifest is written. Each SLURM array task reconstructs its payload from `SLURM_ARRAY_TASK_ID` using O(1) index access into the parameter source.
