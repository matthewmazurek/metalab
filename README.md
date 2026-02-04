# metalab

A general experiment runner: `(ContextSpec, Params, SeedBundle) -> RunRecord + Artifacts`.

metalab is a lightweight, backend-agnostic framework for reproducible experiments. Define your operation once, sweep parameters, control randomness, and capture results with built-in resume, deduplication, and parallel execution (local or SLURM).

## Installation

```bash
uv add metalab

# Optional extras
uv add metalab[numpy]    # Array serialization for capture.data()/capture.artifact()
uv add metalab[pandas]   # DataFrame export helpers (results.to_dataframe / to_csv)
uv add metalab[rich]     # Rich progress bars and nicer CLI output
uv add metalab[postgres] # PostgreSQL store backend for large experiments
uv add metalab[full]     # Installs numpy + pandas + rich + postgres
```

## Getting Started (Minimal Project)

Save this as `mvp.py`:

```python
import metalab

@metalab.operation
def train(params, seeds, capture):
    rng = seeds.rng()
    score = rng.random() * params["scale"]
    capture.metric("score", score)

exp = metalab.Experiment(
    name="mvp",
    version="0.1",
    context={},
    operation=train,
    params=metalab.grid(scale=[0.1, 1.0, 10.0]),
    seeds=metalab.seeds(base=123, replicates=2),
)

results = metalab.run(exp).result()
print(results.to_dataframe())
```

Run it:

```bash
python mvp.py
```

What you get:
- Deterministic run IDs from experiment + params + seeds
- 6 runs (3 parameter values x 2 replicates)
- Metrics captured in each run record
- A quick table or DataFrame for analysis

## Key Features (Quick Tour)

- **Context specs**: lightweight manifests with lazy hashing (`FilePath`/`DirPath`)
- **Parameter sources**: grid, random, or manual
- **Seed discipline**: `SeedBundle` with replicates and derived RNGs
- **Capture system**: metrics, structured data, artifacts, logs
- **Resume + dedupe**: stable run IDs skip completed work
- **Parallel + SLURM**: local executors and cluster runs
- **Stores**: filesystem by default, PostgreSQL for query acceleration

## Storage

By default, metalab stores everything on the filesystem:

```python
metalab.run(exp, store="./runs/my_exp")
```

For large-scale experiments with many runs, add PostgreSQL for fast queries:

```python
# PostgresStore = FileStore (source of truth) + PostgresIndex (fast queries)
metalab.run(
    exp,
    store="postgresql://localhost/db?file_root=/shared/experiments",
)
```

Files remain the source of truth—Postgres accelerates lookups. If the database is lost, rebuild the index from files:

```python
from metalab.store import PostgresStore

store = PostgresStore("postgresql://localhost/db", file_root="/shared/experiments")
store.rebuild_index()  # Restores index from files
```

See [Storage](docs/storage.md) for details on store architecture and data transfer.

## Learn More

- [Key Concepts](docs/key-concepts.md)
- [Execution (Local + SLURM)](docs/execution.md)
- [Storage (FileStore + Postgres)](docs/storage.md)
- [Remote Analysis Workflow](docs/remote-analysis-workflow.md)

## Development

Python 3.11+ required.

```bash
uv sync
uv run pytest
```

## License

MIT
