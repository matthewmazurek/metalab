# Quickstart

This guide walks you through creating and running your first metalab experiment.

## Your First Experiment

Let's estimate the value of π using Monte Carlo simulation. Create a file called `my_experiment.py`:

```python
import metalab

@metalab.operation
def estimate_pi(params, seeds, capture):
    n = params["n_samples"]
    rng = seeds.numpy()
    x, y = rng.random(n), rng.random(n)
    pi_est = 4.0 * (x**2 + y**2 <= 1).mean()
    capture.metric("pi_estimate", pi_est)

exp = metalab.Experiment(
    name="pi_mc",
    version="0.1",
    context={},
    operation=estimate_pi,
    params=metalab.grid(n_samples=[1000, 10000, 100000]),
    seeds=metalab.seeds(base=42, replicates=3),
)

# Run and get results
handle = metalab.run(exp)
results = handle.result()
print(results.table())
```

Run it:

```bash
python my_experiment.py
```

You'll see progress as your experiment runs, then a summary table of results.

## Understanding the Components

Let's break down each part of the experiment.

### The `@operation` Decorator

```python
@metalab.operation
def estimate_pi(params, seeds, capture):
    ...
```

The `@operation` decorator marks a function as an experiment operation. Your operation function can declare any subset of the arguments that metalab provides (injected by name). In this quickstart we use:

- **`params`**: A dictionary of parameter values for this run
- **`seeds`**: A `SeedBundle` for controlled randomness
- **`capture`**: The capture interface for recording metrics and artifacts

If you need them, you can also request `context` and `runtime`. See [Operations](../api/operation.md) for details.

Operations should be pure functions of their inputs—all randomness must come from `seeds`, not global state.

### The `Experiment` Definition

```python
exp = metalab.Experiment(
    name="pi_mc",
    version="0.1",
    context={},
    operation=estimate_pi,
    params=metalab.grid(n_samples=[1000, 10000, 100000]),
    seeds=metalab.seeds(base=42, replicates=3),
)
```

An `Experiment` bundles together:

| Field | Purpose |
|-------|---------|
| `name` | Unique identifier for this experiment |
| `version` | Version string (helps track changes) |
| `context` | Input data specification (empty here, see [Key Concepts](../key-concepts.md)) |
| `operation` | The function to run |
| `params` | Parameter source defining the sweep |
| `seeds` | Seed plan for reproducible randomness |

### Parameter Grids

```python
params=metalab.grid(n_samples=[1000, 10000, 100000])
```

`metalab.grid()` creates a parameter grid. This example produces 3 parameter combinations:

- `{"n_samples": 1000}`
- `{"n_samples": 10000}`
- `{"n_samples": 100000}`

For multiple parameters, it generates the full Cartesian product:

```python
metalab.grid(
    n_samples=[1000, 10000],
    method=["basic", "stratified"]
)
# Produces 4 combinations (2 × 2)
```

### Seed Plans

```python
seeds=metalab.seeds(base=42, replicates=3)
```

`metalab.seeds()` creates a seed plan with:

- **`base`**: The base seed for reproducibility
- **`replicates`**: Number of independent runs per parameter combination

With 3 parameter values and 3 replicates, this experiment runs **9 times total** (3 × 3).

Inside your operation, get random number generators from the seed bundle:

```python
rng = seeds.numpy()      # NumPy Generator
rng = seeds.rng()        # Python random.Random instance
seed_int = seeds.seed()  # Raw integer seed
```

### Capturing Results

```python
capture.metric("pi_estimate", pi_est)
```

The `capture` interface records outputs from your operation:

- **`capture.metric(name, value)`**: Numeric metrics (stored in run record)
- **`capture.data(name, value)`**: Structured data (JSON-serializable or numpy arrays)
- **`capture.artifact(name, data, format)`**: Binary artifacts (images, models, etc.)
- **`capture.log(message)`**: Log messages for debugging

Metrics are lightweight and appear in result tables. Use `capture.data()` or `capture.artifact()` for larger outputs.

### Running the Experiment

```python
handle = metalab.run(exp)
results = handle.result()
```

`metalab.run()` returns a `RunHandle` immediately. Call `.result()` to block until completion and get a `Results` object.

You can also check progress without blocking:

```python
handle = metalab.run(exp)
print(handle.status)  # Check current status
# ... do other work ...
results = handle.result()  # Block when ready
```

## Viewing Results

### Quick Table View

```python
print(results.table())
```

Displays a formatted table of all runs with their parameters and metrics.

### DataFrame Export

For analysis in pandas:

```python
df = results.to_dataframe()
print(df)

# Filter and analyze
successful = df[df["status"] == "completed"]
print(successful.groupby("n_samples")["pi_estimate"].mean())
```

This requires the `pandas` extra:

```bash
uv add metalab[pandas]
```

### Accessing Individual Runs

```python
for run in results:
    print(f"Run {run.run_id}: π ≈ {run.metrics['pi_estimate']:.4f}")
    print(f"  Params: {run.params}")
    print(f"  Status: {run.status}")
```

## Specifying a Store Location

By default, results are stored in a temporary directory. To persist results:

```python
handle = metalab.run(exp, store="./runs/pi_experiment")
```

This creates a `FileStore` at the specified path. You can reload results later:

```python
results = metalab.load_results("./runs/pi_experiment")
```

## Key Features You Get Automatically

Without any extra configuration, metalab provides:

- **Deterministic run IDs**: Each run has a stable ID derived from experiment + params + seeds
- **Automatic deduplication**: Re-running skips already-completed runs
- **Resume support**: Interrupted experiments continue where they left off
- **Parallel execution**: Runs execute concurrently by default

## Next Steps

Now that you've run your first experiment, explore more:

- [Key Concepts](../key-concepts.md) — Context specs, parameter sources, and seed discipline
- [Execution](../execution.md) — Local executors and SLURM cluster runs
- [Storage](../storage.md) — FileStore, PostgreSQL, and data transfer
