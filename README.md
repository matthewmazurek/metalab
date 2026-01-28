# metalab

A general experiment runner: `(Context, Params, SeedBundle) -> RunRecord + Artifacts`

metalab is a lightweight, backend-agnostic framework for running reproducible experiments. Define your experiment logic once, sweep parameters, control randomness, and capture results—all with built-in support for resume, deduplication, and parallel execution.

## Installation

```bash
uv add metalab

# With optional dependencies
uv add metalab[numpy]   # NumPy array serialization
uv add metalab[pandas]  # DataFrame export
uv add metalab[rich]    # Progress bars
uv add metalab[slurm]   # SLURM cluster execution
uv add metalab[full]    # All of the above
```

## Minimal Working Example

Estimate π using Monte Carlo sampling across different sample sizes with multiple replicates:

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

handle = metalab.run(exp)  # Returns a RunHandle, stores results in ./runs/pi_mc
results = handle.result()  # Block until complete
print(results.table())
```

This runs 9 experiments (3 sample sizes × 3 replicates) with deterministic random seeds.

## Key Features

### Parameter Sources

Sweep parameters using grid search, random sampling, or explicit lists:

```python
# Grid search (Cartesian product)
params = metalab.grid(
    learning_rate=[0.001, 0.01, 0.1],
    batch_size=[32, 64],
)  # 6 combinations

# Random sampling with distributions
params = metalab.random(
    space={
        "learning_rate": metalab.loguniform(1e-4, 1e-1),
        "dropout": metalab.uniform(0.1, 0.5),
        "hidden_dim": metalab.choice([64, 128, 256]),
    },
    n_trials=20,
    seed=123,
)

# Explicit parameter list
params = metalab.manual([
    {"lr": 0.01, "epochs": 10},
    {"lr": 0.001, "epochs": 50},
])
```

### Reproducible Seeds

Control all randomness through `SeedBundle`:

```python
@metalab.operation
def my_operation(seeds, capture):
    # Get a NumPy random generator
    rng = seeds.numpy()
    
    # Or a standard library random.Random
    rng = seeds.rng()
    
    # Derive sub-seeds for different components
    model_seed = seeds.derive("model")
    data_seed = seeds.derive("data_split")
    
    capture.metric("result", rng.random())
```

Specify replicates when defining the experiment:

```python
seeds = metalab.seeds(base=42, replicates=5)  # 5 independent runs per param config
```

### Capture System

Record metrics, artifacts, and logs during execution:

```python
@metalab.operation
def train(params, seeds, capture):
    capture.log("Starting training")
    
    # Scalar metrics
    capture.metric("accuracy", 0.95)
    capture.metric("loss", 0.05)
    
    # Multiple metrics at once
    capture.metrics({"precision": 0.92, "recall": 0.89})
    
    # Time-series metrics
    for epoch, loss in enumerate(losses):
        capture.metric("train_loss", loss, step=epoch)
        capture.log(f"Epoch {epoch}: loss={loss:.4f}")
    
    # Artifacts (auto-serialized)
    capture.artifact("predictions", predictions_array)  # NumPy array
    capture.artifact("config", {"model": "resnet", "layers": 50})  # JSON
    
    # Matplotlib figures
    capture.figure("learning_curve", fig)
    
    capture.log("Training completed")
    # No return needed - success is implicit
```

### Logging

Use `capture.log()` for operation logging. Messages include timestamps, log levels, and worker identification, and are saved to the run's log directory.

```python
@metalab.operation
def train(params, seeds, capture):
    capture.log("Starting training")
    capture.log(f"Parameters: lr={params['lr']}", level="debug")
    
    for epoch in range(params['epochs']):
        loss = train_epoch(...)
        capture.log(f"Epoch {epoch}: loss={loss:.4f}")
        capture.metric("loss", loss, step=epoch)
    
    capture.log("Training complete")
    if loss > threshold:
        capture.log("Loss above threshold", level="warning")
```

For advanced logging (custom handlers, stream routing), use Python's standard `logging` module directly:

```python
import logging
logger = logging.getLogger(__name__)

@metalab.operation
def my_operation(params, seeds, capture):
    # Standard Python logging (goes to console/configured handlers)
    logger.info("This uses standard Python logging")
    
    # capture.log() is stored with the run
    capture.log("This is saved to the run's log file")
```

### Results API

Query and analyze results:

```python
handle = metalab.run(exp)

# Check status without blocking
print(handle.status)       # RunStatus(total=9, completed=3, running=6, ...)
print(handle.is_complete)  # False

# Block until complete and get results
results = handle.result()

# Access individual runs
run = results[0]
print(run.params)   # Resolved parameters
print(run.metrics)  # Captured metrics
artifact = run.artifact("predictions")

# Tabular view
df = results.table(as_dataframe=True)

# Filter runs
successful = results.successful
filtered = results.filter(learning_rate=0.01)

# Summary and display
results.display(group_by=["learning_rate"])

# Export
results.to_csv("results.csv")

# Load previous results
old_results = metalab.load_results("./runs/my_exp")
```

### DataFrame Export with Artifact Reducers

Export results to pandas with artifact-derived columns using reducer functions:

```python
# Define reducers to extract scalars from artifacts
def reduce_loss_history(arr):
    """Simple reducer: artifact -> dict of scalars."""
    return {"final_loss": float(arr[-1]), "best_loss": float(arr.min())}

def context_reducer(arr, run):
    """Context-aware reducer: can access run.params."""
    threshold = run.params.get("threshold", 0.5)
    return {"above_threshold": float((arr > threshold).mean())}

# Export to DataFrame with reduced artifacts
df = results.to_dataframe(
    include_params=True,      # Columns prefixed with 'param_'
    include_metrics=True,     # Captured metrics
    include_record=True,      # run_id, status, duration, etc.
    artifact_reducers={
        "loss_history": reduce_loss_history,
        "predictions": context_reducer,
    },
)

# Now use pandas for analysis
summary = df.groupby("param_learning_rate").agg({
    "final_loss": ["mean", "std"],
    "above_threshold": "mean",
})
```

Reducers with 1 argument receive just the artifact; reducers with 2 arguments also receive the `Run` object for context-aware computation.

### Resume and Deduplication

Resume interrupted experiments—completed runs are automatically skipped:

```python
# First run: executes all 100 configurations
results = metalab.run(exp, resume=True).results()

# Second run: skips completed, only runs new/failed
results = metalab.run(exp, resume=True).results()
```

Run IDs are stable hashes derived from experiment + context + params + seed fingerprints, enabling reliable deduplication.

### Progress Tracking

Monitor experiment execution with live progress display:

```python
# Simple progress bar (auto-detects rich, falls back to text)
handle = metalab.run(exp, progress=True)
results = handle.result()  # Shows live progress

# Customized progress display
handle = metalab.run(
    exp,
    progress=metalab.Progress(
        title="Training Models",
        display_metrics=["loss:.4f", "accuracy:.2%"],
    ),
)
results = handle.result()
```

For manual status checking without a progress display:

```python
handle = metalab.run(exp)
print(handle.status)       # RunStatus(total=9, completed=3, running=6, ...)
print(handle.is_complete)  # False
results = handle.result()  # Block until complete
```

## Advanced Usage

### Context Specs

Share configuration across runs with context specs. The context is a lightweight, serializable manifest that operations receive directly:

```python
@metalab.context_spec
class DataContext:
    dataset: metalab.FilePath  # Hash computed lazily at run() time
    vocab_size: int = 10000

@metalab.operation
def nlp_operation(context, params, capture):
    # Operations receive the context spec directly
    # Load data yourself using paths from the context
    data = load_data(str(context.dataset))
    capture.metric("vocab_size", context.vocab_size)

exp = metalab.Experiment(
    name="nlp_exp",
    version="0.1",
    context=DataContext(
        dataset=metalab.FilePath("data/train.csv"),
        vocab_size=10000,
    ),
    operation=nlp_operation,
    params=metalab.grid(hidden=[64, 128]),
    seeds=metalab.seeds(base=42),
)
```

**Key points:**
- Context specs are lightweight manifests (paths, config, parameters)
- Use `metalab.FilePath` for files and `metalab.DirPath` for directories—hashes are computed lazily at `run()` time
- Operations load data themselves using paths from the spec
- Metadata-based hashing (size, mtime, inode) is O(1) regardless of file size
- Each run gets a fresh copy of any mutable data (no shared state issues)

### Data Preprocessing

For experiments with expensive preprocessing, run it explicitly before the experiment. The `FilePath` hash is computed lazily at `run()` time, so the file doesn't need to exist when creating the spec:

```python
from pathlib import Path
import scanpy as sc

@metalab.context_spec
class SingleCellSpec:
    data: metalab.FilePath  # Hash computed at run() time
    min_genes: int = 200
    n_hvg: int = 2000

def preprocess_data(raw_path: str, spec: SingleCellSpec):
    """Run this once before the experiment."""
    cache_path = str(spec.data)  # FilePath supports str() and os.fspath()
    if not Path(cache_path).exists():
        adata = sc.read_10x_mtx(raw_path)
        sc.pp.filter_cells(adata, min_genes=spec.min_genes)
        sc.pp.highly_variable_genes(adata, n_top_genes=spec.n_hvg)
        adata.write_h5ad(cache_path)

@metalab.operation
def analyze(context, params, capture):
    # Load preprocessed data (each run gets its own copy)
    adata = sc.read_h5ad(str(context.data))
    sc.pp.neighbors(adata, n_neighbors=params["n_neighbors"])
    # ... analysis ...

# Create spec (file doesn't need to exist yet)
spec = SingleCellSpec(
    data=metalab.FilePath("./cache/adata.h5ad"),
    min_genes=200,
)

# Preprocess explicitly (run once, before experiment)
preprocess_data("data/raw/", spec)

# Now run experiments - hash computed here, workers load from cached file
exp = metalab.Experiment(
    name="single_cell_exp",
    version="0.1",
    context=spec,
    operation=analyze,
    params=metalab.grid(n_neighbors=[10, 15, 30]),
    seeds=metalab.seeds(base=42),
)
```

**HPC/SLURM**: Run preprocessing interactively on the login node. Workers load from the cached file on shared storage.

### Seeded Preprocessing

If your preprocessing requires randomness (e.g., train/test splits, data augmentation), use `SeedBundle.for_preprocessing()` to get reproducible RNG. Use the same base seed for both preprocessing and the experiment, and include it in your context spec so it becomes part of the fingerprint:

```python
from metalab.seeds import SeedBundle

BASE_SEED = 42

@metalab.context_spec
class MyContext:
    raw_data: metalab.FilePath
    processed_data: metalab.FilePath
    base_seed: int  # Included in context fingerprint

def preprocess(raw_path: str, output_path: str, base_seed: int):
    """Preprocessing with reproducible randomness."""
    # Create a SeedBundle using the same API as inside operations
    seeds = SeedBundle.for_preprocessing(base_seed)
    
    data = load_data(raw_path)
    
    # Use derived RNGs for different preprocessing steps
    rng_split = seeds.numpy("train_test_split")
    train, test = train_test_split(data, rng=rng_split)
    
    rng_aug = seeds.numpy("augmentation")
    train_aug = augment(train, rng=rng_aug)
    
    save(output_path, {"train": train_aug, "test": test})

# Wire it together with one base seed
spec = MyContext(
    raw_data=metalab.FilePath("./data/raw.csv"),
    processed_data=metalab.FilePath("./cache/processed.h5ad"),
    base_seed=BASE_SEED,
)

# Preprocess with seeded RNG
preprocess(str(spec.raw_data), str(spec.processed_data), BASE_SEED)

# Experiment uses same base seed - all randomness shares one hierarchy
exp = metalab.Experiment(
    name="my_exp",
    version="0.1",
    context=spec,
    operation=train_model,
    params=metalab.grid(lr=[0.01, 0.001]),
    seeds=metalab.seeds(base=BASE_SEED, replicates=5),
)
```

**Key points:**
- One base seed controls everything: preprocessing and experiment runs
- `SeedBundle.for_preprocessing(base_seed)` derives a preprocessing-specific seed that won't collide with replicate seeds
- Include `base_seed` in your context spec so changing it changes the context fingerprint (triggering re-runs)
- The preprocessing bundle supports the same interface as operation bundles: `.numpy()`, `.rng()`, `.derive()`

### Parallel Execution

Use process-based parallelism for CPU-bound work:

```python
from metalab import ProcessExecutor

handle = metalab.run(exp, executor=ProcessExecutor(max_workers=4))
results = handle.result()
```

### SLURM Cluster Execution

Submit experiments to a SLURM cluster (requires `submitit`):

```python
handle = metalab.run(
    exp,
    store="/scratch/runs/my_exp",  # Shared filesystem path
    executor=metalab.SlurmExecutor(
        metalab.SlurmConfig(
            partition="gpu",
            time="2:00:00",
            cpus=4,
            memory="16G",
            gpus=1,
        )
    ),
    progress=True,  # Watch job progress
)

# Block until all jobs complete (shows live progress)
results = handle.result()
```

**Reconnecting to SLURM jobs**: If you disconnect (e.g., close your terminal), you can reconnect later:

```python
# In a new session, reconnect to watch progress
handle = metalab.reconnect("/scratch/runs/my_exp", progress=True)
results = handle.result()

# Or just check status
handle = metalab.reconnect("/scratch/runs/my_exp")
print(handle.status)  # RunStatus(total=100, completed=45, ...)
```

The `SlurmExecutor` uses `submitit` under the hood to handle job array submission, serialization, and result collection. Results are written directly to the shared filesystem store.

### Custom Storage

Store results in a custom location:

```python
handle = metalab.run(exp, store="./my_experiments/run_001")
results = handle.result()

# Load previous results
results = metalab.load_results("./my_experiments/run_001")
```

### Experiment Manifests

When you run an experiment, metalab automatically saves an experiment manifest capturing the full configuration:

```
{store}/experiments/{experiment_id}_{timestamp}.json
```

The manifest includes:
- Experiment metadata (name, version, description, tags)
- Operation reference and code hash
- Full parameter source specification (e.g., grid values, random space)
- Seed plan (base seed, number of replicates)
- Context fingerprint
- Total runs and submission timestamp

This complements the per-run `RunRecord` (which captures individual results) by documenting the overall experiment design. Multiple runs create multiple timestamped manifests, enabling version tracking.

## Development

Requirements: Python 3.11+

```bash
# Install with uv
uv sync

# Run tests
uv run pytest
```

## License

MIT
