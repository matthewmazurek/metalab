# metalab

A general experiment runner: `(FrozenContext, Params, SeedBundle) -> RunRecord + Artifacts`

metalab is a lightweight, backend-agnostic framework for running reproducible experiments. Define your experiment logic once, sweep parameters, control randomness, and capture results—all with built-in support for resume, deduplication, and parallel execution.

## Installation

```bash
pip install metalab

# With optional dependencies
pip install metalab[numpy]   # NumPy array serialization
pip install metalab[pandas]  # DataFrame export
pip install metalab[rich]    # Progress bars
pip install metalab[full]    # All of the above
```

## Minimal Working Example

Estimate π using Monte Carlo sampling across different sample sizes with multiple replicates:

```python
import metalab

@metalab.operation(name="pi_mc", version="0.1")
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

result = metalab.run(exp)  # Stores results in ./runs/pi_mc
print(result.table())
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
@metalab.operation(name="my_op")
def my_operation(seeds):
    # Get a NumPy random generator
    rng = seeds.numpy()
    
    # Or a standard library random.Random
    rng = seeds.rng()
    
    # Derive sub-seeds for different components
    model_seed = seeds.derive("model")
    data_seed = seeds.derive("data_split")
    ...
```

Specify replicates when defining the experiment:

```python
seeds = metalab.seeds(base=42, replicates=5)  # 5 independent runs per param config
```

### Capture System

Record metrics, artifacts, and logs during execution:

```python
@metalab.operation(name="train")
def train(params, seeds, capture):
    # Scalar metrics
    capture.metric("accuracy", 0.95)
    capture.metric("loss", 0.05)
    
    # Multiple metrics at once
    capture.metrics({"precision": 0.92, "recall": 0.89})
    
    # Time-series metrics
    for epoch, loss in enumerate(losses):
        capture.metric("train_loss", loss, step=epoch)
    
    # Artifacts (auto-serialized)
    capture.artifact("predictions", predictions_array)  # NumPy array
    capture.artifact("config", {"model": "resnet", "layers": 50})  # JSON
    
    # Matplotlib figures
    capture.figure("learning_curve", fig)
    
    # Log messages
    capture.log("Training completed successfully")
```

### Results API

Query and analyze results:

```python
result = metalab.run(exp)

# Access individual runs
run = result[0]
print(run.metrics)
artifact = run.artifact("predictions")

# Tabular view
df = result.table(as_dataframe=True)

# Filter runs
successful = result.successful
filtered = result.filter(learning_rate=0.01)

# Summary and display
result.display(group_by=["learning_rate"])

# Export
result.to_csv("results.csv")

# Load previous results
old_results = metalab.load_results("./runs/my_exp")
```

### Resume and Deduplication

Resume interrupted experiments—completed runs are automatically skipped:

```python
# First run: executes all 100 configurations
result = metalab.run(exp, resume=True)

# Second run: skips completed, only runs new/failed
result = metalab.run(exp, resume=True)
```

Run IDs are stable hashes derived from experiment + context + params + seed fingerprints, enabling reliable deduplication.

### Progress Display

Track execution with rich progress bars (requires `rich`):

```python
result = metalab.run(exp, progress=True)
```

## Advanced Usage

### Custom Context

Share read-only data across runs with context specs:

```python
@metalab.context_spec
class DataContext:
    dataset_path: str
    vocab_size: int = 10000

exp = metalab.Experiment(
    name="nlp_exp",
    context=DataContext(dataset_path="data/train.csv"),
    operation=my_operation,
    params=metalab.grid(hidden=[64, 128]),
    seeds=metalab.seeds(base=42),
)

# Fingerprint is auto-computed
print(exp.context.fingerprint)
```

### Parallel Execution

Use process-based parallelism for CPU-bound work:

```python
from metalab import ProcessExecutor

result = metalab.run(
    exp,
    executor=ProcessExecutor(max_workers=4),
)
```

### Custom Storage

Store results in a custom location:

```python
result = metalab.run(exp, store="./my_experiments/run_001")

# Load previous results
result = metalab.load_results("./my_experiments/run_001")
```

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
