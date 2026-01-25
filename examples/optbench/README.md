# Optimization Benchmark - metalab Stress Test

A comprehensive stress-test experiment for metalab that exercises multiple system capabilities simultaneously.

## Purpose

This experiment is designed to find breaking points in metalab by stressing:

1. **Combinatorial explosion** - Large parameter grids (up to millions of combinations)
2. **Artifact generation** - Multiple numpy arrays + JSON per run
3. **Stepped metrics** - Time-series data at every iteration
4. **Concurrent execution** - Many parallel workers hitting FileStore
5. **Memory pressure** - Varying problem dimensions (5 to 200)
6. **Context fingerprinting** - Non-trivial context specs
7. **Parameter resolution** - Derived/conditional parameters

## Usage

```bash
# Quick validation (32 runs, ~1 second)
uv run python examples/optbench/run.py --intensity light

# Reasonable stress test (1,152 runs, ~5-10 minutes)
uv run python examples/optbench/run.py --intensity medium --workers 8 --yes

# Heavy stress test (tens of thousands of runs)
uv run python examples/optbench/run.py --intensity heavy --workers 16 --yes

# Extreme (WARNING: may create millions of run configurations)
uv run python examples/optbench/run.py --intensity extreme --workers 32 --yes

# Targeted investigation of specific algorithm/problem
uv run python examples/optbench/run.py --targeted adam rosenbrock --replicates 20

# Random hyperparameter search
uv run python examples/optbench/run.py --random --n-trials 500 --workers 16
```

## Intensity Levels

| Intensity | Param Combos | Replicates | Total Runs | Estimated Time |
|-----------|-------------|------------|------------|----------------|
| light     | 16          | 2          | 32         | ~1 second      |
| medium    | 384         | 3          | 1,152      | ~5-10 minutes  |
| heavy     | ~4,000      | 5          | ~20,000    | ~1-2 hours     |
| extreme   | ~millions   | 10         | millions+  | ∞ (grid explosion) |

## Discovered Breaking Points

### 1. Grid Explosion (Critical)

The Cartesian product of all parameters creates exponential growth even when many parameters don't apply to all algorithms:

```python
# Example: 5 algorithms × 5 problems × 5 dims × 5 lr × 5 momentum × ...
# Each added param multiplies the total, even if it only applies to 1 algorithm
```

**Impact**: The "extreme" intensity creates so many combinations that even counting them times out.

**Recommendation**: For future versions, consider:
- Algorithm-specific parameter grids (compose, not product)
- Conditional parameters that are only included when relevant
- A `grid_union()` helper for combining algorithm-specific grids

### 2. Execution Speed at Scale

Observed performance:
- **~35ms average** per run (light intensity, simple operations)
- **~1.1s average** per run (medium intensity, including overhead)
- Linear scaling with run count

At 1000+ runs, even fast individual runs accumulate significant total time.

**Recommendation**: 
- The current ThreadExecutor is well-suited for CPU-bound tasks
- For I/O-bound or heavily parallel workloads, ProcessExecutor may help
- Consider async artifact writing to reduce blocking

### 3. Artifact Storage Warnings

When multiple workers complete runs simultaneously, artifact files with the same name (e.g., `convergence_curve.npz`) trigger "already exists, overwriting" warnings. This appears to be a race condition in the artifact naming.

**Root cause**: The artifact path uses just the artifact name, not a unique identifier.

**Recommendation**: Include `run_id` or `artifact_id` in the filename to ensure uniqueness.

### 4. Resume Behavior

The resume feature works correctly:
- Skips runs with existing successful records
- Re-runs failed runs
- Handles partial completion gracefully

**Observation**: With 560/1152 runs completed before timeout, re-running with `--resume` correctly skips completed runs.

### 5. FileStore Locking

The per-run `flock()` locking works correctly under concurrent load. No corruption or deadlocks observed with 8 parallel workers.

## Benchmark Components

### Test Functions

| Function   | Characteristics | Difficulty |
|------------|-----------------|------------|
| sphere     | Convex, smooth  | Easy       |
| rosenbrock | Narrow valley   | Medium     |
| ackley     | Many local min  | Hard       |
| rastrigin  | Highly multimodal| Hard      |
| griewank   | Widespread local| Medium     |

### Optimization Algorithms

| Algorithm | Type | Key Params |
|-----------|------|------------|
| gradient_descent | First-order | lr, momentum, lr_decay |
| adam | Adaptive first-order | lr, beta1, beta2 |
| simulated_annealing | Metaheuristic | t_initial, cooling, step_size |
| random_search | Derivative-free | strategy, shrink_factor |
| evolution_strategy | Population-based | pop_size, elite_frac, sigma |

## Artifacts Generated Per Run

1. `convergence_curve.npz` - Function values at each iteration
2. `best_solution.npz` - Final solution vector and value
3. `trajectory.npz` - Subsampled position history
4. `summary.json` - Run configuration and results
5. `report.txt` - Human-readable summary

## Example Analysis

```python
import metalab
from examples.optbench.experiment import build_experiment

# Load results
store = metalab.FileStore("./runs/optbench")
records = store.list_run_records()

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame([
    {
        "run_id": r.run_id,
        "status": r.status.value,
        **r.metrics,
    }
    for r in records
])

# Analyze by algorithm
print(df.groupby("algorithm")["best_f"].agg(["mean", "std", "min"]))

# Find best configuration
best_idx = df[df["status"] == "success"]["best_f"].idxmin()
print(df.loc[best_idx])
```

## Future Stress Tests

To further test metalab, consider:

1. **Memory stress**: Generate very large artifacts (GB-scale arrays)
2. **Failure injection**: Introduce random failures to test error handling
3. **Long-running operations**: Multi-hour optimization runs
4. **Rapid-fire metrics**: Capture thousands of stepped metrics per run
5. **Nested experiments**: Experiment that spawns sub-experiments
