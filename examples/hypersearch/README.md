# Hyperparameter Search Example

This example demonstrates **random parameter sampling**, **resume capability**, and **result analysis** by comparing grid search vs random search on a classic optimization problem.

## Features Demonstrated

| Feature | Description |
|---------|-------------|
| `metalab.random()` | Random sampling with distributions |
| `loguniform()`, `uniform()` | Continuous parameter distributions |
| `resume=True` | Skip completed runs on re-execution |
| `results.filter()` | Filter runs by criteria |
| `results.display()` | Grouped result summaries |
| Multiple experiments | Compare approaches in one script |

## Running the Example

```bash
uv run python examples/hypersearch/run.py
```

## Code Highlights

### Random Parameter Sampling

Use distributions for continuous hyperparameter search:

```python
params = metalab.random(
    space={
        "lr": metalab.loguniform(1e-4, 1e-1),    # Log-uniform for learning rates
        "momentum": metalab.uniform(0.0, 0.99),   # Uniform for bounded params
    },
    n_trials=30,
    seed=123,  # Reproducible sampling
)
```

### Resume Capability

Continue interrupted experiments without re-running completed work:

```python
# First run: executes all runs
handle = metalab.run(exp, resume=True)

# Second run: skips completed, only runs new/failed
handle = metalab.run(exp, resume=True)
```

### Result Filtering and Analysis

```python
# Filter successful runs
successful = results.filter(status="success")

# Grouped display
results.display(group_by=["search_type"])

# Export to pandas DataFrame
df = results.table(as_dataframe=True)
```

### Multiple Experiments

Compare grid and random search in one script:

```python
grid_exp = metalab.Experiment(
    name="hypersearch_grid",
    params=metalab.grid(lr=[0.001, 0.01, 0.1], momentum=[0.0, 0.5, 0.9]),
    ...
)

random_exp = metalab.Experiment(
    name="hypersearch_random",
    params=metalab.random(space={...}, n_trials=30),
    ...
)
```

## Atlas Visualization

In metalab-atlas, try these visualizations:

1. **Scatter Plot**: `metrics.final_value` vs `params.lr` to see parameter sensitivity
2. **Grid vs Random**: Compare efficiency of both search strategies
3. **Filtering**: Filter by `status='success'` to exclude failures
4. **Aggregation**: Mean `final_value` with error bars across replicates
5. **Run Comparison**: Side-by-side comparison of best runs from each strategy

## Parameter Configuration

### Grid Search
- 9 combinations (3 lr × 3 momentum)
- 3 replicates = 27 total runs

### Random Search
- 27 trials with distributions (matching grid search)
- 3 replicates = 81 total runs

**Total runs:** ~108
**Runtime:** ~5-15 seconds

## Domain

Optimizes the **Booth function**: `f(x,y) = (x + 2y - 7)² + (2x + y - 5)²`

- **Optimal solution:** x=1, y=3, f=0
- **Search domain:** [-10, 10] for both x and y
