# Optimization Benchmark Example

This example demonstrates **context specs**, **ProcessExecutor**, and **comprehensive result analysis** by benchmarking multiple optimization algorithms on classic test functions.

## Features Demonstrated

| Feature | Description |
|---------|-------------|
| `@metalab.context_spec` | Shared configuration across runs |
| `ProcessExecutor` | Process-based parallelism for CPU-bound work |
| Multiple algorithms | Compare optimization methods |
| Multiple problems | Classic test functions |
| Stepped metrics | Optimization trajectories |
| Comprehensive artifacts | Convergence curves, solutions |

## Running the Example

```bash
uv run python examples/optbench/run.py
```

## Code Highlights

### Context Specification

Define shared configuration using a frozen dataclass:

```python
@metalab.context_spec
class BenchmarkConfig:
    """Shared configuration for all optimization runs."""
    max_iterations: int = 2000
    convergence_threshold: float = 1e-6

config = BenchmarkConfig(max_iterations=2000)
exp = metalab.Experiment(context=config, ...)
```

Access in operations:

```python
@metalab.operation
def run_optimization(context, params, seeds, capture):
    for i in range(context.max_iterations):
        # ... optimization loop
```

### Process-Based Parallelism

For CPU-bound optimization work, use ProcessExecutor to bypass Python's GIL:

```python
from metalab import ProcessExecutor

executor = ProcessExecutor(max_workers=4)
handle = metalab.run(exp, executor=executor, progress=True)
```

### Multiple Algorithms and Problems

Grid over algorithms and test functions:

```python
params=metalab.grid(
    algorithm=["gd", "adam", "sa"],
    problem=["sphere", "rosenbrock", "rastrigin"],
    dim=[10, 30],
    lr=[0.01, 0.1],
)
```

## Atlas Visualization

This example is designed to showcase multiple atlas chart types:

### Scatter/Line Charts
- `metrics.final_f` vs `params.dim`, grouped by `params.algorithm`
- `convergence_curve` artifact as line chart (convergence trajectory)

### Bar Charts
- Compare mean `final_f` by algorithm
- Compare `iterations_to_threshold` by problem

### Heatmap
- `params.algorithm` × `params.problem` → `metrics.final_f`
- `params.dim` × `params.lr` → `metrics.convergence_rate`

### Radar Charts
Compare algorithms across multiple dimensions:
- `convergence_rate` - How fast the algorithm converges (higher = faster)
- `solution_distance` - Distance from known optimum (lower = better)
- `stability` - Variance in final iterations (lower = more stable)
- `iterations_to_threshold` - Steps to reach f < 0.1 (lower = faster)

### Candlestick/Histogram
- Distribution of `final_f` across 5 replicates
- Compare variance between algorithms
- Show [min, q1, q3, max] for each algorithm/problem combination

## Parameter Grid

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `algorithm` | gd, adam, sa | Categorical grouping |
| `problem` | sphere, rosenbrock, rastrigin | Categorical grouping |
| `dim` | 10, 30 | Numeric dimension |
| `lr` | 0.01, 0.1 | Learning rate |
| replicates | 5 | Robust statistics |

**Total runs:** 180 (36 param combos × 5 seeds)
**Runtime:** ~1-2 minutes

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **gd** (Gradient Descent) | Classic gradient descent with momentum |
| **adam** | Adaptive moment estimation optimizer |
| **sa** (Simulated Annealing) | Probabilistic global optimization |

## Test Functions

| Function | Description | Optimum |
|----------|-------------|---------|
| **Sphere** | Simplest convex quadratic | f(0,...,0) = 0 |
| **Rosenbrock** | Narrow curved valley | f(1,...,1) = 0 |
| **Rastrigin** | Highly multimodal | f(0,...,0) = 0 |

## Metrics Captured

### Core Metrics
- `final_f` - Final function value (lower is better)
- `converged` - Whether optimization converged
- `iterations` - Number of iterations run
- `optimality_gap` - Distance from known optimum value
- `f_value` (stepped) - Optimization trajectory

### Radar Chart Metrics
These metrics enable multi-dimensional algorithm comparison:
- `convergence_rate` - Slope of log(f) over iterations (higher = faster convergence)
- `solution_distance` - Euclidean distance from known optimal point
- `iterations_to_threshold` - Steps to reach f < 0.1 (lower = faster)
- `stability` - Std deviation of last 10% of iterations (lower = more stable)

## Artifacts

- `convergence_curve` - 1D array of function values over iterations
- `solution` - Final solution vector
