"""
Optimization Benchmark Example: Context specs, ProcessExecutor, multiple algorithms.

Usage:
    uv run python examples/optbench/run.py

This example demonstrates:
- Context specs: @metalab.context_spec for shared configuration
- ProcessExecutor: process-based parallelism for CPU-bound work
- Multiple algorithms: comparing optimization methods
- Multiple problems: classic test functions
- Comprehensive artifacts: trajectories, convergence curves
- Stepped metrics: optimization progress over iterations
- Result analysis: grouping by algorithm and problem

Domain: Benchmark optimization algorithms on standard test functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import metalab
from metalab import ProcessExecutor

# =============================================================================
# Test Functions (simplified from benchmark.py)
# =============================================================================


@dataclass(frozen=True)
class TestFunction:
    """A test function for optimization benchmarks."""

    name: str
    fn: Callable[[np.ndarray], float]
    bounds: tuple[float, float]
    optimum: float = 0.0

    def __call__(self, x: np.ndarray) -> float:
        return self.fn(x)


def sphere(x: np.ndarray) -> float:
    """Sphere function - simplest convex quadratic. Optimum at origin."""
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function - narrow curved valley. Optimum at (1,1,...,1)."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal. Optimum at origin."""
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2 * math.pi * x)))


# Registry of test functions
PROBLEMS = {
    "sphere": TestFunction("sphere", sphere, bounds=(-5.12, 5.12)),
    "rosenbrock": TestFunction("rosenbrock", rosenbrock, bounds=(-5.0, 10.0)),
    "rastrigin": TestFunction("rastrigin", rastrigin, bounds=(-5.12, 5.12)),
}


# =============================================================================
# Optimization Algorithms (simplified)
# =============================================================================


@dataclass
class OptResult:
    """Result of an optimization run."""

    best_x: np.ndarray
    best_f: float
    history: list[float]
    converged: bool
    iterations: int


def numerical_gradient(fn: Callable, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute gradient via finite differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad


def gradient_descent(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: np.random.Generator,
    lr: float = 0.01,
    momentum: float = 0.9,
) -> OptResult:
    """Gradient descent with momentum."""
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)
    velocity = np.zeros(dim)

    best_x, best_f = x.copy(), fn(x)
    history = []

    for i in range(max_iters):
        f = fn(x)
        history.append(f)

        if f < best_f:
            best_f = f
            best_x = x.copy()

        # Gradient descent with momentum
        grad = numerical_gradient(fn, x)
        velocity = momentum * velocity - lr * grad
        x = np.clip(x + velocity, lo, hi)

    return OptResult(best_x, best_f, history, best_f < 1e-6, max_iters)


def adam_optimizer(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: np.random.Generator,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> OptResult:
    """Adam optimizer."""
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)
    m = np.zeros(dim)
    v = np.zeros(dim)
    eps = 1e-8

    best_x, best_f = x.copy(), fn(x)
    history = []

    for i in range(max_iters):
        f = fn(x)
        history.append(f)

        if f < best_f:
            best_f = f
            best_x = x.copy()

        grad = numerical_gradient(fn, x)
        t = i + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        x = np.clip(x - lr * m_hat / (np.sqrt(v_hat) + eps), lo, hi)

    return OptResult(best_x, best_f, history, best_f < 1e-6, max_iters)


def simulated_annealing(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: np.random.Generator,
    lr: float = 0.01,  # Ignored, but accept for uniform interface
    t_initial: float = 10.0,
    t_final: float = 0.001,
    step_size: float = 0.5,
) -> OptResult:
    """Simulated annealing optimizer."""
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)

    best_x, best_f = x.copy(), fn(x)
    current_f = best_f
    history = []

    # Exponential cooling
    alpha = (t_final / t_initial) ** (1 / max_iters)
    temperature = t_initial

    for i in range(max_iters):
        history.append(best_f)

        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()

        # Generate neighbor
        x_new = np.clip(x + rng.normal(0, step_size, size=dim), lo, hi)
        f_new = fn(x_new)

        # Accept or reject (Metropolis criterion)
        delta = f_new - current_f
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            x = x_new
            current_f = f_new

        temperature *= alpha

    return OptResult(best_x, best_f, history, best_f < 1e-6, max_iters)


# Algorithm registry
ALGORITHMS = {
    "gd": gradient_descent,
    "adam": adam_optimizer,
    "sa": simulated_annealing,
}


# =============================================================================
# NEW FEATURE: @metalab.context_spec for shared configuration
# =============================================================================


@metalab.context_spec
class BenchmarkConfig:
    """Shared configuration for all optimization runs.

    This is a frozen dataclass that provides:
    - Consistent settings across all runs
    - Automatic fingerprinting for deduplication
    - Type-safe configuration
    """

    max_iterations: int = 500
    convergence_threshold: float = 1e-6


# =============================================================================
# Operation Definition
# =============================================================================


@metalab.operation
def run_optimization(context, params, seeds, capture):
    """Run a single optimization benchmark.

    This operation:
    - Uses context for shared configuration (max iterations)
    - Runs the specified algorithm on the specified problem
    - Captures stepped metrics for visualization
    - Saves artifacts (convergence curve, solution)
    """
    # Extract params
    algorithm_name = params["algorithm"]
    problem_name = params["problem"]
    dim = params["dim"]
    lr = params["lr"]

    # Get function and algorithm
    problem = PROBLEMS[problem_name]
    algorithm = ALGORITHMS[algorithm_name]

    # Get RNG from seeds
    rng = seeds.numpy()

    capture.log(f"Running {algorithm_name} on {problem_name} (dim={dim}, lr={lr})")

    # Run optimization with context-defined max iterations
    result = algorithm(
        fn=problem,
        dim=dim,
        max_iters=context.max_iterations,
        rng=rng,
        lr=lr,
    )

    # Capture stepped metrics (subsample for long runs)
    step_interval = max(1, len(result.history) // 100)
    for i in range(0, len(result.history), step_interval):
        capture.metric("f_value", float(result.history[i]), step=i)

    # Final metrics
    capture.metric("final_f", float(result.best_f))
    capture.metric("converged", result.converged)
    capture.metric("iterations", result.iterations)
    capture.metric("optimality_gap", float(result.best_f - problem.optimum))

    # Artifacts
    capture.artifact("convergence_curve", np.array(result.history))
    capture.artifact("solution", result.best_x)

    capture.log(f"Completed: final_f={result.best_f:.6e}, converged={result.converged}")


# =============================================================================
# Experiment Definition
# =============================================================================

# Create context with custom settings
config = BenchmarkConfig(max_iterations=500, convergence_threshold=1e-6)

exp = metalab.Experiment(
    name="optbench",
    version="2.0",
    context=config,  # NEW FEATURE: context spec
    operation=run_optimization,
    # Grid over algorithms, problems, dimensions, and learning rates
    params=metalab.grid(
        algorithm=["gd", "adam", "sa"],
        problem=["sphere", "rosenbrock", "rastrigin"],
        dim=[2, 10],
        lr=[0.01, 0.1],
    ),
    # 5 replicates for robust statistics
    seeds=metalab.seeds(base=42, replicates=5),
)


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    print("Optimization Benchmark Example")
    print("=" * 60)
    print(f"Algorithms: {list(ALGORITHMS.keys())}")
    print(f"Problems: {list(PROBLEMS.keys())}")
    print(f"Context: max_iterations={config.max_iterations}")

    n_params = len(list(exp.params))
    n_seeds = 5
    print(
        f"Total runs: {n_params} param combos × {n_seeds} seeds = {n_params * n_seeds}"
    )
    print("=" * 60)

    # NEW FEATURE: ProcessExecutor for CPU-bound parallel execution
    # Each optimization run is CPU-intensive, so process-based parallelism
    # bypasses Python's GIL for better performance
    executor = ProcessExecutor(max_workers=4)

    print("\nRunning with ProcessExecutor (4 workers)...")
    handle = metalab.run(exp, executor=executor, progress=True)
    results = handle.result()

    # Analysis
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    successful = results.successful
    print(f"Successful: {len(successful)} / {len(results)}")

    if successful:
        # Best overall result
        best = min(successful, key=lambda r: r.metrics.get("final_f", float("inf")))
        best_params = best.record.params_resolved
        print(f"\nBest run: {best.run_id[:12]}...")
        print(f"  Algorithm: {best_params.get('algorithm', 'N/A')}")
        print(f"  Problem: {best_params.get('problem', 'N/A')}")
        print(f"  Dimension: {best_params.get('dim', 'N/A')}")
        print(f"  Final f: {best.metrics['final_f']:.6e}")
        print(f"  Converged: {best.metrics['converged']}")

        # Summary by algorithm
        print("\n" + "-" * 40)
        print("Mean final_f by algorithm:")
        for algo in ALGORITHMS.keys():
            algo_runs = [
                r
                for r in successful
                if r.record.params_resolved.get("algorithm") == algo
            ]
            if algo_runs:
                mean_f = np.mean([r.metrics["final_f"] for r in algo_runs])
                converged = sum(1 for r in algo_runs if r.metrics["converged"])
                print(
                    f"  {algo}: {mean_f:.6e} ({converged}/{len(algo_runs)} converged)"
                )

        # Summary by problem
        print("\n" + "-" * 40)
        print("Mean final_f by problem:")
        for prob in PROBLEMS.keys():
            prob_runs = [
                r for r in successful if r.record.params_resolved.get("problem") == prob
            ]
            if prob_runs:
                mean_f = np.mean([r.metrics["final_f"] for r in prob_runs])
                print(f"  {prob}: {mean_f:.6e}")

    # Display grouped results
    print("\n" + "=" * 60)
    print("Grouped Display")
    print("=" * 60)
    try:
        results.display(group_by=["algorithm", "problem"])
    except Exception as e:
        print(f"(display error: {e})")

    # Atlas tips
    print("\n" + "=" * 60)
    print("Atlas Visualization Tips")
    print("=" * 60)
    print("In metalab-atlas, try these visualizations:")
    print("  - Plot: metrics.final_f vs params.dim, group by params.algorithm")
    print("  - Plot: metrics.final_f vs params.problem, group by params.algorithm")
    print("  - Compare: algorithms on the same problem")
    print("  - View: convergence_curve artifact as line chart")
    print("  - Filter: by converged=true to see successful optimizations")
