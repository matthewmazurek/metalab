"""
Optimization Benchmark Experiment.

This experiment is designed to stress-test metalab by exercising:
1. Large parameter grid (algorithm × problem × hyperparameters)
2. Heavy artifact generation (convergence curves, solutions, visualizations)
3. Stepped metrics (progress at every iteration)
4. Mixed success/failure (some configs intentionally diverge)
5. Non-trivial context spec (benchmark suite configuration)
6. Parameter resolution (derived parameters)
7. Concurrent stress (many parallel runs)
8. Memory pressure (varying dimensions)

Run with different "intensity" levels to control the stress factor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

import metalab
from examples.optbench.benchmark import (
    ALGORITHMS,
    TEST_FUNCTIONS,
    OptResult,
)


# =============================================================================
# Context Specification
# =============================================================================

@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    """
    Context spec defining the benchmark suite configuration.
    
    This exercises context fingerprinting with meaningful data.
    """
    
    name: str = "optbench"
    version: str = "1.0"
    
    # Which problems are included
    problems: tuple[str, ...] = ("sphere", "rosenbrock", "rastrigin", "ackley")
    
    # Which algorithms are included
    algorithms: tuple[str, ...] = (
        "gradient_descent",
        "adam",
        "simulated_annealing",
        "random_search",
        "evolution_strategy",
    )
    
    # Benchmark configuration
    convergence_threshold: float = 1e-6
    max_function_evals_factor: int = 100  # max_evals = factor * dim
    
    # Metadata for fingerprinting stability
    config_hash: str = ""  # Computed on construction
    
    def __post_init__(self) -> None:
        # Compute a stable config hash for fingerprinting
        config_data = {
            "problems": self.problems,
            "algorithms": self.algorithms,
            "threshold": self.convergence_threshold,
            "eval_factor": self.max_function_evals_factor,
        }
        import hashlib
        h = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:16]
        object.__setattr__(self, "config_hash", h)


# =============================================================================
# Parameter Resolution
# =============================================================================

class OptBenchResolver:
    """Resolver wrapper that conforms to ParamResolver protocol."""
    
    def resolve(
        self, 
        context_meta: dict[str, Any], 
        params_raw: dict[str, Any]
    ) -> dict[str, Any]:
        return resolve_params(context_meta, params_raw)


def resolve_params(context_meta: dict[str, Any], raw_params: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve/derive parameters based on context and other params.
    
    This exercises the parameter resolver feature:
    - Derives max_iters from dimension if not specified
    - Adjusts learning rate based on problem difficulty
    - Sets algorithm-specific defaults
    """
    params = dict(raw_params)
    
    # Get base values
    dim = int(params.get("dim", 10))
    problem = params.get("problem", "sphere")
    algorithm = params.get("algorithm", "gradient_descent")
    
    # Derive max_iters from dimension if using "auto"
    if params.get("max_iters") == "auto":
        # Harder problems get more iterations
        difficulty_factor = {
            "sphere": 50,
            "rosenbrock": 200,
            "rastrigin": 300,
            "ackley": 200,
            "griewank": 150,
        }.get(problem, 100)
        params["max_iters"] = dim * difficulty_factor
    
    # Ensure max_iters is int
    params["max_iters"] = int(params["max_iters"])
    
    # Scale learning rate for gradient-based methods
    if algorithm in ("gradient_descent", "adam"):
        base_lr = float(params.get("lr", 0.01))
        # Scale down for high-dimensional problems
        params["lr"] = base_lr / np.sqrt(dim)
    
    # Algorithm-specific derived params
    if algorithm == "evolution_strategy":
        pop_size = params.get("pop_size", "auto")
        if pop_size == "auto":
            params["pop_size"] = max(20, 4 * dim)
        else:
            params["pop_size"] = int(pop_size)
    
    # Mark problem difficulty for metrics
    params["_problem_difficulty"] = {
        "sphere": 1,
        "rosenbrock": 3,
        "rastrigin": 5,
        "ackley": 4,
        "griewank": 3,
    }.get(problem, 2)
    
    return params


# =============================================================================
# Operation Definition
# =============================================================================

@metalab.operation(name="optbench_run")
def run_optimization(
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    capture: metalab.Capture,
    runtime: metalab.Runtime,
) -> None:
    """
    Run a single optimization benchmark.
    
    This operation:
    - Runs the specified algorithm on the specified problem
    - Captures metrics at configurable step intervals
    - Saves artifacts (convergence curve, final solution)
    - Handles failures gracefully
    
    Note: Only the parameters you need are required in the signature.
    metalab uses signature inspection to inject only what's requested.
    """
    # Extract params
    algorithm_name = params["algorithm"]
    problem_name = params["problem"]
    dim = int(params["dim"])
    max_iters = int(params["max_iters"])
    
    # Get function and algorithm
    test_fn = TEST_FUNCTIONS[problem_name]
    algorithm_fn = ALGORITHMS[algorithm_name]
    
    # Build algorithm kwargs from params
    algo_kwargs = {}
    
    if algorithm_name == "gradient_descent":
        algo_kwargs = {
            "lr": params.get("lr", 0.01),
            "momentum": params.get("momentum", 0.0),
            "lr_decay": params.get("lr_decay", 0.0),
            "grad_clip": params.get("grad_clip"),
        }
    elif algorithm_name == "adam":
        algo_kwargs = {
            "lr": params.get("lr", 0.001),
            "beta1": params.get("beta1", 0.9),
            "beta2": params.get("beta2", 0.999),
        }
    elif algorithm_name == "simulated_annealing":
        algo_kwargs = {
            "t_initial": params.get("t_initial", 1.0),
            "t_final": params.get("t_final", 0.0001),
            "cooling": params.get("cooling", "exponential"),
            "step_size": params.get("step_size", 0.5),
        }
    elif algorithm_name == "random_search":
        algo_kwargs = {
            "strategy": params.get("strategy", "uniform"),
            "shrink_factor": params.get("shrink_factor", 0.99),
        }
    elif algorithm_name == "evolution_strategy":
        algo_kwargs = {
            "pop_size": params.get("pop_size", 20),
            "elite_frac": params.get("elite_frac", 0.2),
            "sigma": params.get("sigma", 0.5),
            "sigma_decay": params.get("sigma_decay", 0.995),
        }
    
    # Get RNG from seeds
    rng = seeds.numpy("optimization")
    
    # Capture start metrics
    capture.metric("algorithm", algorithm_name)
    capture.metric("problem", problem_name)
    capture.metric("dim", dim)
    capture.metric("max_iters", max_iters)
    capture.metric("difficulty", params.get("_problem_difficulty", 0))
    
    # Run optimization
    result: OptResult = algorithm_fn(
        fn=test_fn,
        dim=dim,
        max_iters=max_iters,
        rng=rng,
        **algo_kwargs,
    )
    
    # Capture stepped metrics (subsample for very long runs)
    step_interval = max(1, len(result.history) // 100)  # At most 100 steps
    for i in range(0, len(result.history), step_interval):
        capture.metric("f_value", float(result.history[i]), step=i)
    
    # Capture final metrics (convert numpy types to native Python types)
    capture.metric("best_f", float(result.best_f))
    capture.metric("converged", bool(result.converged))
    capture.metric("final_iters", int(result.iterations))
    capture.metric("optimality_gap", float(result.best_f - test_fn.optimum))
    
    # Compute convergence rate (linear regression on log scale)
    if len(result.history) > 10 and all(f > 0 for f in result.history):
        log_hist = np.log(result.history)
        x = np.arange(len(log_hist))
        slope, _ = np.polyfit(x, log_hist, 1)
        capture.metric("convergence_rate", float(slope))
    
    # Capture extra algorithm-specific metrics (convert numpy types)
    for key, value in result.extra.items():
        if isinstance(value, (int, float, bool, str, np.integer, np.floating, np.bool_)):
            if isinstance(value, (np.integer, int)):
                capture.metric(f"extra_{key}", int(value))
            elif isinstance(value, (np.floating, float)):
                capture.metric(f"extra_{key}", float(value))
            elif isinstance(value, (np.bool_, bool)):
                capture.metric(f"extra_{key}", bool(value))
            else:
                capture.metric(f"extra_{key}", value)
    
    # ==== Artifact Generation ====
    
    # 1. Convergence curve (numpy array - auto-detected as "numpy" kind)
    capture.artifact(
        name="convergence_curve",
        obj={"values": np.array(result.history, dtype=np.float64)},
        metadata={
            "algorithm": algorithm_name,
            "problem": problem_name,
            "dim": dim,
        },
    )
    
    # 2. Final solution
    capture.artifact(
        name="best_solution",
        obj={
            "x": result.best_x,
            "f": np.array([result.best_f]),
        },
        metadata={"optimum": test_fn.optimum},
    )
    
    # 3. Solution trajectory (every Nth point to limit size)
    trajectory_interval = max(1, len(result.x_history) // 50)
    trajectory = np.array(result.x_history[::trajectory_interval])
    capture.artifact(
        name="trajectory",
        obj={"positions": trajectory},
        metadata={"interval": trajectory_interval},
    )
    
    # 4. Run summary as JSON (ensure all values are native Python types)
    def to_native(v):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.floating):
            return float(v)
        elif isinstance(v, np.bool_):
            return bool(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v
    
    summary = {
        "algorithm": algorithm_name,
        "problem": problem_name,
        "dim": int(dim),
        "max_iters": int(max_iters),
        "best_f": float(result.best_f),
        "converged": bool(result.converged),
        "final_iters": int(result.iterations),
        "extra": {k: to_native(v) for k, v in result.extra.items()},
        "hyperparams": {k: to_native(v) for k, v in params.items() 
                       if not k.startswith("_") and k not in 
                       ("algorithm", "problem", "dim", "max_iters")},
    }
    capture.artifact(
        name="summary",
        obj=summary,
        kind="json",
    )
    
    # 5. Optional: Generate a simple text report
    report_lines = [
        f"Optimization Benchmark Report",
        f"=" * 40,
        f"Algorithm: {algorithm_name}",
        f"Problem: {problem_name}",
        f"Dimension: {dim}",
        f"Max Iterations: {max_iters}",
        f"",
        f"Results:",
        f"  Best f(x): {result.best_f:.6e}",
        f"  Known optimum: {test_fn.optimum}",
        f"  Gap: {result.best_f - test_fn.optimum:.6e}",
        f"  Converged: {result.converged}",
        f"  Iterations used: {result.iterations}",
    ]
    if result.extra:
        report_lines.append(f"\nAlgorithm-specific:")
        for k, v in result.extra.items():
            report_lines.append(f"  {k}: {v}")
    
    report_text = "\n".join(report_lines)
    
    # Write report to temp file and capture
    report_path = runtime.scratch_dir / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text)
    capture.file("report", report_path, kind="text")

    # No return needed - success is implicit


# =============================================================================
# Experiment Builder
# =============================================================================

def build_experiment(
    intensity: str = "light",
    custom_grid: dict[str, Any] | None = None,
) -> metalab.Experiment:
    """
    Build the optimization benchmark experiment.
    
    Args:
        intensity: "light" (quick test), "medium" (reasonable stress), 
                  "heavy" (full stress test), "extreme" (find breaking points)
        custom_grid: Override the default parameter grid
    
    Returns:
        Configured Experiment instance
    """
    # Default context
    context_spec = BenchmarkSuiteSpec()
    
    # Intensity-based configuration
    if intensity == "light":
        # Quick test: 2 algorithms × 2 problems × 2 dims × 2 replicates = 16 runs
        params = custom_grid or metalab.grid(
            algorithm=["gradient_descent", "random_search"],
            problem=["sphere", "rosenbrock"],
            dim=[5, 10],
            max_iters=[100, 500],
            lr=[0.1],  # Only for GD
            strategy=["uniform"],  # Only for random search
        )
        seeds = metalab.seeds(base=42, replicates=2)
        
    elif intensity == "medium":
        # Medium: 4 algorithms × 4 problems × 3 dims × 3 replicates = 144 runs
        params = custom_grid or metalab.grid(
            algorithm=["gradient_descent", "adam", "simulated_annealing", "random_search"],
            problem=["sphere", "rosenbrock", "rastrigin", "ackley"],
            dim=[10, 20, 50],
            max_iters=["auto"],  # Will be resolved
            lr=[0.1, 0.01],
            momentum=[0.0, 0.9],
            t_initial=[1.0],
            cooling=["exponential"],
            strategy=["uniform", "shrinking"],
        )
        seeds = metalab.seeds(base=42, replicates=3)
        
    elif intensity == "heavy":
        # Heavy: 5 algorithms × 5 problems × 4 dims × many hyperparams × 5 replicates
        params = custom_grid or metalab.grid(
            algorithm=list(ALGORITHMS.keys()),
            problem=list(TEST_FUNCTIONS.keys()),
            dim=[10, 25, 50, 100],
            max_iters=["auto"],
            lr=[0.1, 0.01, 0.001],
            momentum=[0.0, 0.5, 0.9],
            lr_decay=[0.0, 0.001],
            t_initial=[1.0, 10.0],
            cooling=["exponential", "linear"],
            step_size=[0.1, 0.5, 1.0],
            strategy=["uniform", "shrinking", "restart"],
            pop_size=["auto"],
            elite_frac=[0.1, 0.2],
            sigma=[0.3, 0.5, 1.0],
        )
        seeds = metalab.seeds(base=42, replicates=5)
        
    elif intensity == "extreme":
        # Extreme: push limits
        # This will create a massive grid - thousands of runs
        params = custom_grid or metalab.grid(
            algorithm=list(ALGORITHMS.keys()),
            problem=list(TEST_FUNCTIONS.keys()),
            dim=[10, 25, 50, 100, 200],
            max_iters=[500, 1000, 2000, 5000],
            lr=[0.5, 0.1, 0.01, 0.001, 0.0001],
            momentum=[0.0, 0.5, 0.9, 0.99],
            lr_decay=[0.0, 0.0001, 0.001],
            grad_clip=[None, 1.0, 10.0],
            t_initial=[0.1, 1.0, 10.0, 100.0],
            t_final=[0.0001, 0.001],
            cooling=["exponential", "linear"],
            step_size=[0.1, 0.3, 0.5, 1.0, 2.0],
            strategy=["uniform", "shrinking", "restart"],
            shrink_factor=[0.95, 0.99, 0.999],
            pop_size=[10, 20, 50, 100],
            elite_frac=[0.1, 0.2, 0.3],
            sigma=[0.1, 0.3, 0.5, 1.0],
            sigma_decay=[0.99, 0.995, 0.999],
        )
        seeds = metalab.seeds(base=42, replicates=10)
        
    else:
        raise ValueError(f"Unknown intensity: {intensity}. Use light/medium/heavy/extreme")
    
    return metalab.Experiment(
        name="optbench",
        version="1.0",
        context=context_spec,
        operation=run_optimization,
        params=params,
        seeds=seeds,
        tags=["benchmark", "optimization", f"intensity:{intensity}"],
        param_resolver=OptBenchResolver(),
    )


def build_targeted_experiment(
    algorithm: str,
    problem: str,
    dim_range: tuple[int, int, int] = (10, 100, 10),  # start, stop, step
    replicates: int = 10,
) -> metalab.Experiment:
    """
    Build a targeted experiment for a specific algorithm-problem pair.
    
    Useful for deep investigation of a particular combination.
    """
    context_spec = BenchmarkSuiteSpec(
        problems=(problem,),
        algorithms=(algorithm,),
    )
    
    dims = list(range(dim_range[0], dim_range[1] + 1, dim_range[2]))
    
    # Algorithm-specific hyperparameter grids
    if algorithm == "gradient_descent":
        hyperparams = metalab.grid(
            algorithm=[algorithm],
            problem=[problem],
            dim=dims,
            max_iters=["auto"],
            lr=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
            momentum=[0.0, 0.5, 0.9, 0.95, 0.99],
            lr_decay=[0.0, 0.0001, 0.001, 0.01],
            grad_clip=[None, 1.0, 5.0, 10.0],
        )
    elif algorithm == "adam":
        hyperparams = metalab.grid(
            algorithm=[algorithm],
            problem=[problem],
            dim=dims,
            max_iters=["auto"],
            lr=[0.1, 0.01, 0.001, 0.0001],
            beta1=[0.8, 0.9, 0.95],
            beta2=[0.99, 0.999, 0.9999],
        )
    elif algorithm == "simulated_annealing":
        hyperparams = metalab.grid(
            algorithm=[algorithm],
            problem=[problem],
            dim=dims,
            max_iters=["auto"],
            t_initial=[0.1, 1.0, 10.0, 100.0],
            t_final=[0.0001, 0.001, 0.01],
            cooling=["exponential", "linear"],
            step_size=[0.1, 0.3, 0.5, 1.0, 2.0],
        )
    elif algorithm == "random_search":
        hyperparams = metalab.grid(
            algorithm=[algorithm],
            problem=[problem],
            dim=dims,
            max_iters=["auto"],
            strategy=["uniform", "shrinking", "restart"],
            shrink_factor=[0.9, 0.95, 0.99, 0.999],
        )
    elif algorithm == "evolution_strategy":
        hyperparams = metalab.grid(
            algorithm=[algorithm],
            problem=[problem],
            dim=dims,
            max_iters=["auto"],
            pop_size=[10, 20, 50, 100],
            elite_frac=[0.05, 0.1, 0.2, 0.3],
            sigma=[0.1, 0.3, 0.5, 1.0],
            sigma_decay=[0.98, 0.99, 0.995, 0.999],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return metalab.Experiment(
        name=f"optbench_{algorithm}_{problem}",
        version="1.0",
        context=context_spec,
        operation=run_optimization,
        params=hyperparams,
        seeds=metalab.seeds(base=42, replicates=replicates),
        tags=["benchmark", "optimization", "targeted", algorithm, problem],
        param_resolver=OptBenchResolver(),
    )


def build_random_search_experiment(
    n_trials: int = 100,
    replicates: int = 3,
) -> metalab.Experiment:
    """
    Build an experiment using random parameter sampling.
    
    This exercises the RandomSource feature.
    """
    context_spec = BenchmarkSuiteSpec()
    
    params = metalab.random(
        space={
            "algorithm": metalab.choice(list(ALGORITHMS.keys())),
            "problem": metalab.choice(list(TEST_FUNCTIONS.keys())),
            "dim": metalab.randint(5, 100),
            "max_iters": metalab.choice([500, 1000, 2000, 5000]),
            "lr": metalab.loguniform(1e-4, 1.0),
            "momentum": metalab.uniform(0.0, 0.99),
            "t_initial": metalab.loguniform(0.1, 100.0),
            "t_final": metalab.loguniform(1e-5, 1e-2),
            "cooling": metalab.choice(["exponential", "linear"]),
            "step_size": metalab.uniform(0.1, 2.0),
            "strategy": metalab.choice(["uniform", "shrinking", "restart"]),
            "pop_size": metalab.choice([10, 20, 50, 100]),
            "elite_frac": metalab.uniform(0.05, 0.3),
            "sigma": metalab.uniform(0.1, 1.0),
            "sigma_decay": metalab.uniform(0.95, 0.999),
        },
        n_trials=n_trials,
        seed=12345,
    )
    
    return metalab.Experiment(
        name="optbench_random",
        version="1.0",
        context=context_spec,
        operation=run_optimization,
        params=params,
        seeds=metalab.seeds(base=42, replicates=replicates),
        tags=["benchmark", "optimization", "random_search"],
        param_resolver=OptBenchResolver(),
    )
