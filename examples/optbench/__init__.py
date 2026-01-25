"""
Optimization Benchmark Example.

A comprehensive stress-test for metalab featuring:
- Multiple optimization algorithms (GD, Adam, SA, Random, ES)
- Classic test functions (Rastrigin, Rosenbrock, Sphere, Ackley, Griewank)
- Large parameter grids
- Heavy artifact generation
- Stepped metrics tracking
- Concurrent execution stress testing
"""

from examples.optbench.benchmark import (
    ALGORITHMS,
    TEST_FUNCTIONS,
    OptResult,
    TestFunction,
)
from examples.optbench.experiment import (
    BenchmarkSuiteSpec,
    build_experiment,
    build_random_search_experiment,
    build_targeted_experiment,
    run_optimization,
)

__all__ = [
    # Benchmark components
    "ALGORITHMS",
    "TEST_FUNCTIONS",
    "TestFunction",
    "OptResult",
    # Experiment builders
    "BenchmarkSuiteSpec",
    "build_experiment",
    "build_targeted_experiment",
    "build_random_search_experiment",
    "run_optimization",
]
