"""
Optimization benchmark functions and algorithms.

This module provides:
- Classic test functions (Rastrigin, Rosenbrock, Ackley, Sphere, Griewank)
- Simple optimization algorithms (GD, Adam, Simulated Annealing, Random Search, ES)
- Utilities for tracking convergence

These are intentionally simple implementations to stress-test metalab,
not production optimization code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.random import Generator


# =============================================================================
# Test Functions
# =============================================================================

@dataclass(frozen=True)
class TestFunction:
    """A test function for optimization benchmarks."""
    
    name: str
    fn: Callable[[np.ndarray], float]
    bounds: tuple[float, float]  # Per-dimension bounds
    optimum: float  # Known global minimum value
    default_dim: int = 10
    
    def __call__(self, x: np.ndarray) -> float:
        return self.fn(x)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function - highly multimodal."""
    A = 10.0
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * math.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function - narrow curved valley."""
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


def sphere(x: np.ndarray) -> float:
    """Sphere function - simplest convex quadratic."""
    return float(np.sum(x**2))


def ackley(x: np.ndarray) -> float:
    """Ackley function - many local minima, one global."""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * math.pi * x))
    return float(
        -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20
        + math.e
    )


def griewank(x: np.ndarray) -> float:
    """Griewank function - many widespread local minima."""
    sum_sq = np.sum(x**2)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(1 + sum_sq / 4000 - prod_cos)


# Registry of test functions
TEST_FUNCTIONS = {
    "rastrigin": TestFunction(
        name="rastrigin",
        fn=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        default_dim=10,
    ),
    "rosenbrock": TestFunction(
        name="rosenbrock",
        fn=rosenbrock,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        default_dim=10,
    ),
    "sphere": TestFunction(
        name="sphere",
        fn=sphere,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        default_dim=10,
    ),
    "ackley": TestFunction(
        name="ackley",
        fn=ackley,
        bounds=(-32.768, 32.768),
        optimum=0.0,
        default_dim=10,
    ),
    "griewank": TestFunction(
        name="griewank",
        fn=griewank,
        bounds=(-600.0, 600.0),
        optimum=0.0,
        default_dim=10,
    ),
}


# =============================================================================
# Optimization Algorithms
# =============================================================================

@dataclass
class OptResult:
    """Result of an optimization run."""
    
    best_x: np.ndarray
    best_f: float
    history: list[float]  # Function values at each iteration
    x_history: list[np.ndarray]  # Solutions at each iteration
    converged: bool
    iterations: int
    extra: dict[str, Any] = field(default_factory=dict)


def gradient_descent(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    lr: float,
    rng: Generator,
    *,
    momentum: float = 0.0,
    lr_decay: float = 0.0,
    grad_clip: float | None = None,
    diverge_threshold: float = 1e10,
) -> OptResult:
    """
    Gradient descent with finite differences.
    
    Can be configured to converge, diverge, or oscillate via hyperparameters.
    """
    # Initialize in bounds
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)
    
    history = []
    x_history = []
    velocity = np.zeros(dim)
    eps = 1e-8  # For numerical gradient
    
    best_x = x.copy()
    best_f = fn(x)
    
    for i in range(max_iters):
        f = fn(x)
        history.append(f)
        x_history.append(x.copy())
        
        if f < best_f:
            best_f = f
            best_x = x.copy()
        
        # Check divergence
        if not np.isfinite(f) or f > diverge_threshold:
            return OptResult(
                best_x=best_x,
                best_f=best_f,
                history=history,
                x_history=x_history,
                converged=False,
                iterations=i + 1,
                extra={"reason": "diverged"},
            )
        
        # Compute gradient via finite differences
        grad = np.zeros(dim)
        for j in range(dim):
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy()
            x_minus[j] -= eps
            grad[j] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
        
        # Gradient clipping
        if grad_clip is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > grad_clip:
                grad = grad * grad_clip / grad_norm
        
        # Apply momentum
        velocity = momentum * velocity - lr * grad
        
        # Update with learning rate decay
        effective_lr = lr / (1 + lr_decay * i)
        x = x + velocity * (effective_lr / lr)
        
        # Project back to bounds
        x = np.clip(x, lo, hi)
    
    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=history,
        x_history=x_history,
        converged=best_f < 1e-6,
        iterations=max_iters,
    )


def adam(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    lr: float,
    rng: Generator,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    diverge_threshold: float = 1e10,
) -> OptResult:
    """Adam optimizer with finite differences."""
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)
    
    history = []
    x_history = []
    m = np.zeros(dim)  # First moment
    v = np.zeros(dim)  # Second moment
    eps_grad = 1e-8
    
    best_x = x.copy()
    best_f = fn(x)
    
    for i in range(max_iters):
        f = fn(x)
        history.append(f)
        x_history.append(x.copy())
        
        if f < best_f:
            best_f = f
            best_x = x.copy()
        
        if not np.isfinite(f) or f > diverge_threshold:
            return OptResult(
                best_x=best_x,
                best_f=best_f,
                history=history,
                x_history=x_history,
                converged=False,
                iterations=i + 1,
                extra={"reason": "diverged"},
            )
        
        # Compute gradient
        grad = np.zeros(dim)
        for j in range(dim):
            x_plus = x.copy()
            x_plus[j] += eps_grad
            x_minus = x.copy()
            x_minus[j] -= eps_grad
            grad[j] = (fn(x_plus) - fn(x_minus)) / (2 * eps_grad)
        
        # Adam updates
        t = i + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        x = np.clip(x, lo, hi)
    
    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=history,
        x_history=x_history,
        converged=best_f < 1e-6,
        iterations=max_iters,
    )


def simulated_annealing(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: Generator,
    *,
    t_initial: float = 1.0,
    t_final: float = 0.0001,
    cooling: str = "exponential",  # "exponential" or "linear"
    step_size: float = 0.5,
) -> OptResult:
    """Simulated annealing optimizer."""
    lo, hi = fn.bounds
    x = rng.uniform(lo, hi, size=dim)
    
    history = []
    x_history = []
    
    best_x = x.copy()
    best_f = fn(x)
    current_f = best_f
    
    if cooling == "exponential":
        alpha = (t_final / t_initial) ** (1 / max_iters)
    else:
        alpha = (t_initial - t_final) / max_iters
    
    temperature = t_initial
    
    for i in range(max_iters):
        history.append(current_f)
        x_history.append(x.copy())
        
        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()
        
        # Generate neighbor
        perturbation = rng.normal(0, step_size, size=dim)
        x_new = np.clip(x + perturbation, lo, hi)
        f_new = fn(x_new)
        
        # Accept or reject
        delta = f_new - current_f
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            x = x_new
            current_f = f_new
        
        # Cool down
        if cooling == "exponential":
            temperature *= alpha
        else:
            temperature = max(t_final, temperature - alpha)
    
    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=history,
        x_history=x_history,
        converged=best_f < 1e-6,
        iterations=max_iters,
        extra={"final_temperature": temperature},
    )


def random_search(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: Generator,
    *,
    strategy: str = "uniform",  # "uniform", "shrinking", "restart"
    shrink_factor: float = 0.99,
) -> OptResult:
    """Random search optimizer."""
    lo, hi = fn.bounds
    
    best_x = rng.uniform(lo, hi, size=dim)
    best_f = fn(best_x)
    
    history = [best_f]
    x_history = [best_x.copy()]
    
    center = best_x.copy()
    radius = (hi - lo) / 2
    
    for i in range(1, max_iters):
        if strategy == "uniform":
            x_new = rng.uniform(lo, hi, size=dim)
        elif strategy == "shrinking":
            x_new = center + rng.uniform(-radius, radius, size=dim)
            x_new = np.clip(x_new, lo, hi)
            radius *= shrink_factor
        else:  # restart
            if i % 100 == 0:
                x_new = rng.uniform(lo, hi, size=dim)
            else:
                x_new = best_x + rng.normal(0, 0.1, size=dim)
                x_new = np.clip(x_new, lo, hi)
        
        f_new = fn(x_new)
        
        if f_new < best_f:
            best_f = f_new
            best_x = x_new.copy()
            center = best_x.copy()
        
        history.append(best_f)
        x_history.append(best_x.copy())
    
    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=history,
        x_history=x_history,
        converged=best_f < 1e-6,
        iterations=max_iters,
    )


def evolution_strategy(
    fn: TestFunction,
    dim: int,
    max_iters: int,
    rng: Generator,
    *,
    pop_size: int = 20,
    elite_frac: float = 0.2,
    sigma: float = 0.5,
    sigma_decay: float = 0.995,
) -> OptResult:
    """(mu, lambda) Evolution Strategy."""
    lo, hi = fn.bounds
    
    # Initialize population
    population = rng.uniform(lo, hi, size=(pop_size, dim))
    fitness = np.array([fn(x) for x in population])
    
    best_idx = np.argmin(fitness)
    best_x = population[best_idx].copy()
    best_f = fitness[best_idx]
    
    history = [best_f]
    x_history = [best_x.copy()]
    
    n_elite = max(1, int(pop_size * elite_frac))
    
    for i in range(1, max_iters):
        # Select elites
        elite_idx = np.argsort(fitness)[:n_elite]
        elites = population[elite_idx]
        
        # Generate new population from elites
        new_pop = []
        for _ in range(pop_size):
            parent_idx = rng.integers(n_elite)
            parent = elites[parent_idx]
            child = parent + rng.normal(0, sigma, size=dim)
            child = np.clip(child, lo, hi)
            new_pop.append(child)
        
        population = np.array(new_pop)
        fitness = np.array([fn(x) for x in population])
        
        # Update best
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = population[gen_best_idx].copy()
        
        history.append(best_f)
        x_history.append(best_x.copy())
        
        # Decay sigma
        sigma *= sigma_decay
    
    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=history,
        x_history=x_history,
        converged=best_f < 1e-6,
        iterations=max_iters,
        extra={"final_sigma": sigma},
    )


# Registry of algorithms
ALGORITHMS = {
    "gradient_descent": gradient_descent,
    "adam": adam,
    "simulated_annealing": simulated_annealing,
    "random_search": random_search,
    "evolution_strategy": evolution_strategy,
}
