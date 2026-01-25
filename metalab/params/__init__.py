"""
Params module: Parameter generation, canonicalization, and resolution.

Provides:
- ParamSource: Protocol for parameter generators
- ParamCase: A single parameter configuration
- grid(): Cartesian product of parameter values
- random(): Replayable random sampling
- manual(): Explicit list of parameter cases
- with_resolver(): Wrap a source with parameter resolution
"""

from metalab.params.grid import GridSource, grid
from metalab.params.manual import ManualSource, manual
from metalab.params.random import (
    RandomSource,
    choice,
    loguniform,
    loguniform_int,
    randint,
    random,
    uniform,
)
from metalab.params.resolver import ParamResolver, ResolvedSource, with_resolver
from metalab.params.source import ParamCase, ParamSource

__all__ = [
    # Core types
    "ParamSource",
    "ParamCase",
    # Sources
    "GridSource",
    "RandomSource",
    "ManualSource",
    "ResolvedSource",
    # Factories
    "grid",
    "random",
    "manual",
    "with_resolver",
    # Distributions
    "uniform",
    "loguniform",
    "loguniform_int",
    "randint",
    "choice",
    # Resolver
    "ParamResolver",
]
