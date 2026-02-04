"""
Seeds module: Explicit RNG control with deterministic derivation.

Provides:

- SeedBundle: Manages root seed and derived sub-seeds
- SeedPlan: Generates seed bundles for replicates
- seeds(): Factory for creating seed plans
"""

from metalab.seeds.bundle import SeedBundle
from metalab.seeds.plan import SeedPlan, seeds

__all__ = [
    "SeedBundle",
    "SeedPlan",
    "seeds",
]
