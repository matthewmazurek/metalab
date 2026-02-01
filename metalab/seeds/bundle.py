"""
SeedBundle: Explicit RNG control with deterministic derivation.

A SeedBundle provides:
- A root seed for reproducibility
- Deterministic sub-seed derivation via sha256
- Convenience methods for creating RNG instances
"""

from __future__ import annotations

import hashlib
import random as stdlib_random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Avoid importing numpy at module level (optional dependency)
    pass


def _normalize_seed(seed: int) -> int:
    """Normalize seed to numpy-compatible range."""
    return abs(seed) % (2**32)


@dataclass(frozen=True)
class SeedBundle:
    """
    Bundle of seeds for reproducible random number generation.

    All randomness in an operation should be derived from this bundle,
    ensuring reproducibility given the same SeedBundle.

    Attributes:
        root_seed: The base seed for all derivations.
        replicate_index: The replicate number (if part of a SeedPlan).
    """

    root_seed: int
    replicate_index: int | None = None

    def derive(self, name: str) -> int:
        """
        Derive a sub-seed deterministically from root + name + replicate.

        Uses SHA-256 for cross-platform stability.

        Args:
            name: A unique name for this sub-seed (e.g., "sampling", "init").

        Returns:
            A 64-bit integer seed.

        Example:
            bundle = SeedBundle(root_seed=42, replicate_index=0)
            seed1 = bundle.derive("sampling")
            seed2 = bundle.derive("initialization")
        """
        data = f"{self.root_seed}:{name}:{self.replicate_index or 0}"
        h = hashlib.sha256(data.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def rng(self, name: str = "default") -> stdlib_random.Random:
        """
        Create a stdlib Random instance seeded from this bundle.

        Args:
            name: Name for the derived seed (default: "default").

        Returns:
            A seeded random.Random instance.

        Example:
            bundle = SeedBundle(root_seed=42)
            rng = bundle.rng("sampling")
            value = rng.random()
        """
        seed = self.derive(name)
        return stdlib_random.Random(seed)

    def numpy(self, name: str = "default") -> Any:
        """
        Create a NumPy Generator instance seeded from this bundle.

        Requires numpy to be installed (optional dependency).

        Args:
            name: Name for the derived seed (default: "default").

        Returns:
            A seeded numpy.random.Generator instance.

        Raises:
            ImportError: If numpy is not installed.

        Example:
            bundle = SeedBundle(root_seed=42)
            rng = bundle.numpy("sampling")
            values = rng.random(100)
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for SeedBundle.numpy(). "
                "Install it with: pip install metalab[numpy]"
            ) from e

        seed = self.numpy_seed(name)
        return np.random.default_rng(seed)

    def numpy_seed(self, name: str = "default") -> int:
        """
        Return a NumPy-safe integer seed derived from this bundle.

        Args:
            name: Name for the derived seed (default: "default").

        Returns:
            A 32-bit non-negative integer seed suitable for NumPy.
        """
        return _normalize_seed(self.derive(name))

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the bundle to a dictionary.

        Returns:
            A dictionary representation of the bundle.
        """
        return {
            "root_seed": self.root_seed,
            "replicate_index": self.replicate_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeedBundle:
        """
        Create a SeedBundle from a dictionary.

        Args:
            data: Dictionary with root_seed and optional replicate_index.

        Returns:
            A SeedBundle instance.
        """
        return cls(
            root_seed=data["root_seed"],
            replicate_index=data.get("replicate_index"),
        )

    @classmethod
    def for_preprocessing(cls, base_seed: int) -> SeedBundle:
        """
        Create a SeedBundle for preprocessing steps.

        Use this when you need reproducible randomness during data
        preprocessing, before the experiment runs. The preprocessing seed
        is derived from the base seed using a "preprocessing" namespace,
        ensuring it doesn't collide with replicate seeds.

        Include the seed in the preprocessed filename so changing it
        automatically triggers new preprocessing (cache miss).

        Args:
            base_seed: The experiment's base seed (same value you pass to
                metalab.seeds(base=...)). This ensures preprocessing and
                experiment runs share the same seed hierarchy.

        Returns:
            A SeedBundle for use in preprocessing code.

        Example:
            BASE_SEED = 42

            # Use for preprocessing (before metalab.run)
            seeds = SeedBundle.for_preprocessing(BASE_SEED)
            rng = seeds.numpy("train_test_split")
            train, test = my_split(data, rng=rng)

            # Include seed in filename for automatic cache invalidation
            output_path = f"./cache/processed_seed{BASE_SEED}.h5ad"

            # Same base seed for experiment
            exp = metalab.Experiment(
                context=MyContext(data=metalab.FilePath(output_path)),
                seeds=metalab.seeds(base=BASE_SEED, replicates=5),
                ...
            )
        """
        # Derive a preprocessing-specific root seed from the base
        data = f"{base_seed}:preprocessing:0"
        h = hashlib.sha256(data.encode("utf-8")).digest()
        prep_root = int.from_bytes(h[:8], "big")
        return cls(root_seed=prep_root, replicate_index=None)
