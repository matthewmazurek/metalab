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
    import numpy as np


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

        seed = self.derive(name)
        return np.random.default_rng(seed)

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
