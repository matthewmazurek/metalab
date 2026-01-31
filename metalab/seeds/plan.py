"""
SeedPlan: Generate seed bundles for replicates.

A SeedPlan manages multiple replicates of an experiment,
each with its own deterministically-derived SeedBundle.
"""

from __future__ import annotations

from typing import Any, Iterator

from metalab.seeds.bundle import SeedBundle


class SeedPlan:
    """
    Plan for generating seed bundles across replicates.

    Each replicate gets a unique SeedBundle with a distinct replicate_index,
    allowing for reproducible replication studies.

    Example:
        plan = SeedPlan(base=42, replicates=3)
        for bundle in plan:
            # bundle.replicate_index: 0, 1, 2
            rng = bundle.numpy()
            ...
    """

    def __init__(self, base: int, replicates: int = 1) -> None:
        """
        Initialize the seed plan.

        Args:
            base: The base seed for all bundles.
            replicates: Number of replicates to generate.
        """
        self._base = base
        self._replicates = replicates

    @property
    def base_seed(self) -> int:
        """The base seed for this plan."""
        return self._base

    @property
    def replicates(self) -> int:
        """The number of replicates in this plan."""
        return self._replicates

    def __iter__(self) -> Iterator[SeedBundle]:
        """Yield a SeedBundle for each replicate."""
        for i in range(self._replicates):
            yield SeedBundle(root_seed=self._base, replicate_index=i)

    def __len__(self) -> int:
        """Return the number of replicates."""
        return self._replicates

    def __getitem__(self, index: int) -> SeedBundle:
        """Get the SeedBundle for a specific replicate index."""
        if index < 0 or index >= self._replicates:
            raise IndexError(
                f"Replicate index {index} out of range [0, {self._replicates})"
            )
        return SeedBundle(root_seed=self._base, replicate_index=index)

    def __repr__(self) -> str:
        return f"SeedPlan(base={self._base}, replicates={self._replicates})"

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation for experiment manifests."""
        return {
            "type": "SeedPlan",
            "base": self._base,
            "replicates": self._replicates,
        }

    @classmethod
    def from_manifest_dict(cls, manifest: dict[str, Any]) -> "SeedPlan":
        """
        Reconstruct SeedPlan from manifest dict.

        Args:
            manifest: Dict with "base" and "replicates" fields.

        Returns:
            A SeedPlan with the same configuration.
        """
        return cls(
            base=manifest["base"],
            replicates=manifest["replicates"],
        )


def seeds(base: int, replicates: int = 1) -> SeedPlan:
    """
    Create a SeedPlan for generating seed bundles.

    Args:
        base: The base seed for reproducibility.
        replicates: Number of replicates (default: 1).

    Returns:
        A SeedPlan that yields SeedBundle instances.

    Example:
        seed_plan = seeds(base=42, replicates=3)

        # Use with an experiment
        exp = Experiment(
            name="my_exp",
            seeds=seed_plan,
            ...
        )

        # Or iterate directly
        for bundle in seed_plan:
            rng = bundle.numpy("sampling")
            ...
    """
    return SeedPlan(base=base, replicates=replicates)
