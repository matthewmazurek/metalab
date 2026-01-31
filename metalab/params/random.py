"""
RandomSource: Replayable random parameter sampling.

Generates random parameter combinations using a seeded RNG,
ensuring reproducibility.
"""

from __future__ import annotations

import math
import random as stdlib_random
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

from metalab._ids import fingerprint_params
from metalab.params.source import ParamCase

# Distribution protocols and implementations


class Distribution(Protocol):
    """Protocol for parameter distributions."""

    def sample(self, rng: stdlib_random.Random) -> Any:
        """Sample a value from this distribution."""
        ...


@dataclass(frozen=True)
class Uniform:
    """Uniform distribution over [low, high)."""

    low: float
    high: float

    def sample(self, rng: stdlib_random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {"type": "Uniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogUniform:
    """Log-uniform distribution over [low, high)."""

    low: float
    high: float

    def sample(self, rng: stdlib_random.Random) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {"type": "LogUniform", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class LogUniformInt:
    """Log-uniform distribution over integers in [low, high]."""

    low: int
    high: int

    def sample(self, rng: stdlib_random.Random) -> int:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return int(round(math.exp(rng.uniform(log_low, log_high))))

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {"type": "LogUniformInt", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class RandInt:
    """Uniform distribution over integers in [low, high]."""

    low: int
    high: int

    def sample(self, rng: stdlib_random.Random) -> int:
        return rng.randint(self.low, self.high)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {"type": "RandInt", "low": self.low, "high": self.high}


@dataclass(frozen=True)
class Choice:
    """Uniform choice from a list of options."""

    options: tuple[Any, ...]

    def sample(self, rng: stdlib_random.Random) -> Any:
        return rng.choice(self.options)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {"type": "Choice", "options": list(self.options)}


# Convenience constructors for distributions


def uniform(low: float, high: float) -> Uniform:
    """Create a uniform distribution over [low, high)."""
    return Uniform(low=low, high=high)


def loguniform(low: float, high: float) -> LogUniform:
    """Create a log-uniform distribution over [low, high)."""
    if low <= 0 or high <= 0:
        raise ValueError("loguniform requires positive bounds")
    return LogUniform(low=low, high=high)


def loguniform_int(low: int, high: int) -> LogUniformInt:
    """Create a log-uniform distribution over integers in [low, high]."""
    if low <= 0 or high <= 0:
        raise ValueError("loguniform_int requires positive bounds")
    return LogUniformInt(low=low, high=high)


def randint(low: int, high: int) -> RandInt:
    """Create a uniform distribution over integers in [low, high]."""
    return RandInt(low=low, high=high)


def choice(options: list[Any]) -> Choice:
    """Create a uniform choice distribution from a list of options."""
    return Choice(options=tuple(options))


# RandomSource implementation


class RandomSource:
    """
    Parameter source that generates random samples.

    Uses a seeded RNG for reproducibility. Given the same seed and space,
    always generates the same sequence of parameter cases.

    Supports O(1) index-based access via __getitem__, where each index
    derives its own deterministic seed, enabling SLURM array submission
    without pre-generating all samples.
    """

    def __init__(
        self,
        space: dict[str, Distribution],
        n_trials: int,
        seed: int,
    ) -> None:
        """
        Initialize the random source.

        Args:
            space: Parameter names mapped to Distribution objects.
            n_trials: Number of random samples to generate.
            seed: Random seed for reproducibility.
        """
        self._space = space
        self._n_trials = n_trials
        self._seed = seed
        self._keys = sorted(space.keys())  # Sorted for determinism

    def _generate_sample(self, index: int) -> ParamCase:
        """
        Generate a single sample for the given index.

        Uses a deterministic seed derived from the base seed and index,
        ensuring the same sample is always generated for the same index
        regardless of access order.

        Args:
            index: The trial index (0-based).

        Returns:
            The ParamCase for this index.
        """
        # Derive a unique seed for this index
        # Using a simple but effective combination that ensures each index
        # gets a completely different RNG state
        sample_seed = self._seed * 1_000_000_007 + index
        rng = stdlib_random.Random(sample_seed)

        params = {}
        for key in self._keys:
            params[key] = self._space[key].sample(rng)

        case_id = fingerprint_params(params)
        return ParamCase(
            params=params,
            case_id=case_id,
            tags=[f"trial_{index}"],
        )

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield random parameter cases."""
        for i in range(self._n_trials):
            yield self._generate_sample(i)

    def __len__(self) -> int:
        """Return the number of trials."""
        return self._n_trials

    def __getitem__(self, index: int) -> ParamCase:
        """
        Get parameter case by index in O(1) time.

        Each index derives its own deterministic seed, so accessing
        source[i] always returns the same sample regardless of what
        other indices have been accessed.

        Args:
            index: The index of the parameter case (0-based).

        Returns:
            The ParamCase at the given index.

        Raises:
            IndexError: If index is out of range.
        """
        # Handle negative indices
        if index < 0:
            index = self._n_trials + index
        if index < 0 or index >= self._n_trials:
            raise IndexError(f"Index {index} out of range [0, {self._n_trials})")

        return self._generate_sample(index)

    def __repr__(self) -> str:
        return f"RandomSource(n_trials={self._n_trials}, seed={self._seed})"

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation for experiment manifests."""
        from metalab.manifest import serialize

        return {
            "type": "RandomSource",
            "space": {k: serialize(v) for k, v in self._space.items()},
            "n_trials": self._n_trials,
            "seed": self._seed,
        }

    @classmethod
    def from_manifest_dict(cls, manifest: dict[str, Any]) -> "RandomSource":
        """
        Reconstruct RandomSource from manifest dict.

        Args:
            manifest: Dict with "space", "n_trials", and "seed" fields.

        Returns:
            A RandomSource with the same configuration.
        """
        space_raw = manifest["space"]
        space: dict[str, Distribution] = {}

        for key, dist_dict in space_raw.items():
            dist_type = dist_dict["type"]
            if dist_type == "Uniform":
                space[key] = Uniform(low=dist_dict["low"], high=dist_dict["high"])
            elif dist_type == "LogUniform":
                space[key] = LogUniform(low=dist_dict["low"], high=dist_dict["high"])
            elif dist_type == "LogUniformInt":
                space[key] = LogUniformInt(low=dist_dict["low"], high=dist_dict["high"])
            elif dist_type == "RandInt":
                space[key] = RandInt(low=dist_dict["low"], high=dist_dict["high"])
            elif dist_type == "Choice":
                space[key] = Choice(options=tuple(dist_dict["options"]))
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")

        return cls(
            space=space,
            n_trials=manifest["n_trials"],
            seed=manifest["seed"],
        )


def random(
    space: dict[str, Distribution],
    n_trials: int,
    seed: int,
) -> RandomSource:
    """
    Create a RandomSource for replayable random parameter sampling.

    Args:
        space: Parameter names mapped to Distribution objects.
        n_trials: Number of random samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A RandomSource that yields random parameter cases.

    Example:
        params = random(
            space={
                "n_samples": loguniform_int(1000, 1000000),
                "store_points": choice([True, False]),
            },
            n_trials=20,
            seed=123,
        )
    """
    return RandomSource(space=space, n_trials=n_trials, seed=seed)
