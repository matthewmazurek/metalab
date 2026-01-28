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

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield random parameter cases."""
        rng = stdlib_random.Random(self._seed)

        for i in range(self._n_trials):
            params = {}
            for key in self._keys:
                params[key] = self._space[key].sample(rng)

            case_id = fingerprint_params(params)
            yield ParamCase(
                params=params,
                case_id=case_id,
                tags=[f"trial_{i}"],
            )

    def __len__(self) -> int:
        """Return the number of trials."""
        return self._n_trials

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
