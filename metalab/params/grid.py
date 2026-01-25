"""
GridSource: Cartesian product parameter generation.

Generates all combinations of the provided parameter values.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator

from metalab._ids import fingerprint_params
from metalab.params.source import ParamCase


class GridSource:
    """
    Parameter source that generates a Cartesian product of values.

    Example:
        source = GridSource(learning_rate=[0.01, 0.1], batch_size=[32, 64])
        # Yields 4 cases: all combinations of learning_rate x batch_size
    """

    def __init__(self, **kwargs: list[Any]) -> None:
        """
        Initialize with parameter names mapped to lists of values.

        Args:
            **kwargs: Parameter names mapped to lists of possible values.
        """
        self._params = kwargs
        self._keys = sorted(kwargs.keys())  # Sorted for determinism

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield all combinations of parameter values."""
        if not self._keys:
            # No parameters: yield single empty case
            yield ParamCase(params={}, case_id="empty")
            return

        # Get values in sorted key order
        values_lists = [self._params[k] for k in self._keys]

        for combo in itertools.product(*values_lists):
            params = dict(zip(self._keys, combo))
            case_id = fingerprint_params(params)
            yield ParamCase(params=params, case_id=case_id)

    def __len__(self) -> int:
        """Return the total number of parameter combinations."""
        if not self._params:
            return 1
        total = 1
        for values in self._params.values():
            total *= len(values)
        return total

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in sorted(self._params.items()))
        return f"GridSource({params_str})"


def grid(**kwargs: list[Any]) -> GridSource:
    """
    Create a GridSource for Cartesian product parameter generation.

    Args:
        **kwargs: Parameter names mapped to lists of possible values.

    Returns:
        A GridSource that yields all combinations.

    Example:
        params = grid(
            n_samples=[1000, 10000, 100000],
            store_points=[True, False],
        )
        # Yields 6 cases: 3 n_samples values x 2 store_points values
    """
    return GridSource(**kwargs)
