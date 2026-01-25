"""
ParamSource protocol and ParamCase dataclass.

The ParamSource is the core abstraction for parameter generation.
It yields ParamCases, each representing a single parameter configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass(frozen=True)
class ParamCase:
    """
    A single parameter configuration for an experiment run.

    Attributes:
        params: The parameter dictionary.
        case_id: A stable identifier for this case (for deduplication/tracking).
        tags: Optional labels for filtering/grouping.
    """

    params: dict[str, Any]
    case_id: str
    tags: list[str] = field(default_factory=list)


class ParamSource(Protocol):
    """
    Protocol for parameter generators.

    A ParamSource yields ParamCases when iterated. Implementations include:
    - GridSource: Cartesian product of parameter values
    - RandomSource: Replayable random sampling
    - ManualSource: Explicit list of cases

    Example:
        source = grid(learning_rate=[0.01, 0.1], batch_size=[32, 64])
        for case in source:
            print(case.params)  # {'learning_rate': 0.01, 'batch_size': 32}, ...
    """

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield parameter cases."""
        ...
