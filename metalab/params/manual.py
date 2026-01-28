"""
ManualSource: Explicit list of parameter cases.

For when you want to specify exact parameter combinations.
"""

from __future__ import annotations

from typing import Any, Iterator

from metalab._ids import fingerprint_params
from metalab.params.source import ParamCase


class ManualSource:
    """
    Parameter source from an explicit list of parameter dictionaries.

    Example:
        source = ManualSource([
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.1, "batch_size": 64},
        ])
    """

    def __init__(
        self,
        cases: list[dict[str, Any]],
        tags: list[str] | None = None,
    ) -> None:
        """
        Initialize with a list of parameter dictionaries.

        Args:
            cases: List of parameter dictionaries.
            tags: Optional tags to apply to all cases.
        """
        self._cases = cases
        self._tags = tags or []

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield parameter cases."""
        for i, params in enumerate(self._cases):
            case_id = fingerprint_params(params)
            yield ParamCase(
                params=params,
                case_id=case_id,
                tags=self._tags + [f"manual_{i}"],
            )

    def __len__(self) -> int:
        """Return the number of cases."""
        return len(self._cases)

    def __repr__(self) -> str:
        return f"ManualSource({len(self._cases)} cases)"

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation for experiment manifests."""
        return {
            "type": "ManualSource",
            "cases": self._cases,
            "tags": self._tags,
            "total_cases": len(self._cases),
        }


def manual(
    cases: list[dict[str, Any]],
    tags: list[str] | None = None,
) -> ManualSource:
    """
    Create a ManualSource from an explicit list of parameter dictionaries.

    Args:
        cases: List of parameter dictionaries.
        tags: Optional tags to apply to all cases.

    Returns:
        A ManualSource that yields the specified cases.

    Example:
        params = manual([
            {"learning_rate": 0.01, "batch_size": 32},
            {"learning_rate": 0.1, "batch_size": 64},
        ])
    """
    return ManualSource(cases=cases, tags=tags)
