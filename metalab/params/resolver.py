"""
ParamResolver: Validate and derive parameters before execution.

A resolver can:
- Validate parameter values
- Derive conditional parameters based on context
- Transform raw parameters into resolved form
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Protocol

from metalab._ids import fingerprint_params
from metalab.params.source import ParamCase, ParamSource


class ParamResolver(Protocol):
    """
    Protocol for parameter resolvers.

    A resolver transforms raw parameters into resolved parameters,
    optionally using context metadata for conditional logic.
    """

    def resolve(
        self,
        context_meta: dict[str, Any],
        params_raw: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve raw parameters into final form.

        Args:
            context_meta: Metadata from the context (for conditional params).
            params_raw: The raw parameter dictionary.

        Returns:
            The resolved parameter dictionary.

        Raises:
            ValueError: If parameters are invalid.
        """
        ...


class ResolvedSource:
    """
    A ParamSource wrapper that applies a resolver to each case.

    The resolver is called lazily when iterating.
    """

    def __init__(
        self,
        source: ParamSource,
        resolver: ParamResolver | Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        context_meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize with a source and resolver.

        Args:
            source: The underlying parameter source.
            resolver: The resolver to apply (protocol or callable).
            context_meta: Optional context metadata for resolution.
        """
        self._source = source
        self._resolver = resolver
        self._context_meta = context_meta or {}

    def __iter__(self) -> Iterator[ParamCase]:
        """Yield resolved parameter cases."""
        for case in self._source:
            # Resolve parameters
            if hasattr(self._resolver, "resolve"):
                resolved = self._resolver.resolve(self._context_meta, case.params)
            else:
                # Callable form
                resolved = self._resolver(self._context_meta, case.params)

            # Compute new case_id based on resolved params
            case_id = fingerprint_params(resolved)
            yield ParamCase(
                params=resolved,
                case_id=case_id,
                tags=case.tags,
            )

    def __len__(self) -> int:
        """Return the number of cases (same as underlying source)."""
        return len(self._source)  # type: ignore

    def __repr__(self) -> str:
        return f"ResolvedSource({self._source!r})"

    def with_context(self, context_meta: dict[str, Any]) -> ResolvedSource:
        """
        Create a new ResolvedSource with updated context metadata.

        Args:
            context_meta: The context metadata to use for resolution.

        Returns:
            A new ResolvedSource with the specified context.
        """
        return ResolvedSource(
            source=self._source,
            resolver=self._resolver,
            context_meta=context_meta,
        )


def with_resolver(
    source: ParamSource,
    resolver: ParamResolver | Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    context_meta: dict[str, Any] | None = None,
) -> ResolvedSource:
    """
    Wrap a ParamSource with a resolver for parameter transformation.

    Args:
        source: The underlying parameter source.
        resolver: The resolver to apply. Can be:
            - An object implementing ParamResolver protocol
            - A callable (context_meta, params_raw) -> params_resolved
        context_meta: Optional context metadata for resolution.

    Returns:
        A ResolvedSource that applies the resolver to each case.

    Example:
        def my_resolver(context_meta, params_raw):
            resolved = dict(params_raw)
            resolved["n_samples"] = int(resolved["n_samples"])
            return resolved

        params = with_resolver(
            grid(n_samples=[1000.0, 10000.0]),
            resolver=my_resolver,
        )
    """
    return ResolvedSource(source=source, resolver=resolver, context_meta=context_meta)
