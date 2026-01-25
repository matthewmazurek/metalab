"""
ContextBuilder protocol and default implementation.

A ContextBuilder transforms a ContextSpec (serializable manifest) into
a FrozenContext (in-memory, ready-to-use data).

Contract:
- build() must be deterministic given the same environment/resources
- build() should produce thread-safe output (no lazy mutation after build)
- The resulting FrozenContext must be treated as read-only by operations
"""

from __future__ import annotations

from typing import Protocol

from metalab.context.spec import ContextSpec, FrozenContext


class ContextBuilder(Protocol):
    """
    Protocol for building FrozenContext from ContextSpec.

    Implementations should:
    1. Be deterministic: same spec + same environment = same context
    2. Produce thread-safe output: no lazy mutation after build()
    3. Handle resource loading (files, databases, etc.)

    Example:
        class MyContextBuilder:
            def build(self, spec: MyContextSpec) -> MyContext:
                data = load_dataset(spec.dataset_path)
                return MyContext(data=data, config=spec.config)
    """

    def build(self, spec: ContextSpec) -> FrozenContext:
        """
        Build a FrozenContext from a ContextSpec.

        Args:
            spec: The serializable context specification.

        Returns:
            A FrozenContext ready for use by operations.
            This should be treated as immutable.
        """
        ...


class DefaultContextBuilder:
    """
    Default ContextBuilder that returns the spec as-is (passthrough).

    This is useful when:
    - The ContextSpec is already the FrozenContext (e.g., a config dict)
    - No heavy resource loading is needed
    - Testing or simple use cases
    """

    def build(self, spec: ContextSpec) -> FrozenContext:
        """Return the spec unchanged."""
        return spec
