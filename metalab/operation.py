"""
Operation decorator and wrapper.

The @operation decorator marks a function as a metalab operation,
adding metadata and providing a consistent interface.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from metalab.capture import Capture
    from metalab.context.spec import FrozenContext
    from metalab.runtime import Runtime
    from metalab.seeds.bundle import SeedBundle
    from metalab.types import RunRecord

# Type for operation functions
F = TypeVar("F", bound=Callable[..., Any])


class OperationWrapper:
    """
    Wrapper around an operation function.

    Provides:
    - Metadata (name, version)
    - Consistent interface
    - Code hash for provenance
    """

    def __init__(
        self,
        func: Callable[..., RunRecord],
        name: str,
        version: str | None = None,
    ) -> None:
        """
        Initialize the wrapper.

        Args:
            func: The operation function.
            name: The operation name.
            version: Optional version string.
        """
        self._func = func
        self._name = name
        self._version = version or "0.0.0"
        self._code_hash: str | None = None

        # Preserve function metadata
        functools.update_wrapper(self, func)

    @property
    def name(self) -> str:
        """The operation name."""
        return self._name

    @property
    def version(self) -> str:
        """The operation version."""
        return self._version

    @property
    def code_hash(self) -> str:
        """
        A hash of the operation's source code.

        Useful for provenance tracking. Returns empty string if
        source cannot be retrieved.
        """
        if self._code_hash is None:
            try:
                source = inspect.getsource(self._func)
                self._code_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
            except (OSError, TypeError):
                self._code_hash = ""
        return self._code_hash

    def run(
        self,
        context: FrozenContext,
        params: dict[str, Any],
        seeds: SeedBundle,
        runtime: Runtime,
        capture: Capture,
    ) -> RunRecord:
        """
        Execute the operation.

        Args:
            context: The frozen context.
            params: The resolved parameters.
            seeds: The seed bundle.
            runtime: The runtime context.
            capture: The capture interface.

        Returns:
            A RunRecord describing the run outcome.
        """
        return self._func(
            context=context,
            params=params,
            seeds=seeds,
            runtime=runtime,
            capture=capture,
        )

    def __call__(
        self,
        context: FrozenContext,
        params: dict[str, Any],
        seeds: SeedBundle,
        runtime: Runtime,
        capture: Capture,
    ) -> RunRecord:
        """Allow calling the wrapper directly."""
        return self.run(context, params, seeds, runtime, capture)

    def __repr__(self) -> str:
        return f"Operation({self._name}:{self._version})"

    @property
    def ref(self) -> str:
        """
        Get a reference string for this operation.

        Format: "module:name" for reconstruction in workers.
        """
        module = self._func.__module__
        qualname = self._func.__qualname__
        return f"{module}:{qualname}"


def operation(
    name: str,
    version: str | None = None,
) -> Callable[[F], OperationWrapper]:
    """
    Decorator to mark a function as a metalab operation.

    The decorated function should have the signature:
        def my_op(context, params, seeds, runtime, capture) -> RunRecord

    Args:
        name: The operation name (used in experiment_id).
        version: Optional version string.

    Returns:
        A decorator that wraps the function in an OperationWrapper.

    Example:
        @metalab.operation(name="pi_mc", version="1.0")
        def estimate_pi(context, params, seeds, runtime, capture):
            n = params["n_samples"]
            rng = seeds.numpy()
            # ... computation ...
            capture.metric("pi_estimate", pi_est)
            return RunRecord.success()
    """

    def decorator(func: F) -> OperationWrapper:
        return OperationWrapper(func, name=name, version=version)

    return decorator


def import_operation(ref: str) -> OperationWrapper:
    """
    Import an operation from a reference string.

    Args:
        ref: Reference in format "module:name".

    Returns:
        The OperationWrapper instance.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the operation cannot be found.
    """
    module_name, attr_name = ref.rsplit(":", 1)

    import importlib

    module = importlib.import_module(module_name)

    # Handle nested attributes (e.g., "module:Class.method")
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)

    if not isinstance(obj, OperationWrapper):
        raise TypeError(f"Expected OperationWrapper, got {type(obj).__name__}")

    return obj
