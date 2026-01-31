"""
Derived metrics: Post-hoc computed metrics from run artifacts.

This module provides:
- DerivedMetricFn: Type alias for derived metric functions
- import_derived_metric: Import a function by 'module:func' reference
- compute_and_store_derived: Worker-side computation and storage
- compute_derived_for_run: Client-side computation

Derived metrics are computed after runs complete and stored separately
from the run record. They do NOT affect experiment fingerprints.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Callable

from metalab.types import Metric

if TYPE_CHECKING:
    from metalab.result import Run
    from metalab.store.base import Store

logger = logging.getLogger(__name__)

# Type alias for derived metric functions
# A derived metric function receives a Run object and returns a dict of metrics
DerivedMetricFn = Callable[["Run"], dict[str, Metric]]


def import_derived_metric(ref: str) -> DerivedMetricFn:
    """
    Import a derived metric function by 'module:func' reference.

    Args:
        ref: Reference string in format 'module.path:function_name'.

    Returns:
        The imported function.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the function cannot be found.

    Example:
        >>> fn = import_derived_metric("myproject.metrics:final_loss")
        >>> result = fn(run)  # Returns dict[str, Metric]
    """
    if not ref:
        raise ValueError("Empty reference string")

    if ":" not in ref:
        raise ValueError(f"Invalid reference format: {ref}. Expected 'module:func'")

    module_name, func_name = ref.rsplit(":", 1)
    module = importlib.import_module(module_name)

    # Handle nested attributes (e.g., 'module:Class.method')
    obj = module
    for part in func_name.split("."):
        obj = getattr(obj, part)

    return obj


def get_func_ref(func: DerivedMetricFn) -> str:
    """
    Get the import reference string for a function.

    Args:
        func: The function to get a reference for.

    Returns:
        Reference string in format 'module:func'.

    Raises:
        ValueError: If the function cannot be referenced (e.g., lambda).
    """
    module = getattr(func, "__module__", None)
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)

    if not module or not name:
        raise ValueError(
            f"Cannot get reference for {func}. "
            "Derived metric functions must be importable (not lambdas)."
        )

    # Check if it's a lambda or local function
    if "<lambda>" in name or "<locals>" in name:
        raise ValueError(
            f"Cannot get reference for {func}. "
            "Derived metric functions must be importable (not lambdas or local functions)."
        )

    return f"{module}:{name}"


def compute_derived_for_run(
    run: "Run",
    metrics: list[DerivedMetricFn],
) -> dict[str, Metric]:
    """
    Apply derived metric functions to a run.

    Args:
        run: The Run object to compute metrics for.
        metrics: List of derived metric functions.

    Returns:
        Dict of all computed derived metrics (merged).

    Note:
        If a function fails, a warning is logged and that function's
        metrics are skipped. The computation continues with remaining functions.
    """
    derived: dict[str, Metric] = {}

    for func in metrics:
        try:
            result = func(run)
            if result:
                derived.update(result)
        except Exception as e:
            func_name = getattr(func, "__name__", str(func))
            logger.warning(
                f"Derived metric function '{func_name}' failed for run "
                f"{run.run_id[:8]}: {e}"
            )

    return derived


def compute_and_store_derived(
    run_id: str,
    store: "Store",
    metric_refs: list[str],
) -> dict[str, Metric]:
    """
    Import metric functions, apply to run, and persist results.

    This function is used by worker-side computation (thread/slurm executors).
    It creates a Run object from the store, applies all metric functions,
    and persists the results to the store's derived directory.

    Args:
        run_id: The run identifier.
        store: The store containing the run record and artifacts.
        metric_refs: List of function references ('module:func' format).

    Returns:
        Dict of computed derived metrics.

    Note:
        If all functions fail, an empty dict is stored. Individual function
        failures are logged as warnings but don't stop the computation.
    """
    # Import the Run class here to avoid circular imports
    from metalab.result import Run

    # Get the run record
    record = store.get_run_record(run_id)
    if record is None:
        logger.warning(f"Cannot compute derived metrics: run {run_id} not found")
        return {}

    # Create Run object
    run = Run(record, store)

    # Import and apply metric functions
    functions: list[DerivedMetricFn] = []
    for ref in metric_refs:
        try:
            func = import_derived_metric(ref)
            functions.append(func)
        except Exception as e:
            logger.warning(f"Failed to import derived metric '{ref}': {e}")

    # Compute metrics
    derived = compute_derived_for_run(run, functions)

    # Store results
    store.put_derived(run_id, derived)

    return derived
