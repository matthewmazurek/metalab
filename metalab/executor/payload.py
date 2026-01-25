"""
RunPayload: Serializable payload for worker execution.

The payload contains everything needed to execute a run on a worker,
without any callable objects (for pickle safety).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from metalab.seeds.bundle import SeedBundle


@dataclass
class RunPayload:
    """
    Serializable payload for a single run.

    This contains everything a worker needs to execute a run,
    with all references as strings (no callable objects).

    Attributes:
        run_id: The unique run identifier.
        experiment_id: The experiment identifier (name:version).
        context_spec: The serializable context specification.
        params_resolved: The resolved parameter dictionary.
        seed_bundle: The seed bundle for this run.
        store_locator: Path or URI to the store.
        runtime_hints: Serializable hints (no logger objects).
        operation_ref: Reference to operation (e.g., "module:name").
        context_builder_ref: Reference to context builder (None = default).
    """

    run_id: str
    experiment_id: str
    context_spec: Any  # Serializable
    params_resolved: dict[str, Any]
    seed_bundle: SeedBundle
    store_locator: str
    runtime_hints: dict[str, Any] = field(default_factory=dict)
    operation_ref: str = ""
    context_builder_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "context_spec": self.context_spec,
            "params_resolved": self.params_resolved,
            "seed_bundle": self.seed_bundle.to_dict(),
            "store_locator": self.store_locator,
            "runtime_hints": self.runtime_hints,
            "operation_ref": self.operation_ref,
            "context_builder_ref": self.context_builder_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunPayload:
        """Create from a dictionary."""
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            context_spec=data["context_spec"],
            params_resolved=data["params_resolved"],
            seed_bundle=SeedBundle.from_dict(data["seed_bundle"]),
            store_locator=data["store_locator"],
            runtime_hints=data.get("runtime_hints", {}),
            operation_ref=data.get("operation_ref", ""),
            context_builder_ref=data.get("context_builder_ref"),
        )


def import_ref(ref: str) -> Any:
    """
    Import an object from a reference string.

    Args:
        ref: Reference in format "module:name" or "module:class.attr".

    Returns:
        The imported object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute cannot be found.
    """
    if not ref:
        return None

    module_name, attr_path = ref.rsplit(":", 1)
    module = importlib.import_module(module_name)

    # Handle nested attributes
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)

    return obj
