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
        fingerprints: Dict with context_fingerprint, params_fingerprint, seed_fingerprint.
        runtime_hints: Serializable hints (no logger objects).
        operation_ref: Reference to operation (e.g., "module:name").
    """

    run_id: str
    experiment_id: str
    context_spec: Any  # Serializable
    params_resolved: dict[str, Any]
    seed_bundle: SeedBundle
    store_locator: str
    fingerprints: dict[str, str] = field(default_factory=dict)
    runtime_hints: dict[str, Any] = field(default_factory=dict)
    operation_ref: str = ""

    def make_log_label(self, max_params: int = 3) -> str:
        """
        Generate a human-readable label for log filenames.

        Creates a label from key parameter values and seed replicate index.
        Format: {param1}_{param2}_r{replicate_index}

        Args:
            max_params: Maximum number of param values to include.

        Returns:
            A sanitized label suitable for filenames.
        """
        parts: list[str] = []

        # Extract string/numeric param values (skip internal params starting with _)
        for key, value in sorted(self.params_resolved.items()):
            if key.startswith("_"):
                continue
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, bool):
                if value:
                    parts.append(key)
            elif isinstance(value, (int, float)):
                # For numbers, include key name for clarity
                parts.append(f"{key}{value}")

            if len(parts) >= max_params:
                break

        # Add replicate index
        parts.append(f"r{self.seed_bundle.replicate_index}")

        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "context_spec": self.context_spec,
            "params_resolved": self.params_resolved,
            "seed_bundle": self.seed_bundle.to_dict(),
            "store_locator": self.store_locator,
            "fingerprints": self.fingerprints,
            "runtime_hints": self.runtime_hints,
            "operation_ref": self.operation_ref,
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
            fingerprints=data.get("fingerprints", {}),
            runtime_hints=data.get("runtime_hints", {}),
            operation_ref=data.get("operation_ref", ""),
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
