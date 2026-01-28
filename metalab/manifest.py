"""
Experiment manifest serialization.

This module provides utilities for building experiment manifests that capture
the full configuration of an experiment (params, seeds, operation, etc.) for
documentation and reproducibility purposes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from metalab._serializable import ManifestSerializable

if TYPE_CHECKING:
    from metalab.experiment import Experiment


def serialize(obj: Any) -> Any:
    """
    Serialize an object for the experiment manifest.

    - If obj implements ManifestSerializable, use to_manifest_dict()
    - Otherwise, fall back to type name + repr()

    Args:
        obj: The object to serialize.

    Returns:
        A JSON-serializable representation of the object.
    """
    if isinstance(obj, ManifestSerializable):
        return obj.to_manifest_dict()

    # Fallback for custom sources that don't implement the protocol
    result: dict[str, Any] = {"type": type(obj).__name__, "repr": repr(obj)}
    if hasattr(obj, "__len__"):
        try:
            result["total_cases"] = len(obj)
        except TypeError:
            pass
    return result


def build_experiment_manifest(
    experiment: Experiment,
    context_fingerprint: str,
    total_runs: int,
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build experiment manifest dict.

    Args:
        experiment: The experiment to serialize.
        context_fingerprint: The computed context fingerprint.
        total_runs: The total number of runs in this experiment.
        run_ids: Optional list of all expected run IDs for this experiment.

    Returns:
        A JSON-serializable dict containing the full experiment configuration.
    """
    return {
        "experiment_id": experiment.experiment_id,
        "name": experiment.name,
        "version": experiment.version,
        "description": experiment.description,
        "tags": experiment.tags,
        "operation": {
            "ref": experiment.operation.ref,
            "name": experiment.operation.name,
            "code_hash": experiment.operation.code_hash,
        },
        "params": serialize(experiment.params),
        "seeds": serialize(experiment.seeds),
        "context_fingerprint": context_fingerprint,
        "metadata": experiment.metadata,
        "total_runs": total_runs,
        "run_ids": run_ids,
        "submitted_at": datetime.now().isoformat(),
    }
