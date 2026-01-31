"""
Experiment manifest serialization and deserialization.

This module provides utilities for building experiment manifests that capture
the full configuration of an experiment (params, seeds, operation, etc.) for
documentation and reproducibility purposes.

It also provides deserialization helpers for reconstructing param sources
and seed plans from manifest dicts, enabling index-based SLURM array workers
to reconstruct exactly the same parameter/seed configuration as the original
experiment without enumerating all cases.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from metalab._serializable import ManifestSerializable

if TYPE_CHECKING:
    from metalab.experiment import Experiment
    from metalab.params.source import ParamSource
    from metalab.seeds.plan import SeedPlan


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


def deserialize_param_source(manifest: dict[str, Any]) -> "ParamSource":
    """
    Deserialize a ParamSource from a manifest dict.

    Supports GridSource, ManualSource, and RandomSource. All support
    O(1) index-based access for SLURM array submission.

    Args:
        manifest: Dict with "type" field indicating the source type,
                  plus type-specific fields (e.g., "spec" for GridSource).

    Returns:
        A ParamSource instance that can be indexed or iterated.

    Raises:
        ValueError: If the source type is unknown or not supported.

    Example:
        manifest = {"type": "GridSource", "spec": {"lr": [0.01, 0.1]}}
        source = deserialize_param_source(manifest)
        case = source[0]  # Get first param case by index
    """
    from metalab.params.grid import GridSource
    from metalab.params.manual import ManualSource
    from metalab.params.random import RandomSource

    source_type = manifest.get("type")

    if source_type == "GridSource":
        return GridSource.from_manifest_dict(manifest)
    elif source_type == "ManualSource":
        return ManualSource.from_manifest_dict(manifest)
    elif source_type == "RandomSource":
        return RandomSource.from_manifest_dict(manifest)
    else:
        raise ValueError(
            f"Unknown or unsupported ParamSource type: {source_type}. "
            f"Supported types: GridSource, ManualSource, RandomSource"
        )


def deserialize_seed_plan(manifest: dict[str, Any]) -> "SeedPlan":
    """
    Deserialize a SeedPlan from a manifest dict.

    Args:
        manifest: Dict with "base" and "replicates" fields.

    Returns:
        A SeedPlan instance that can be indexed or iterated.

    Raises:
        ValueError: If the manifest type is not SeedPlan.

    Example:
        manifest = {"type": "SeedPlan", "base": 42, "replicates": 3}
        plan = deserialize_seed_plan(manifest)
        bundle = plan[0]  # Get first seed bundle by index
    """
    from metalab.seeds.plan import SeedPlan

    source_type = manifest.get("type")
    if source_type != "SeedPlan":
        raise ValueError(f"Expected SeedPlan manifest, got: {source_type}")

    return SeedPlan.from_manifest_dict(manifest)
