"""
Schema versioning and migration helpers.

This module provides:
- SCHEMA_VERSION constant for tracking data format versions
- Tolerant loaders that handle missing fields gracefully
- Migration stubs for future schema evolution

Design principle: Old runs should remain readable even as the schema evolves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from metalab.types import ArtifactDescriptor, Provenance, RunRecord, Status

# Current schema version
SCHEMA_VERSION = "0.1"


def load_run_record(data: dict[str, Any]) -> RunRecord:
    """
    Load a RunRecord from a dictionary, tolerating missing fields.

    This function applies sensible defaults for fields that may be missing
    in older schema versions, ensuring backward compatibility.

    Args:
        data: Dictionary representation of a RunRecord.

    Returns:
        A RunRecord instance.

    Example:
        >>> data = {"run_id": "abc123", "status": "success", ...}
        >>> record = load_run_record(data)
    """
    # Handle status as string or enum
    status = data.get("status", "failed")
    if isinstance(status, str):
        status = Status(status)

    # Parse timestamps
    started_at = data.get("started_at")
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at)
    elif started_at is None:
        started_at = datetime.now()

    finished_at = data.get("finished_at")
    if isinstance(finished_at, str):
        finished_at = datetime.fromisoformat(finished_at)
    elif finished_at is None:
        finished_at = datetime.now()

    # Load provenance
    prov_data = data.get("provenance", {})
    if isinstance(prov_data, dict):
        provenance = Provenance(
            code_hash=prov_data.get("code_hash"),
            python_version=prov_data.get("python_version"),
            metalab_version=prov_data.get("metalab_version"),
            executor_id=prov_data.get("executor_id"),
            host=prov_data.get("host"),
            extra=prov_data.get("extra", {}),
        )
    else:
        provenance = Provenance()

    # Load artifacts
    artifacts = []
    for art_data in data.get("artifacts", []):
        artifacts.append(load_artifact_descriptor(art_data))

    return RunRecord(
        run_id=data.get("run_id", ""),
        experiment_id=data.get("experiment_id", ""),
        status=status,
        context_fingerprint=data.get("context_fingerprint", ""),
        params_fingerprint=data.get("params_fingerprint", ""),
        seed_fingerprint=data.get("seed_fingerprint", ""),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=data.get("duration_ms", 0),
        metrics=data.get("metrics", {}),
        provenance=provenance,
        error=data.get("error"),
        tags=data.get("tags", []),
        warnings=data.get("warnings", []),
        notes=data.get("notes"),
        artifacts=artifacts,
    )


def load_artifact_descriptor(data: dict[str, Any]) -> ArtifactDescriptor:
    """
    Load an ArtifactDescriptor from a dictionary, tolerating missing fields.

    Args:
        data: Dictionary representation of an ArtifactDescriptor.

    Returns:
        An ArtifactDescriptor instance.
    """
    return ArtifactDescriptor(
        artifact_id=data.get("artifact_id", ""),
        name=data.get("name", ""),
        kind=data.get("kind", "blob"),
        format=data.get("format", "binary"),
        uri=data.get("uri", ""),
        content_hash=data.get("content_hash"),
        size_bytes=data.get("size_bytes"),
        metadata=data.get("metadata", {}),
    )


def dump_run_record(record: RunRecord) -> dict[str, Any]:
    """
    Serialize a RunRecord to a dictionary for storage.

    Args:
        record: The RunRecord to serialize.

    Returns:
        A dictionary suitable for JSON serialization.
    """
    return {
        "_schema_version": SCHEMA_VERSION,
        "run_id": record.run_id,
        "experiment_id": record.experiment_id,
        "status": record.status.value,
        "context_fingerprint": record.context_fingerprint,
        "params_fingerprint": record.params_fingerprint,
        "seed_fingerprint": record.seed_fingerprint,
        "started_at": record.started_at.isoformat(),
        "finished_at": record.finished_at.isoformat(),
        "duration_ms": record.duration_ms,
        "metrics": record.metrics,
        "provenance": {
            "code_hash": record.provenance.code_hash,
            "python_version": record.provenance.python_version,
            "metalab_version": record.provenance.metalab_version,
            "executor_id": record.provenance.executor_id,
            "host": record.provenance.host,
            "extra": record.provenance.extra,
        },
        "error": record.error,
        "tags": record.tags,
        "warnings": record.warnings,
        "notes": record.notes,
        "artifacts": [dump_artifact_descriptor(a) for a in record.artifacts],
    }


def dump_artifact_descriptor(descriptor: ArtifactDescriptor) -> dict[str, Any]:
    """
    Serialize an ArtifactDescriptor to a dictionary.

    Args:
        descriptor: The ArtifactDescriptor to serialize.

    Returns:
        A dictionary suitable for JSON serialization.
    """
    return {
        "artifact_id": descriptor.artifact_id,
        "name": descriptor.name,
        "kind": descriptor.kind,
        "format": descriptor.format,
        "uri": descriptor.uri,
        "content_hash": descriptor.content_hash,
        "size_bytes": descriptor.size_bytes,
        "metadata": descriptor.metadata,
    }


# Migration stubs for future schema evolution


def migrate_v01_to_v02(data: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate data from schema v0.1 to v0.2.

    Stub for future use - currently a no-op.
    """
    # Future migrations will be implemented here
    return data


def get_schema_version(data: dict[str, Any]) -> str:
    """
    Extract the schema version from serialized data.

    Args:
        data: Serialized record data.

    Returns:
        The schema version string, or "0.1" if not present.
    """
    return data.get("_schema_version", "0.1")
