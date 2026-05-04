"""Strict v2 run-record serialization for the clean-break filesystem store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from metalab.types import ArtifactDescriptor, Provenance, RunRecord, Status

# Current schema version
SCHEMA_VERSION = "2"


def load_run_record(data: dict[str, Any]) -> RunRecord:
    """
    Load a RunRecord from a v2 dictionary.

    Args:
        data: Dictionary representation of a RunRecord.

    Returns:
        A RunRecord instance.

    Example:
        >>> data = {"run_id": "abc123", "status": "success", ...}
        >>> record = load_run_record(data)
    """
    version = data.get("_schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run-record schema: {version!r}")

    # Handle status as string or enum
    status = data["status"]
    if isinstance(status, str):
        status = Status(status)

    # Parse timestamps
    started_at = data["started_at"]
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at)
    elif started_at is None:
        raise ValueError("Run record missing started_at")

    finished_at = data["finished_at"]
    if isinstance(finished_at, str):
        finished_at = datetime.fromisoformat(finished_at)
    elif finished_at is None:
        raise ValueError("Run record missing finished_at")

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
        run_id=data["run_id"],
        experiment_id=data["experiment_id"],
        status=status,
        context_fingerprint=data["context_fingerprint"],
        params_fingerprint=data["params_fingerprint"],
        seed_fingerprint=data["seed_fingerprint"],
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=data["duration_ms"],
        metrics=data.get("metrics", {}),
        provenance=provenance,
        error=data.get("error"),
        params_resolved=data.get("params_resolved", {}),
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
        "params_resolved": record.params_resolved,
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


def get_schema_version(data: dict[str, Any]) -> str:
    """Extract the schema version from serialized data."""
    return data.get("_schema_version", "")
