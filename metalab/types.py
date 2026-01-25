"""
Core types for metalab (PUBLIC).

This module defines the fundamental data structures used throughout the framework:
- Status: Run completion status
- Provenance: Code/environment tracking
- ArtifactDescriptor: Metadata for captured artifacts
- RunRecord: Complete record of a single run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Run completion status."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Provenance:
    """
    Tracks code and environment information for reproducibility.

    Attributes:
        code_hash: Hash of the operation code (if available)
        python_version: Python version string
        metalab_version: metalab package version
        executor_id: Identifier of the executor used
        host: Hostname where the run executed
        extra: Additional provenance information
    """

    code_hash: str | None = None
    python_version: str | None = None
    metalab_version: str | None = None
    executor_id: str | None = None
    host: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactDescriptor:
    """
    Metadata describing a captured artifact.

    Attributes:
        artifact_id: Unique identifier for this artifact
        name: User-defined name
        kind: Type of artifact (e.g., 'blob', 'json', 'ndarray', 'image')
        format: Serialization format (e.g., 'json', 'npz', 'png')
        uri: Location where the artifact is stored
        content_hash: Hash of the artifact content (optional)
        size_bytes: Size of the artifact in bytes (optional)
        metadata: Additional user-defined metadata
    """

    artifact_id: str
    name: str
    kind: str
    format: str
    uri: str
    content_hash: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    """
    Complete record of a single experiment run.

    This is the primary output of an Operation and is designed to be
    small, serializable, and appendable to a results table.

    Required fields (per contract):
        run_id: Stable identifier derived from experiment + context + params + seed
        experiment_id: Name + version of the experiment
        status: Completion status
        context_fingerprint: Hash of the context spec
        params_fingerprint: Hash of the resolved params
        seed_fingerprint: Hash of the seed bundle
        started_at: When the run started
        finished_at: When the run completed
        duration_ms: Total duration in milliseconds
        metrics: Flat dict of scalar metric values
        provenance: Code/environment tracking
        error: Error information (if failed)

    Optional fields:
        params_resolved: The resolved parameter values for this run
        tags: Labels for filtering/grouping
        warnings: Structured warnings from the run
        notes: Freeform notes
        artifacts: List of artifact descriptors
    """

    # Required fields
    run_id: str
    experiment_id: str
    status: Status
    context_fingerprint: str
    params_fingerprint: str
    seed_fingerprint: str
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    error: dict[str, Any] | None = None

    # Optional fields
    params_resolved: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None
    artifacts: list[ArtifactDescriptor] = field(default_factory=list)

    @classmethod
    def success(
        cls,
        *,
        run_id: str = "",
        experiment_id: str = "",
        context_fingerprint: str = "",
        params_fingerprint: str = "",
        seed_fingerprint: str = "",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        params_resolved: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        artifacts: list[ArtifactDescriptor] | None = None,
        notes: str | None = None,
    ) -> RunRecord:
        """Factory method to create a successful RunRecord."""
        now = datetime.now()
        start = started_at or now
        end = finished_at or now
        duration = int((end - start).total_seconds() * 1000)

        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            status=Status.SUCCESS,
            context_fingerprint=context_fingerprint,
            params_fingerprint=params_fingerprint,
            seed_fingerprint=seed_fingerprint,
            started_at=start,
            finished_at=end,
            duration_ms=duration,
            metrics=metrics or {},
            provenance=provenance or Provenance(),
            params_resolved=params_resolved or {},
            tags=tags or [],
            artifacts=artifacts or [],
            notes=notes,
        )

    @classmethod
    def failed(
        cls,
        *,
        run_id: str = "",
        experiment_id: str = "",
        context_fingerprint: str = "",
        params_fingerprint: str = "",
        seed_fingerprint: str = "",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_type: str = "",
        error_message: str = "",
        error_traceback: str | None = None,
        metrics: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        params_resolved: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        artifacts: list[ArtifactDescriptor] | None = None,
    ) -> RunRecord:
        """Factory method to create a failed RunRecord."""
        now = datetime.now()
        start = started_at or now
        end = finished_at or now
        duration = int((end - start).total_seconds() * 1000)

        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            status=Status.FAILED,
            context_fingerprint=context_fingerprint,
            params_fingerprint=params_fingerprint,
            seed_fingerprint=seed_fingerprint,
            started_at=start,
            finished_at=end,
            duration_ms=duration,
            metrics=metrics or {},
            provenance=provenance or Provenance(),
            error={
                "type": error_type,
                "message": error_message,
                "traceback": error_traceback,
            },
            params_resolved=params_resolved or {},
            tags=tags or [],
            artifacts=artifacts or [],
        )

    @classmethod
    def cancelled(
        cls,
        *,
        run_id: str = "",
        experiment_id: str = "",
        context_fingerprint: str = "",
        params_fingerprint: str = "",
        seed_fingerprint: str = "",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metrics: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        params_resolved: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunRecord:
        """Factory method to create a cancelled RunRecord."""
        now = datetime.now()
        start = started_at or now
        end = finished_at or now
        duration = int((end - start).total_seconds() * 1000)

        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            status=Status.CANCELLED,
            context_fingerprint=context_fingerprint,
            params_fingerprint=params_fingerprint,
            seed_fingerprint=seed_fingerprint,
            started_at=start,
            finished_at=end,
            duration_ms=duration,
            metrics=metrics or {},
            provenance=provenance or Provenance(),
            params_resolved=params_resolved or {},
            tags=tags or [],
        )
