"""Filesystem layout for the clean-break HPC run store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SHARD_COUNT = 64
LAYOUT_VERSION = 4
OUTPUTS_LAYOUT = "hash-mod-sharded-ndjson"
LOGS_LAYOUT = "sharded-live-log-v1"


def safe_experiment_id(experiment_id: str) -> str:
    """Sanitize experiment_id for use as directory/filename.

    Experiment IDs may contain colons (e.g., 'my_exp:1.0') which are
    not valid in all filesystems. This replaces them with underscores.

    Example:
        'my_exp:1.0' -> 'my_exp_1.0'
    """
    return experiment_id.replace(":", "_")


@dataclass(frozen=True)
class FileStoreLayout:
    """Path builder for the v4 filesystem-only run store."""

    root: Path
    layout_version: int = LAYOUT_VERSION
    shard_count: int = DEFAULT_SHARD_COUNT

    # Directory names
    records_dir: str = "records"
    outputs_dir: str = "outputs"
    artifacts_dir: str = "artifacts"
    artifact_files_dir: str = "files"
    artifact_metadata_dir: str = "metadata"
    internal_dir: str = ".metalab"
    events_dir: str = "events"
    heartbeats_dir: str = "heartbeats"
    index_dir: str = "index"
    locks_dir: str = "locks"
    scratch_dir: str = "scratch"
    slurm_logs_dir: str = "slurm-logs"
    log_content_dir: str = "content"
    log_index_dir: str = "index"

    # File names
    meta_file: str = "meta.json"
    root_manifest_file: str = "manifest.json"
    status_cache_file: str = "status-cache.json"
    success_cache_file: str = "success-cache.json"
    duckdb_file: str = "metalab.duckdb"
    duckdb_meta_file: str = "metalab.duckdb.meta.json"
    planned_runs_dir: str = "planned-runs"
    shard_map_file: str = "shard-map.ndjson"
    record_shards_manifest_file: str = "manifest.json"
    submissions_file: str = "submissions.ndjson"

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"FileStoreLayout(root={self.root!r})"

    def __post_init__(self) -> None:
        # Ensure root is a Path
        if not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))
        if self.shard_count < 1:
            raise ValueError("shard_count must be positive")

    # ─────────────────────────────────────────────────────────────────
    # Run record paths
    # ─────────────────────────────────────────────────────────────────

    def shard_id(self, run_id: str) -> str:
        """Stable hash-mod shard id for a run id."""
        digest = hashlib.sha256(run_id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        return f"{value % self.shard_count:04d}"

    def record_path(self, run_id: str) -> Path:
        """Path to the canonical run-record shard for a run id."""
        return self.record_shard_path(run_id)

    def record_shard_dir(self, run_id: str) -> Path:
        """Directory containing run-record shards."""
        return self.record_shards_dir_path()

    def record_shards_dir_path(self) -> Path:
        """Path to canonical run-record shards."""
        return self.records_dir_path()

    def record_shard_path(self, run_id: str) -> Path:
        """Path to the run-record NDJSON shard for a run id."""
        return self.record_shards_dir_path() / f"{self.shard_id(run_id)}.ndjson"

    def record_shard_index_path(self, run_id: str) -> Path:
        """Path to the run-record byte index shard for a run id."""
        return self.record_shards_dir_path() / f"{self.shard_id(run_id)}.idx"

    def record_shards_manifest_path(self) -> Path:
        """Path to the record shard manifest."""
        return self.records_manifest_path()

    def records_dir_path(self) -> Path:
        """Path to canonical run-record shards."""
        return self.root / self.records_dir

    def records_manifest_path(self) -> Path:
        """Path to the record shard manifest."""
        return self.records_dir_path() / self.record_shards_manifest_file

    def records_manifest_reference(self, shard_id: str) -> str:
        """Relative manifest reference for a record shard."""
        return f"{self.records_dir}/{shard_id}.ndjson"

    # ─────────────────────────────────────────────────────────────────
    # Packed output paths
    # ─────────────────────────────────────────────────────────────────

    def outputs_dir_path(self) -> Path:
        """Path to packed output shards."""
        return self.root / self.outputs_dir

    def output_kind_dir_path(self, kind: str) -> Path:
        """Path to one packed output kind."""
        if kind == "artifacts":
            return self.artifact_metadata_dir_path()
        return self.outputs_dir_path() / kind

    def output_shard_path(self, kind: str, run_id: str) -> Path:
        """Path to an output NDJSON shard for a run id."""
        return self.output_kind_dir_path(kind) / f"{self.shard_id(run_id)}.ndjson"

    def outputs_manifest_path(self) -> Path:
        """Path to the output shard manifest."""
        return self.outputs_dir_path() / "manifest.json"

    # ─────────────────────────────────────────────────────────────────
    # Artifact paths
    # ─────────────────────────────────────────────────────────────────

    def artifact_dir(self, run_id: str) -> Path:
        """Path to a run's artifact directory."""
        return self.artifact_files_dir_path() / run_id[:2] / run_id

    def artifact_manifest_path(self, run_id: str) -> Path:
        """Path to packed artifact descriptor shard for a run id."""
        return self.output_shard_path("artifacts", run_id)

    def artifacts_dir_path(self) -> Path:
        """Path to the artifacts directory."""
        return self.outputs_dir_path() / self.artifacts_dir

    def artifact_files_dir_path(self) -> Path:
        """Path to artifact payload files."""
        return self.artifacts_dir_path() / self.artifact_files_dir

    def artifact_metadata_dir_path(self) -> Path:
        """Path to packed artifact metadata shards."""
        return self.artifacts_dir_path() / self.artifact_metadata_dir

    # ─────────────────────────────────────────────────────────────────
    # Log paths
    # ─────────────────────────────────────────────────────────────────

    def log_path(self, run_id: str, name: str) -> Path:
        """Path to the live log content shard for a run id."""
        return self.log_content_path(run_id)

    def log_content_dir_path(self) -> Path:
        """Path to sharded raw log content files."""
        return self.output_kind_dir_path("logs") / self.log_content_dir

    def log_index_dir_path(self) -> Path:
        """Path to sharded log byte-range indexes."""
        return self.output_kind_dir_path("logs") / self.log_index_dir

    def log_content_path(self, run_id: str) -> Path:
        """Path to the raw log content shard for a run id."""
        return self.log_content_dir_path() / f"{self.shard_id(run_id)}.log"

    def log_index_path(self, run_id: str) -> Path:
        """Path to the log byte-range index shard for a run id."""
        return self.log_index_dir_path() / f"{self.shard_id(run_id)}.ndjson"

    def logs_manifest_path(self) -> Path:
        """Path to the live log layout manifest."""
        return self.output_kind_dir_path("logs") / "manifest.json"

    # ─────────────────────────────────────────────────────────────────
    # Result paths
    # ─────────────────────────────────────────────────────────────────

    def result_path(self, run_id: str, name: str) -> Path:
        """Path to a packed result output shard for a run id."""
        return self.output_shard_path("results", run_id)

    def result_dir(self, run_id: str) -> Path:
        """Path to packed result output shards."""
        return self.output_kind_dir_path("results")


    # ─────────────────────────────────────────────────────────────────
    # Submission manifest paths
    # ─────────────────────────────────────────────────────────────────

    def submissions_path(self) -> Path:
        """Path to append-only submission manifest history."""
        return self.internal_dir_path() / self.submissions_file

    # ─────────────────────────────────────────────────────────────────
    # Lock paths
    # ─────────────────────────────────────────────────────────────────

    def lock_path(self, run_id: str) -> Path:
        """Path to a shard lock file for a run id."""
        return self.shard_lock_path(self.shard_id(run_id))

    def shard_lock_path(self, shard_id: str) -> Path:
        """Path to a shard lock file."""
        return self.locks_dir_path() / "shards" / f"{shard_id}.lock"

    def locks_dir_path(self) -> Path:
        """Path to the locks directory."""
        return self.internal_dir_path() / self.locks_dir

    # ─────────────────────────────────────────────────────────────────
    # Meta file path
    # ─────────────────────────────────────────────────────────────────

    def meta_path(self) -> Path:
        """Path to the store metadata file."""
        return self.internal_dir_path() / self.meta_file

    def root_manifest_path(self) -> Path:
        """Path to the active run-store manifest."""
        return self.root / self.root_manifest_file

    def internal_dir_path(self) -> Path:
        """Path to internal rebuildable metalab files."""
        return self.root / self.internal_dir

    def events_dir_path(self) -> Path:
        """Path to the persistent event root."""
        return self.internal_dir_path() / self.events_dir

    def event_log_path(self, job_id: str, worker_id: str) -> Path:
        """Path to a worker's append-only event log."""
        safe_worker = worker_id.replace(":", "_").replace("/", "_")
        return self.events_dir_path() / job_id / f"{safe_worker}.ndjson"

    def heartbeats_dir_path(self) -> Path:
        """Path to the heartbeat root."""
        return self.internal_dir_path() / self.heartbeats_dir

    def heartbeat_path(self, job_id: str, worker_id: str) -> Path:
        """Path to a worker heartbeat JSON file."""
        safe_worker = worker_id.replace(":", "_").replace("/", "_")
        return self.heartbeats_dir_path() / job_id / f"{safe_worker}.json"

    def index_dir_path(self) -> Path:
        """Path to the sidecar index directory."""
        return self.internal_dir_path() / self.index_dir

    def status_cache_path(self) -> Path:
        """Path to the incremental status cache."""
        return self.index_dir_path() / self.status_cache_file

    def success_cache_path(self) -> Path:
        """Path to the rebuildable successful-run cache."""
        return self.index_dir_path() / self.success_cache_file

    def duckdb_path(self) -> Path:
        """Path to the DuckDB sidecar index."""
        return self.index_dir_path() / self.duckdb_file

    def duckdb_meta_path(self) -> Path:
        """Path to the DuckDB sidecar freshness metadata."""
        return self.index_dir_path() / self.duckdb_meta_file

    def planned_runs_dir_path(self) -> Path:
        """Path to sharded planned run id files."""
        return self.index_dir_path() / self.planned_runs_dir

    def planned_runs_path(self, prefix: str) -> Path:
        """Path to one planned run id shard."""
        return self.planned_runs_dir_path() / f"{prefix}.ndjson"

    def shard_map_path(self) -> Path:
        """Path to the rebuildable latest run-record lookup map."""
        return self.index_dir_path() / self.shard_map_file

    def slurm_logs_dir_path(self) -> Path:
        """Path to SLURM stdout/stderr logs."""
        return self.internal_dir_path() / self.slurm_logs_dir

    # ─────────────────────────────────────────────────────────────────
    # Directory management
    # ─────────────────────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_name in [
            self.outputs_dir,
            self.internal_dir,
        ]:
            (self.root / dir_name).mkdir(parents=True, exist_ok=True)
        self.record_shards_dir_path().mkdir(parents=True, exist_ok=True)
        for kind in ("results", "logs"):
            self.output_kind_dir_path(kind).mkdir(parents=True, exist_ok=True)
        self.log_content_dir_path().mkdir(parents=True, exist_ok=True)
        self.log_index_dir_path().mkdir(parents=True, exist_ok=True)
        self.artifact_files_dir_path().mkdir(parents=True, exist_ok=True)
        self.artifact_metadata_dir_path().mkdir(parents=True, exist_ok=True)
        for path in [
            self.events_dir_path(),
            self.heartbeats_dir_path(),
            self.index_dir_path(),
            self.locks_dir_path(),
            self.internal_dir_path() / self.scratch_dir,
            self.slurm_logs_dir_path(),
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def all_directories(self) -> list[Path]:
        """List all layout directories."""
        return [
            self.records_dir_path(),
            self.outputs_dir_path(),
            self.log_content_dir_path(),
            self.log_index_dir_path(),
            self.artifacts_dir_path(),
            self.internal_dir_path(),
            self.events_dir_path(),
            self.heartbeats_dir_path(),
            self.index_dir_path(),
            self.locks_dir_path(),
            self.internal_dir_path() / self.scratch_dir,
            self.slurm_logs_dir_path(),
        ]
