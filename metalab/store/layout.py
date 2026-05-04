"""Filesystem layout for the clean-break HPC run store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SHARD_COUNT = 64
METADATA_LAYOUT = "hash-mod-sharded-ndjson"


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
    """Path builder for the v3 filesystem-only run store."""

    root: Path
    layout_version: int = 3
    shard_count: int = DEFAULT_SHARD_COUNT

    # Directory names
    runs_dir: str = "runs"
    metadata_dir: str = "metadata"
    artifacts_dir: str = "artifacts"
    events_dir: str = "events"
    heartbeats_dir: str = "heartbeats"
    index_dir: str = "index"
    logs_dir: str = "logs"
    results_dir: str = "results"
    experiments_dir: str = "experiments"
    locks_dir: str = ".locks"
    scratch_dir: str = ".scratch"

    # File names
    meta_file: str = "_meta.json"
    manifest_file: str = "_manifest.json"
    root_manifest_file: str = "manifest.json"
    status_cache_file: str = "status-cache.json"
    success_cache_file: str = "success-cache.json"
    duckdb_file: str = "metalab.duckdb"
    duckdb_meta_file: str = "metalab.duckdb.meta.json"
    planned_runs_dir: str = "planned-runs"
    shard_map_file: str = "shard-map.ndjson"
    run_shards_manifest_file: str = "manifest.json"

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

    def run_path(self, run_id: str) -> Path:
        """Path to the canonical run-record shard for a run id."""
        return self.run_shard_path(run_id)

    def run_shard_dir(self, run_id: str) -> Path:
        """Directory containing run-record shards."""
        return self.run_shards_dir_path()

    def run_shards_dir_path(self) -> Path:
        """Path to canonical run-record shards."""
        return self.root / self.runs_dir / "shards"

    def run_shard_path(self, run_id: str) -> Path:
        """Path to the run-record NDJSON shard for a run id."""
        return self.run_shards_dir_path() / f"{self.shard_id(run_id)}.ndjson"

    def run_shard_index_path(self, run_id: str) -> Path:
        """Path to the run-record byte index shard for a run id."""
        return self.run_shards_dir_path() / f"{self.shard_id(run_id)}.idx"

    def run_shards_manifest_path(self) -> Path:
        """Path to the run shard manifest."""
        return self.root / self.runs_dir / self.run_shards_manifest_file

    def runs_dir_path(self) -> Path:
        """Path to the runs directory."""
        return self.root / self.runs_dir

    # ─────────────────────────────────────────────────────────────────
    # Packed metadata paths
    # ─────────────────────────────────────────────────────────────────

    def metadata_dir_path(self) -> Path:
        """Path to packed metadata shards."""
        return self.root / self.metadata_dir

    def metadata_kind_dir_path(self, kind: str) -> Path:
        """Path to one packed metadata kind."""
        return self.metadata_dir_path() / kind

    def metadata_shard_path(self, kind: str, run_id: str) -> Path:
        """Path to a metadata NDJSON shard for a run id."""
        return self.metadata_kind_dir_path(kind) / f"{self.shard_id(run_id)}.ndjson"

    def metadata_manifest_path(self) -> Path:
        """Path to the metadata shard manifest."""
        return self.metadata_dir_path() / "manifest.json"

    # ─────────────────────────────────────────────────────────────────
    # Artifact paths
    # ─────────────────────────────────────────────────────────────────

    def artifact_dir(self, run_id: str) -> Path:
        """Path to a run's artifact directory."""
        return self.root / self.artifacts_dir / run_id[:2] / run_id

    def artifact_manifest_path(self, run_id: str) -> Path:
        """Path to packed artifact metadata shard for a run id."""
        return self.metadata_shard_path("artifacts", run_id)

    def artifacts_dir_path(self) -> Path:
        """Path to the artifacts directory."""
        return self.root / self.artifacts_dir

    # ─────────────────────────────────────────────────────────────────
    # Log paths
    # ─────────────────────────────────────────────────────────────────

    def log_path(self, run_id: str, name: str) -> Path:
        """Path to a packed log metadata shard for a run id."""
        return self.metadata_shard_path("logs", run_id)

    def scratch_log_path(self, run_id: str, name: str) -> Path:
        """Non-canonical path for live log streaming before finalization."""
        return (
            self.root
            / self.scratch_dir
            / "logs"
            / self.shard_id(run_id)
            / f"{run_id}_{name}.log"
        )

    def logs_dir_path(self) -> Path:
        """Path to the logs directory."""
        return self.root / self.logs_dir

    # ─────────────────────────────────────────────────────────────────
    # Result paths
    # ─────────────────────────────────────────────────────────────────

    def result_path(self, run_id: str, name: str) -> Path:
        """Path to a packed result metadata shard for a run id."""
        return self.metadata_shard_path("results", run_id)

    def result_dir(self, run_id: str) -> Path:
        """Path to packed result metadata shards."""
        return self.metadata_kind_dir_path("results")

    def results_dir_path(self) -> Path:
        """Path to the results directory."""
        return self.root / self.results_dir

    # ─────────────────────────────────────────────────────────────────
    # Experiment manifest paths
    # ─────────────────────────────────────────────────────────────────

    def experiment_manifest_path(self, experiment_id: str, timestamp: str) -> Path:
        """Path to an experiment manifest JSON file."""
        safe_id = safe_experiment_id(experiment_id)
        return self.root / self.experiments_dir / f"{safe_id}_{timestamp}.json"

    def experiments_dir_path(self) -> Path:
        """Path to the experiments directory."""
        return self.root / self.experiments_dir

    # ─────────────────────────────────────────────────────────────────
    # Lock paths
    # ─────────────────────────────────────────────────────────────────

    def lock_path(self, run_id: str) -> Path:
        """Path to a shard lock file for a run id."""
        return self.shard_lock_path(self.shard_id(run_id))

    def shard_lock_path(self, shard_id: str) -> Path:
        """Path to a shard lock file."""
        return self.root / self.locks_dir / "shards" / f"{shard_id}.lock"

    def locks_dir_path(self) -> Path:
        """Path to the locks directory."""
        return self.root / self.locks_dir

    # ─────────────────────────────────────────────────────────────────
    # Meta file path
    # ─────────────────────────────────────────────────────────────────

    def meta_path(self) -> Path:
        """Path to the store metadata file."""
        return self.root / self.meta_file

    def root_manifest_path(self) -> Path:
        """Path to the active run-store manifest."""
        return self.root / self.root_manifest_file

    def events_dir_path(self) -> Path:
        """Path to the persistent event root."""
        return self.root / self.events_dir

    def event_log_path(self, job_id: str, worker_id: str) -> Path:
        """Path to a worker's append-only event log."""
        safe_worker = worker_id.replace(":", "_").replace("/", "_")
        return self.events_dir_path() / job_id / f"{safe_worker}.ndjson"

    def heartbeats_dir_path(self) -> Path:
        """Path to the heartbeat root."""
        return self.root / self.heartbeats_dir

    def heartbeat_path(self, job_id: str, worker_id: str) -> Path:
        """Path to a worker heartbeat JSON file."""
        safe_worker = worker_id.replace(":", "_").replace("/", "_")
        return self.heartbeats_dir_path() / job_id / f"{safe_worker}.json"

    def index_dir_path(self) -> Path:
        """Path to the sidecar index directory."""
        return self.root / self.index_dir

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

    # ─────────────────────────────────────────────────────────────────
    # Directory management
    # ─────────────────────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_name in [
            self.runs_dir,
            self.metadata_dir,
            self.artifacts_dir,
            self.events_dir,
            self.heartbeats_dir,
            self.index_dir,
            self.logs_dir,
            self.results_dir,
            self.experiments_dir,
            self.locks_dir,
            self.scratch_dir,
        ]:
            (self.root / dir_name).mkdir(parents=True, exist_ok=True)
        self.run_shards_dir_path().mkdir(parents=True, exist_ok=True)
        for kind in ("results", "artifacts", "logs"):
            self.metadata_kind_dir_path(kind).mkdir(parents=True, exist_ok=True)

    def all_directories(self) -> list[Path]:
        """List all layout directories."""
        return [
            self.runs_dir_path(),
            self.metadata_dir_path(),
            self.artifacts_dir_path(),
            self.events_dir_path(),
            self.heartbeats_dir_path(),
            self.index_dir_path(),
            self.logs_dir_path(),
            self.results_dir_path(),
            self.experiments_dir_path(),
            self.locks_dir_path(),
        ]
