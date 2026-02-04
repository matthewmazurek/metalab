"""
FileStore layout: Encapsulates filesystem directory structure.

Centralizes all path construction logic for FileStore, making the
layout configurable and the code cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    """
    Encapsulates FileStore filesystem layout.

    All path construction is centralized here, making it easy to:
    - Understand the directory structure
    - Customize layouts if needed
    - Keep FileStore code clean

    Default layout:
        {root}/
        ├── runs/{run_id}.json
        ├── derived/{run_id}.json
        ├── artifacts/{run_id}/_manifest.json, {name}.{ext}
        ├── logs/{run_id}_{name}.log
        ├── results/{run_id}/{name}.json
        ├── experiments/{exp_id}_{timestamp}.json
        ├── .locks/{run_id}.lock
        └── _meta.json
    """

    root: Path

    # Directory names
    runs_dir: str = "runs"
    derived_dir: str = "derived"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    results_dir: str = "results"
    experiments_dir: str = "experiments"
    locks_dir: str = ".locks"

    # File names
    meta_file: str = "_meta.json"
    manifest_file: str = "_manifest.json"

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"FileStoreLayout(root={self.root!r})"

    def __post_init__(self) -> None:
        # Ensure root is a Path
        if not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))

    # ─────────────────────────────────────────────────────────────────
    # Run record paths
    # ─────────────────────────────────────────────────────────────────

    def run_path(self, run_id: str) -> Path:
        """Path to a run record JSON file."""
        return self.root / self.runs_dir / f"{run_id}.json"

    def runs_dir_path(self) -> Path:
        """Path to the runs directory."""
        return self.root / self.runs_dir

    # ─────────────────────────────────────────────────────────────────
    # Derived metrics paths
    # ─────────────────────────────────────────────────────────────────

    def derived_path(self, run_id: str) -> Path:
        """Path to derived metrics JSON file."""
        return self.root / self.derived_dir / f"{run_id}.json"

    def derived_dir_path(self) -> Path:
        """Path to the derived directory."""
        return self.root / self.derived_dir

    # ─────────────────────────────────────────────────────────────────
    # Artifact paths
    # ─────────────────────────────────────────────────────────────────

    def artifact_dir(self, run_id: str) -> Path:
        """Path to a run's artifact directory."""
        return self.root / self.artifacts_dir / run_id

    def artifact_manifest_path(self, run_id: str) -> Path:
        """Path to a run's artifact manifest."""
        return self.artifact_dir(run_id) / self.manifest_file

    def artifacts_dir_path(self) -> Path:
        """Path to the artifacts directory."""
        return self.root / self.artifacts_dir

    # ─────────────────────────────────────────────────────────────────
    # Log paths
    # ─────────────────────────────────────────────────────────────────

    def log_path(self, run_id: str, name: str) -> Path:
        """Path to a log file."""
        return self.root / self.logs_dir / f"{run_id}_{name}.log"

    def logs_dir_path(self) -> Path:
        """Path to the logs directory."""
        return self.root / self.logs_dir

    # ─────────────────────────────────────────────────────────────────
    # Result paths
    # ─────────────────────────────────────────────────────────────────

    def result_path(self, run_id: str, name: str) -> Path:
        """Path to a structured result JSON file."""
        return self.root / self.results_dir / run_id / f"{name}.json"

    def result_dir(self, run_id: str) -> Path:
        """Path to a run's results directory."""
        return self.root / self.results_dir / run_id

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
        """Path to a run's lock file."""
        return self.root / self.locks_dir / f"{run_id}.lock"

    def locks_dir_path(self) -> Path:
        """Path to the locks directory."""
        return self.root / self.locks_dir

    # ─────────────────────────────────────────────────────────────────
    # Meta file path
    # ─────────────────────────────────────────────────────────────────

    def meta_path(self) -> Path:
        """Path to the store metadata file."""
        return self.root / self.meta_file

    # ─────────────────────────────────────────────────────────────────
    # Directory management
    # ─────────────────────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for dir_name in [
            self.runs_dir,
            self.derived_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.results_dir,
            self.experiments_dir,
            self.locks_dir,
        ]:
            (self.root / dir_name).mkdir(parents=True, exist_ok=True)

    def all_directories(self) -> list[Path]:
        """List all layout directories."""
        return [
            self.runs_dir_path(),
            self.derived_dir_path(),
            self.artifacts_dir_path(),
            self.logs_dir_path(),
            self.results_dir_path(),
            self.experiments_dir_path(),
            self.locks_dir_path(),
        ]
