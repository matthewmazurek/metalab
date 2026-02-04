"""
FileStore: Filesystem-based storage backend.

Layout (when experiment_id provided):
    {file_root}/
    └── {experiment_id}/             # Experiment directory (sanitized)
        ├── runs/{run_id}.json       # RunRecord (schema-versioned)
        ├── derived/{run_id}.json    # Derived metrics (post-hoc computed)
        ├── artifacts/{run_id}/      # Artifact files + _manifest.json
        ├── logs/{run_id}_{name}.log # Log files
        ├── results/{run_id}/{name}.json # Structured results
        ├── experiments/{exp_id}_{ts}.json # Experiment manifests
        ├── .locks/{run_id}.lock     # Per-run lock files
        └── _meta.json               # Store metadata

Layout (when no experiment_id - unscoped mode):
    {root}/
    ├── runs/{run_id}.json
    ├── derived/{run_id}.json
    ├── artifacts/{run_id}/
    ├── logs/{run_id}_{name}.log
    ├── results/{run_id}/{name}.json
    ├── experiments/{exp_id}_{ts}.json
    ├── .locks/{run_id}.lock
    └── _meta.json

Concurrency guarantees:

- Atomic writes: temp file + os.rename()
- Per-run locking: fcntl.flock() on {run_id}.lock
- Artifact collision: overwrite with warning
"""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generator

if TYPE_CHECKING:
    from typing import BinaryIO

    from metalab.store.locator import LocatorInfo

from metalab.schema import (
    SCHEMA_VERSION,
    dump_artifact_descriptor,
    dump_run_record,
    load_artifact_descriptor,
    load_run_record,
)
from metalab.store.config import StoreConfig
from metalab.store.layout import FileStoreLayout, safe_experiment_id
from metalab.types import ArtifactDescriptor, Metric, RunRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FileStoreConfig(StoreConfig):
    """
    Configuration for FileStore.

    Example:
    ```python
    config = FileStoreConfig(root="./experiments")
    scoped = config.scoped("my_exp:1.0")
    store = scoped.connect()

    # Or let runner handle scoping:
    metalab.run(exp, store=config)  # auto-scopes to experiment
    ```
    """

    scheme: ClassVar[str] = "file"
    root: str
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        # Normalize to absolute path
        resolved = str(Path(self.root).resolve())
        if self.root != resolved:
            object.__setattr__(self, "root", resolved)

    def connect(self) -> "FileStore":
        """Create a FileStore from this config."""
        return FileStore(self)

    @classmethod
    def from_locator(cls, info: "LocatorInfo", **kwargs: Any) -> "FileStoreConfig":
        """Parse file:// locator into config."""
        experiment_id = kwargs.pop("experiment_id", None) or info.params.get(
            "experiment_id"
        )
        return cls(root=info.path, experiment_id=experiment_id)

    def list_experiments(self) -> list[str]:
        """
        List all experiment IDs in this collection.

        Discovers experiments by scanning subdirectories for _meta.json files
        that contain experiment_id. Only works on unscoped configs.

        Returns:
            List of experiment IDs found in this collection.

        Raises:
            ValueError: If called on a scoped config.

        Example:
        ```python
        config = FileStoreConfig(root="./experiments")
        experiments = config.list_experiments()
        # ['my_exp:1.0', 'my_exp:2.0', 'other_exp:1.0']
        ```
        """
        if self.experiment_id is not None:
            raise ValueError("Cannot list experiments on a scoped config")

        root = Path(self.root)
        if not root.exists():
            return []

        experiments = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / "_meta.json"
            if meta_path.exists():
                try:
                    import json

                    with open(meta_path) as f:
                        meta = json.load(f)
                    if "experiment_id" in meta:
                        experiments.append(meta["experiment_id"])
                except (json.JSONDecodeError, OSError):
                    # Skip malformed or inaccessible meta files
                    pass
        return sorted(experiments)

    def for_experiment(self, experiment_id: str) -> "FileStoreConfig":
        """
        Get a scoped config for a specific experiment.

        Alias for scoped() with a clearer name for loading/browsing context.

        Args:
            experiment_id: The experiment ID to scope to.

        Returns:
            A new FileStoreConfig scoped to the experiment.

        Example:
        ```python
        collection = FileStoreConfig(root="./experiments")
        config = collection.for_experiment("my_exp:1.0")
        results = load_results(config)
        ```
        """
        return self.scoped(experiment_id)


class FileStore:
    """
    Filesystem-based storage backend.

    Provides atomic writes and per-run locking for concurrent access.
    Uses FileStoreLayout for all path construction.

    Create via FileStoreConfig:
    ```python
    config = FileStoreConfig(root="./experiments")
    store = config.connect()
    ```
    """

    def __init__(self, config: FileStoreConfig) -> None:
        """
        Initialize from config.

        Use FileStoreConfig(...).connect() to create instances.

        Args:
            config: The FileStoreConfig for this store.
        """
        self._config = config

        # Compute effective root
        effective_root = Path(config.root)
        if config.experiment_id:
            effective_root = effective_root / safe_experiment_id(config.experiment_id)

        self._layout = FileStoreLayout(effective_root)
        self._ensure_layout()

    @property
    def config(self) -> FileStoreConfig:
        """The configuration for this store."""
        return self._config

    @property
    def is_scoped(self) -> bool:
        """True if this store is scoped to a specific experiment."""
        return self._config.is_scoped

    def scoped(self, experiment_id: str) -> "FileStore":
        """
        Return a store scoped to the given experiment.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            A new FileStore scoped to the experiment.
        """
        return self._config.scoped(experiment_id).connect()

    def _ensure_layout(self) -> None:
        """Create the directory structure and meta file if needed."""
        self._layout.ensure_directories()

        # Write meta file if it doesn't exist
        meta_path = self._layout.meta_path()
        if not meta_path.exists():
            meta_data: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "created_by": "metalab",
            }
            # Include experiment_id for scoped stores (enables collection discovery)
            if self._config.experiment_id:
                meta_data["experiment_id"] = self._config.experiment_id
            self._atomic_write_json(meta_path, meta_data)

    @property
    def root(self) -> Path:
        """The root directory of this store."""
        return self._layout.root

    @property
    def layout(self) -> FileStoreLayout:
        """The layout configuration for this store."""
        return self._layout

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return f"FileStore(root={self._layout.root!r})"

    # =========================================================================
    # Capability: SupportsWorkingDirectory
    # =========================================================================

    def get_working_directory(self) -> Path:
        """
        Return the root directory of this store.

        Implements SupportsWorkingDirectory capability.
        """
        return self._layout.root

    # =========================================================================
    # Atomic write utilities
    # =========================================================================

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write data atomically using temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
        try:
            tmp.write_bytes(data)
            tmp.rename(path)  # Atomic on POSIX
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON atomically."""
        content = json.dumps(data, indent=2, sort_keys=True)
        FileStore._atomic_write(path, content.encode("utf-8"))

    @contextmanager
    def _run_lock(self, run_id: str) -> Generator[None, None, None]:
        """Acquire a per-run lock using flock."""
        lock_path = self._layout.lock_path(run_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()

        with lock_path.open("r") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # =========================================================================
    # Run record operations
    # =========================================================================

    def put_run_record(self, record: RunRecord) -> None:
        """Persist a run record atomically."""
        path = self._layout.run_path(record.run_id)

        with self._run_lock(record.run_id):
            data = dump_run_record(record)
            self._atomic_write_json(path, data)

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID."""
        path = self._layout.run_path(run_id)

        if not path.exists():
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        return load_run_record(data)

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """List run records, optionally filtered by experiment."""
        records = []
        runs_dir = self._layout.runs_dir_path()

        if not runs_dir.exists():
            return records

        for path in runs_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                record = load_run_record(data)

                if experiment_id is None or record.experiment_id == experiment_id:
                    records.append(record)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load run record {path}: {e}")

        return records

    def run_exists(self, run_id: str) -> bool:
        """Check if a run record exists."""
        return self._layout.run_path(run_id).exists()

    # =========================================================================
    # Artifact operations
    # =========================================================================

    def put_artifact(
        self,
        data: bytes | Path,
        descriptor: ArtifactDescriptor,
    ) -> ArtifactDescriptor:
        """Store an artifact."""
        # Get run_id from descriptor metadata (preferred) or fall back to extraction
        run_id = descriptor.metadata.get("_run_id")

        if not run_id:
            # Fallback: try to extract from path (legacy behavior)
            if isinstance(data, Path):
                for parent in data.parents:
                    if parent.name.startswith("metalab_"):
                        parts = parent.name.split("_")
                        if len(parts) >= 2:
                            run_id = parts[1]
                            break

            if not run_id:
                # Last resort: use artifact_id prefix
                run_id = descriptor.artifact_id[:16]

        # Determine destination path
        artifact_dir = self._layout.artifact_dir(run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Use the original filename/extension
        if isinstance(data, Path):
            dest_name = f"{descriptor.name}{data.suffix}"
        else:
            ext = f".{descriptor.format}" if descriptor.format else ""
            dest_name = f"{descriptor.name}{ext}"

        dest_path = artifact_dir / dest_name

        # Check for collision
        if dest_path.exists():
            logger.warning(f"Artifact {dest_name} already exists, overwriting")

        with self._run_lock(run_id):
            # Copy or write the artifact
            if isinstance(data, Path):
                shutil.copy2(data, dest_path)
            else:
                self._atomic_write(dest_path, data)

            # Update manifest
            self._update_manifest(run_id, descriptor, str(dest_path))

        # Return updated descriptor with new URI
        return ArtifactDescriptor(
            artifact_id=descriptor.artifact_id,
            name=descriptor.name,
            kind=descriptor.kind,
            format=descriptor.format,
            uri=str(dest_path),
            content_hash=descriptor.content_hash,
            size_bytes=descriptor.size_bytes,
            metadata=descriptor.metadata,
        )

    def _update_manifest(
        self,
        run_id: str,
        descriptor: ArtifactDescriptor,
        uri: str,
    ) -> None:
        """Update the artifact manifest for a run."""
        manifest_path = self._layout.artifact_manifest_path(run_id)

        # Load existing manifest
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"artifacts": []}

        # Add or update entry (strip internal _run_id from metadata)
        clean_metadata = {
            k: v for k, v in descriptor.metadata.items() if not k.startswith("_")
        }
        updated_descriptor = dump_artifact_descriptor(
            ArtifactDescriptor(
                artifact_id=descriptor.artifact_id,
                name=descriptor.name,
                kind=descriptor.kind,
                format=descriptor.format,
                uri=uri,
                content_hash=descriptor.content_hash,
                size_bytes=descriptor.size_bytes,
                metadata=clean_metadata,
            )
        )

        # Remove existing entry with same name (overwrite)
        manifest["artifacts"] = [
            a for a in manifest["artifacts"] if a.get("name") != descriptor.name
        ]
        manifest["artifacts"].append(updated_descriptor)

        # Write manifest atomically
        self._atomic_write_json(manifest_path, manifest)

    def get_artifact(self, uri: str) -> bytes:
        """Retrieve artifact data."""
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {uri}")
        return path.read_bytes()

    # =========================================================================
    # Capability: SupportsArtifactOpen
    # =========================================================================

    def open_artifact(self, uri: str) -> "BinaryIO":
        """
        Open an artifact for reading.

        Implements SupportsArtifactOpen capability.
        """
        path = Path(uri)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {uri}")
        return open(path, "rb")  # type: ignore[return-value]

    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """List artifacts for a run."""
        manifest_path = self._layout.artifact_manifest_path(run_id)

        if not manifest_path.exists():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [load_artifact_descriptor(a) for a in manifest.get("artifacts", [])]

    # =========================================================================
    # Derived metrics operations
    # =========================================================================

    def put_derived(self, run_id: str, derived: dict[str, Metric]) -> None:
        """Persist derived metrics for a run."""
        path = self._layout.derived_path(run_id)

        with self._run_lock(run_id):
            self._atomic_write_json(path, derived)

    def get_derived(self, run_id: str) -> dict[str, Metric] | None:
        """Retrieve derived metrics for a run."""
        path = self._layout.derived_path(run_id)

        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist for a run."""
        return self._layout.derived_path(run_id).exists()

    # =========================================================================
    # Structured results operations
    # =========================================================================

    def put_result(
        self,
        run_id: str,
        name: str,
        data: Any,
        dtype: str | None = None,
        shape: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store structured result data for a run.

        Results are stored in JSON format at results/{run_id}/{name}.json.
        """
        result_dir = self._layout.result_dir(run_id)
        result_dir.mkdir(parents=True, exist_ok=True)
        path = self._layout.result_path(run_id, name)

        result_obj = {
            "data": data,
            "dtype": dtype,
            "shape": shape,
            "metadata": metadata or {},
        }

        with self._run_lock(run_id):
            self._atomic_write_json(path, result_obj)

    def get_result(self, run_id: str, name: str) -> dict[str, Any] | None:
        """Retrieve structured result data."""
        path = self._layout.result_path(run_id, name)

        if not path.exists():
            return None

        return json.loads(path.read_text(encoding="utf-8"))

    def list_results(self, run_id: str) -> list[str]:
        """List result names for a run."""
        result_dir = self._layout.result_dir(run_id)

        if not result_dir.exists():
            return []

        return [p.stem for p in result_dir.glob("*.json")]

    # =========================================================================
    # Log operations
    # =========================================================================

    def put_log(self, run_id: str, name: str, content: str) -> None:
        """Store a log file for a run."""
        log_path = self._layout.log_path(run_id, name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(log_path, content.encode("utf-8"))

    def get_log_path(self, run_id: str, name: str) -> Path:
        """
        Get the path where a log file should be written.

        Enables streaming loggers to write directly to the store.
        """
        log_path = self._layout.log_path(run_id, name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path

    def get_log(self, run_id: str, name: str) -> str | None:
        """
        Retrieve a log file.

        Searches for logs matching the run_id. Handles legacy formats.
        """
        log_dir = self._layout.logs_dir_path()

        # Try new format: {run_id}_{name}.log
        new_path = self._layout.log_path(run_id, name)
        if new_path.exists():
            return new_path.read_text(encoding="utf-8")

        # Fall back to legacy label format: *_{short_id}_{name}.log
        short_id = run_id[:8]
        for pattern in [f"*_{short_id}_{name}.log"]:
            matches = list(log_dir.glob(pattern))
            if matches:
                return matches[0].read_text(encoding="utf-8")

        # Fall back to legacy nested format: {run_id}/{name}.txt
        legacy_path = log_dir / run_id / f"{name}.txt"
        if legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8")

        return None

    def list_logs(self, run_id: str) -> list[str]:
        """
        List available log names for a run.

        Returns a list of log names (e.g., ["run", "stdout", "stderr"]).
        """
        log_dir = self._layout.logs_dir_path()
        short_id = run_id[:8]
        log_names: set[str] = set()

        # Search new flat format
        for pattern in [f"*_{short_id}_*.log", f"{run_id}_*.log"]:
            for log_file in log_dir.glob(pattern):
                filename = log_file.stem
                if f"_{short_id}_" in filename:
                    name = filename.split(f"_{short_id}_", 1)[-1]
                    log_names.add(name)
                elif filename.startswith(f"{run_id}_"):
                    name = filename[len(run_id) + 1 :]
                    log_names.add(name)

        # Search legacy nested format
        legacy_dir = log_dir / run_id
        if legacy_dir.exists() and legacy_dir.is_dir():
            for log_file in legacy_dir.glob("*.txt"):
                log_names.add(log_file.stem)

        return sorted(log_names)

    # =========================================================================
    # Capability: SupportsExperimentManifests
    # =========================================================================

    def put_experiment_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Store an experiment manifest."""
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_dir = self._layout.experiments_dir_path()
        exp_dir.mkdir(parents=True, exist_ok=True)

        exp_manifest_path = self._layout.experiment_manifest_path(
            experiment_id, timestamp
        )
        self._atomic_write_json(exp_manifest_path, manifest)
        logger.debug(f"Saved experiment manifest to {exp_manifest_path}")

    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """
        Retrieve experiment manifest by ID (most recent version).
        """
        exp_dir = self._layout.experiments_dir_path()
        if not exp_dir.exists():
            return None

        # Find manifest files matching this experiment_id
        safe_id = safe_experiment_id(experiment_id)
        matching = sorted(
            exp_dir.glob(f"{safe_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not matching:
            return None

        try:
            return json.loads(matching[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load experiment manifest {matching[0]}: {e}")
            return None

    # =========================================================================
    # Utility methods
    # =========================================================================

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its artifacts/logs/derived metrics/results."""
        with self._run_lock(run_id):
            # Delete run record
            run_path = self._layout.run_path(run_id)
            if run_path.exists():
                run_path.unlink()

            # Delete derived metrics
            derived_path = self._layout.derived_path(run_id)
            if derived_path.exists():
                derived_path.unlink()

            # Delete results
            result_dir = self._layout.result_dir(run_id)
            if result_dir.exists():
                shutil.rmtree(result_dir)

            # Delete artifacts
            artifact_dir = self._layout.artifact_dir(run_id)
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)

            # Delete logs - handle current and legacy formats
            log_dir = self._layout.logs_dir_path()

            # Delete current format: {run_id}_*.log
            for log_file in log_dir.glob(f"{run_id}_*.log"):
                log_file.unlink()

            # Delete legacy label format: *_{short_id}_*.log
            short_id = run_id[:8]
            for log_file in log_dir.glob(f"*_{short_id}_*.log"):
                log_file.unlink()

            # Delete legacy nested format
            nested_log_dir = log_dir / run_id
            if nested_log_dir.exists():
                shutil.rmtree(nested_log_dir)

        # Delete lock file
        lock_path = self._layout.lock_path(run_id)
        if lock_path.exists():
            lock_path.unlink()
