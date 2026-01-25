"""
FileStore: Filesystem-based storage backend.

Layout:
    {root}/
    ├── runs/
    │   └── {run_id}.json           # RunRecord (schema-versioned)
    ├── artifacts/
    │   └── {run_id}/
    │       ├── {name}.{ext}        # Artifact files
    │       └── _manifest.json      # ArtifactDescriptor list
    ├── logs/
    │   └── {run_id}/
    │       ├── stdout.txt
    │       └── stderr.txt
    ├── .locks/
    │   └── {run_id}.lock           # Per-run lock files
    └── _meta.json                  # Store metadata (schema version)

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
from pathlib import Path
from typing import Generator

from metalab.schema import (
    SCHEMA_VERSION,
    dump_artifact_descriptor,
    dump_run_record,
    load_artifact_descriptor,
    load_run_record,
)
from metalab.types import ArtifactDescriptor, RunRecord

logger = logging.getLogger(__name__)


class FileStore:
    """
    Filesystem-based storage backend.

    Provides atomic writes and per-run locking for concurrent access.
    """

    # Layout constants
    RUNS_DIR = "runs"
    ARTIFACTS_DIR = "artifacts"
    LOGS_DIR = "logs"
    LOCKS_DIR = ".locks"
    META_FILE = "_meta.json"
    MANIFEST_FILE = "_manifest.json"

    def __init__(self, root: str | Path) -> None:
        """
        Initialize the file store.

        Args:
            root: Root directory for storage.
        """
        self._root = Path(root)
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        """Create the directory structure if needed."""
        (self._root / self.RUNS_DIR).mkdir(parents=True, exist_ok=True)
        (self._root / self.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
        (self._root / self.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        (self._root / self.LOCKS_DIR).mkdir(parents=True, exist_ok=True)

        # Write meta file if it doesn't exist
        meta_path = self._root / self.META_FILE
        if not meta_path.exists():
            self._atomic_write_json(meta_path, {
                "schema_version": SCHEMA_VERSION,
                "created_by": "metalab",
            })

    @property
    def root(self) -> Path:
        """The root directory of this store."""
        return self._root

    # Atomic write utilities

    def _atomic_write(self, path: Path, data: bytes) -> None:
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

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        """Write JSON atomically."""
        content = json.dumps(data, indent=2, sort_keys=True)
        self._atomic_write(path, content.encode("utf-8"))

    @contextmanager
    def _run_lock(self, run_id: str) -> Generator[None, None, None]:
        """Acquire a per-run lock using flock."""
        lock_path = self._root / self.LOCKS_DIR / f"{run_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()

        with lock_path.open("r") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # Run record operations

    def put_run_record(self, record: RunRecord) -> None:
        """Persist a run record atomically."""
        path = self._root / self.RUNS_DIR / f"{record.run_id}.json"

        with self._run_lock(record.run_id):
            data = dump_run_record(record)
            self._atomic_write_json(path, data)

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID."""
        path = self._root / self.RUNS_DIR / f"{run_id}.json"

        if not path.exists():
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        return load_run_record(data)

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """List run records, optionally filtered by experiment."""
        records = []
        runs_dir = self._root / self.RUNS_DIR

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
        path = self._root / self.RUNS_DIR / f"{run_id}.json"
        return path.exists()

    # Artifact operations

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
                # Check parent directories for metalab temp dir pattern
                for parent in data.parents:
                    if parent.name.startswith("metalab_"):
                        # Extract run_id from temp dir name (metalab_{run_id}_xxx)
                        parts = parent.name.split("_")
                        if len(parts) >= 2:
                            run_id = parts[1]
                            break
            
            if not run_id:
                # Last resort: use artifact_id prefix
                run_id = descriptor.artifact_id[:16]

        # Determine destination path
        artifact_dir = self._root / self.ARTIFACTS_DIR / run_id
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
        artifact_dir = self._root / self.ARTIFACTS_DIR / run_id
        manifest_path = artifact_dir / self.MANIFEST_FILE

        # Load existing manifest
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"artifacts": []}

        # Add or update entry (strip internal _run_id from metadata)
        clean_metadata = {k: v for k, v in descriptor.metadata.items() if not k.startswith("_")}
        updated_descriptor = dump_artifact_descriptor(ArtifactDescriptor(
            artifact_id=descriptor.artifact_id,
            name=descriptor.name,
            kind=descriptor.kind,
            format=descriptor.format,
            uri=uri,
            content_hash=descriptor.content_hash,
            size_bytes=descriptor.size_bytes,
            metadata=clean_metadata,
        ))

        # Remove existing entry with same name (overwrite)
        manifest["artifacts"] = [
            a for a in manifest["artifacts"]
            if a.get("name") != descriptor.name
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

    def list_artifacts(self, run_id: str) -> list[ArtifactDescriptor]:
        """List artifacts for a run."""
        artifact_dir = self._root / self.ARTIFACTS_DIR / run_id
        manifest_path = artifact_dir / self.MANIFEST_FILE

        if not manifest_path.exists():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [
            load_artifact_descriptor(a)
            for a in manifest.get("artifacts", [])
        ]

    # Log operations

    def put_log(self, run_id: str, name: str, content: str) -> None:
        """Store a log file for a run."""
        log_dir = self._root / self.LOGS_DIR / run_id
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{name}.txt"
        self._atomic_write(log_path, content.encode("utf-8"))

    def get_log(self, run_id: str, name: str) -> str | None:
        """Retrieve a log file."""
        log_path = self._root / self.LOGS_DIR / run_id / f"{name}.txt"

        if not log_path.exists():
            return None

        return log_path.read_text(encoding="utf-8")

    # Utility methods

    def delete_run(self, run_id: str) -> None:
        """Delete a run and all its artifacts/logs."""
        with self._run_lock(run_id):
            # Delete run record
            run_path = self._root / self.RUNS_DIR / f"{run_id}.json"
            if run_path.exists():
                run_path.unlink()

            # Delete artifacts
            artifact_dir = self._root / self.ARTIFACTS_DIR / run_id
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)

            # Delete logs
            log_dir = self._root / self.LOGS_DIR / run_id
            if log_dir.exists():
                shutil.rmtree(log_dir)

        # Delete lock file
        lock_path = self._root / self.LOCKS_DIR / f"{run_id}.lock"
        if lock_path.exists():
            lock_path.unlink()
