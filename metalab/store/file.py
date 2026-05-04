"""Filesystem-only v4 hash-sharded run store for HPC execution."""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
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
from metalab.store.events import FileEventSink
from metalab.store.layout import (
    LAYOUT_VERSION,
    OUTPUTS_LAYOUT,
    FileStoreLayout,
    safe_experiment_id,
)
from metalab.store.records import (
    iter_ndjson_rows,
    latest_run_index_entries,
    latest_run_index_entry,
    read_run_at_index,
)
from metalab.types import ArtifactDescriptor, RunRecord, Status

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

    # Or point the runner at a run-store root directly:
    metalab.run(exp, store="./runs/my_exp")
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

        Discovers experiments by scanning subdirectories for .metalab/meta.json files
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
            meta_path = FileStoreLayout(child).meta_path()
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

        shard_count = 64
        meta_path = FileStoreLayout(effective_root).meta_path()
        if not meta_path.exists() and (
            (effective_root / "_meta.json").exists()
            or (effective_root / "runs" / "shards").exists()
        ):
            raise ValueError(
                f"Unsupported metalab file store layout at {effective_root}: "
                f"expected layout_version={LAYOUT_VERSION}"
            )
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                shard_count = int(meta.get("shard_count", shard_count))
            except Exception:
                pass

        self._layout = FileStoreLayout(effective_root, shard_count=shard_count)
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
                "layout_version": LAYOUT_VERSION,
                "outputs_layout": OUTPUTS_LAYOUT,
                "shard_hash": "sha256",
                "shard_count": self._layout.shard_count,
                "record_schema_version": SCHEMA_VERSION,
                "created_by": "metalab",
            }
            # Include experiment_id for explicitly scoped stores.
            if self._config.experiment_id:
                meta_data["experiment_id"] = self._config.experiment_id
            self._atomic_write_json(meta_path, meta_data)
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise ValueError(f"Malformed metalab store metadata: {meta_path}") from e
            if meta.get("layout_version") != LAYOUT_VERSION:
                raise ValueError(
                    f"Unsupported metalab file store layout at {self._layout.root}: "
                    f"expected layout_version={LAYOUT_VERSION}"
                )

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

    @staticmethod
    def _json_line(data: dict[str, Any]) -> bytes:
        """Serialize one compact JSON row for NDJSON append."""
        return (
            json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")

    @contextmanager
    def _run_lock(self, run_id: str) -> Generator[None, None, None]:
        """Acquire the shard lock for a run id using flock."""
        lock_path = self._layout.lock_path(run_id)
        with self._path_lock(lock_path):
            yield

    @contextmanager
    def _path_lock(self, lock_path: Path) -> Generator[None, None, None]:
        """Acquire a filesystem lock at an arbitrary path."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch()

        with lock_path.open("r") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _append_json_row(self, path: Path, row: dict[str, Any]) -> tuple[int, int]:
        """Append one NDJSON row and return byte offset and length."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = self._json_line(row)
        with path.open("ab") as handle:
            offset = handle.tell()
            handle.write(line)
            handle.flush()
        return offset, len(line)

    def _iter_ndjson(self, path: Path) -> Generator[dict[str, Any], None, None]:
        """Yield JSON rows from an NDJSON file, skipping malformed rows."""
        yield from iter_ndjson_rows(path)

    def _latest_run_index_entries(self) -> dict[str, dict[str, Any]]:
        """Return latest run index entry per run id by sequence."""
        return latest_run_index_entries(self._layout)

    def _record_shard_freshness(self) -> dict[str, list[int]]:
        """Return cheap freshness signals for canonical run shards."""
        freshness: dict[str, list[int]] = {}
        for path in sorted(self._layout.record_shards_dir_path().glob("*.ndjson")):
            stat = path.stat()
            freshness[str(path.relative_to(self._layout.root))] = [
                stat.st_size,
                stat.st_mtime_ns,
            ]
        return freshness

    def _load_success_cache(self) -> set[str] | None:
        """Load fresh successful run ids, or return None when stale/missing."""
        path = self._layout.success_cache_path()
        if not path.exists():
            return None
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if cache.get("layout_version") != 1:
            return None
        if cache.get("shards") != self._record_shard_freshness():
            return None
        run_ids = cache.get("success_run_ids")
        if not isinstance(run_ids, list):
            return None
        return {run_id for run_id in run_ids if isinstance(run_id, str)}

    def _write_success_cache(self, success_run_ids: set[str]) -> None:
        """Atomically write the rebuildable successful-run cache."""
        self._atomic_write_json(
            self._layout.success_cache_path(),
            {
                "layout_version": 1,
                "built_at": datetime.now().isoformat(),
                "shards": self._record_shard_freshness(),
                "success_run_ids": sorted(success_run_ids),
            },
        )

    def _success_run_ids(self) -> set[str]:
        """Return successful run ids using a freshness-checked cache."""
        cached = self._load_success_cache()
        if cached is not None:
            return cached

        success_ids: set[str] = set()
        for row in self._latest_run_index_entries().values():
            record = self._read_run_at_index(row)
            if record is not None and record.status == Status.SUCCESS:
                success_ids.add(record.run_id)
        self._write_success_cache(success_ids)
        return success_ids

    def _latest_run_index_entry(self, run_id: str) -> dict[str, Any] | None:
        """Return latest run index entry for a run id."""
        return latest_run_index_entry(self._layout, run_id)

    def _read_run_at_index(self, row: dict[str, Any]) -> RunRecord | None:
        """Read a run record via a byte index row."""
        record_ref = read_run_at_index(self._layout, row)
        return record_ref[0] if record_ref is not None else None

    def _append_output(
        self,
        kind: str,
        run_id: str,
        row: dict[str, Any],
    ) -> None:
        """Append one packed output row."""
        sequence = time.time_ns()
        envelope = {
            "run_id": run_id,
            "kind": kind,
            "sequence": sequence,
            "written_at": datetime.now().isoformat(),
            **row,
        }
        with self._run_lock(run_id):
            self._append_json_row(self._layout.output_shard_path(kind, run_id), envelope)

    def _latest_output_rows(
        self,
        kind: str,
        run_id: str,
        *,
        key_field: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return latest output rows for a run, keyed by field or singleton."""
        latest: dict[str, dict[str, Any]] = {}
        for row in self._iter_ndjson(self._layout.output_shard_path(kind, run_id)):
            if row.get("run_id") != run_id:
                continue
            key = str(row.get(key_field)) if key_field else "_"
            prev = latest.get(key)
            if prev is None or int(row.get("sequence", 0)) >= int(
                prev.get("sequence", 0)
            ):
                latest[key] = row
        return latest

    # =========================================================================
    # Run record operations
    # =========================================================================

    def put_run_record(self, record: RunRecord) -> None:
        """Persist a run record as an append-only canonical shard row."""
        shard_id = self._layout.shard_id(record.run_id)
        data_path = self._layout.record_shard_path(record.run_id)
        index_path = self._layout.record_shard_index_path(record.run_id)
        sequence = time.time_ns()
        with self._run_lock(record.run_id):
            data = dump_run_record(record)
            offset, length = self._append_json_row(data_path, data)
            index_row = {
                "run_id": record.run_id,
                "shard_id": shard_id,
                "offset": offset,
                "length": length,
                "status": record.status.value,
                "sequence": sequence,
            }
            self._append_json_row(index_path, index_row)
            with self._path_lock(self._layout.index_dir_path() / "shard-map.lock"):
                self._append_json_row(self._layout.shard_map_path(), index_row)

    def get_run_record(self, run_id: str) -> RunRecord | None:
        """Retrieve a run record by ID."""
        index_row = self._latest_run_index_entry(run_id)
        if index_row is None:
            return None
        return self._read_run_at_index(index_row)

    def list_run_records(self, experiment_id: str | None = None) -> list[RunRecord]:
        """List run records, optionally filtered by experiment."""
        records = []
        for row in self._latest_run_index_entries().values():
            record = self._read_run_at_index(row)
            if record is None:
                continue
            if experiment_id is None or record.experiment_id == experiment_id:
                records.append(record)

        return records

    def run_exists(self, run_id: str) -> bool:
        """Check if a run record exists."""
        return self.get_run_record(run_id) is not None

    def get_run_statuses(self, run_ids: list[str]) -> dict[str, Status]:
        """Bulk lookup successful run statuses for resume."""
        success_ids = self._success_run_ids()
        return {
            run_id: Status.SUCCESS
            for run_id in run_ids
            if run_id in success_ids
        }

    def successful_run_ids(self, experiment_id: str) -> set[str]:
        """Return successful run ids for an experiment."""
        return {
            r.run_id
            for r in self.list_run_records(experiment_id)
            if r.status == Status.SUCCESS
        }

    def write_root_manifest(self, manifest: dict[str, Any]) -> None:
        """Write the root run-store manifest."""
        self._atomic_write_json(
            self._layout.root_manifest_path(),
            {
                "layout_version": LAYOUT_VERSION,
                "outputs_layout": OUTPUTS_LAYOUT,
                "shard_hash": "sha256",
                "shard_count": self._layout.shard_count,
                "record_schema_version": SCHEMA_VERSION,
                **manifest,
            },
        )
        self._write_shard_manifests()

    def _write_shard_manifests(self) -> None:
        """Write compact manifests for canonical packed output layout."""
        common = {
            "layout_version": LAYOUT_VERSION,
            "outputs_layout": OUTPUTS_LAYOUT,
            "shard_hash": "sha256",
            "shard_count": self._layout.shard_count,
        }
        self._atomic_write_json(
            self._layout.record_shards_manifest_path(),
            {
                **common,
                "record_schema_version": SCHEMA_VERSION,
                "shards": [
                    self._layout.records_manifest_reference(f"{idx:04d}")
                    for idx in range(self._layout.shard_count)
                ],
            },
        )
        self._atomic_write_json(
            self._layout.outputs_manifest_path(),
            {
                **common,
                "kinds": ["results", "artifacts", "logs"],
            },
        )

    def rebuild_shard_indexes(self) -> None:
        """Rebuild run-record shard indexes and shard-map from canonical shards."""
        latest_rows: list[dict[str, Any]] = []
        for idx_path in self._layout.record_shards_dir_path().glob("*.idx"):
            idx_path.unlink()
        shard_map = self._layout.shard_map_path()
        if shard_map.exists():
            shard_map.unlink()

        sequence = 0
        for data_path in sorted(self._layout.record_shards_dir_path().glob("*.ndjson")):
            shard_id = data_path.stem
            idx_path = data_path.with_suffix(".idx")
            offset = 0
            with data_path.open("rb") as handle:
                for raw_line in handle:
                    length = len(raw_line)
                    try:
                        data = json.loads(raw_line.decode("utf-8"))
                        record = load_run_record(data)
                    except Exception as e:
                        logger.warning(f"Skipping malformed run shard row {data_path}: {e}")
                        offset += length
                        continue
                    sequence += 1
                    row = {
                        "run_id": record.run_id,
                        "shard_id": shard_id,
                        "offset": offset,
                        "length": length,
                        "status": record.status.value,
                        "sequence": sequence,
                    }
                    self._append_json_row(idx_path, row)
                    latest_rows.append(row)
                    offset += length

        for row in latest_rows:
            self._append_json_row(shard_map, row)

    def write_planned_run_ids(self, run_ids: list[str]) -> None:
        """Write expected run IDs to sharded sidecar files."""
        shards: dict[str, list[str]] = {}
        for run_id in run_ids:
            shards.setdefault(run_id[:2], []).append(run_id)

        planned_dir = self._layout.planned_runs_dir_path()
        planned_dir.mkdir(parents=True, exist_ok=True)
        for prefix, shard_run_ids in shards.items():
            content = "".join(
                json.dumps(
                    {
                        "run_id": run_id,
                        "shard_id": self._layout.shard_id(run_id),
                    },
                    sort_keys=True,
                )
                + "\n"
                for run_id in shard_run_ids
            )
            self._atomic_write(
                self._layout.planned_runs_path(prefix),
                content.encode("utf-8"),
            )

    def get_root_manifest(self) -> dict[str, Any] | None:
        """Read the root run-store manifest."""
        path = self._layout.root_manifest_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def event_sink(self, job_id: str, worker_id: str, experiment_id: str) -> FileEventSink:
        """Create an append-only event sink for a worker."""
        return FileEventSink(
            self._layout.event_log_path(job_id, worker_id),
            experiment_id=experiment_id,
            job_id=job_id,
            worker_id=worker_id,
        )

    def put_heartbeat(
        self,
        *,
        job_id: str,
        worker_id: str,
        experiment_id: str,
        state: str,
        current_run_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Update a worker heartbeat file."""
        self._atomic_write_json(
            self._layout.heartbeat_path(job_id, worker_id),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "experiment_id": experiment_id,
                "state": state,
                "current_run_id": current_run_id,
                "updated_at": datetime.now().isoformat(),
                "payload": payload or {},
            },
        )

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
            # Fallback: try to extract from metalab scratch paths.
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

        # Copy or write the artifact payload. The packed output append below
        # takes the shard lock; avoid recursively acquiring the same flock.
        if isinstance(data, Path):
            shutil.copy2(data, dest_path)
        else:
            self._atomic_write(dest_path, data)

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
        """Update the packed artifact descriptor output for a run."""
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
        self._append_output(
            "artifacts",
            run_id,
            {
                "name": descriptor.name,
                "descriptor": updated_descriptor,
            },
        )

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
        rows = self._latest_output_rows("artifacts", run_id, key_field="name")
        artifacts = []
        for row in rows.values():
            descriptor = row.get("descriptor")
            if isinstance(descriptor, dict):
                artifacts.append(load_artifact_descriptor(descriptor))
        return sorted(artifacts, key=lambda item: item.name)

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

        Results are stored as append-only packed output rows.
        """
        result_obj = {
            "data": data,
            "dtype": dtype,
            "shape": shape,
            "metadata": metadata or {},
        }
        self._append_output(
            "results",
            run_id,
            {
                "name": name,
                "result": result_obj,
            },
        )

    def get_result(self, run_id: str, name: str) -> dict[str, Any] | None:
        """Retrieve structured result data."""
        rows = self._latest_output_rows("results", run_id, key_field="name")
        row = rows.get(name)
        if row is None:
            return None
        result = row.get("result")
        return result if isinstance(result, dict) else None

    def list_results(self, run_id: str) -> list[str]:
        """List result names for a run."""
        return sorted(self._latest_output_rows("results", run_id, key_field="name"))

    # =========================================================================
    # Log operations
    # =========================================================================

    def put_log(
        self,
        run_id: str,
        name: str,
        content: str,
        label: str | None = None,
    ) -> None:
        """Store finalized log content as a packed output."""
        self._append_output(
            "logs",
            run_id,
            {
                "name": name,
                "content": content,
                "label": label,
            },
        )

    def get_log_path(self, run_id: str, name: str) -> Path:
        """
        Get the path where a log file should be written.

        Enables streaming loggers to write directly to the store.
        """
        log_path = self._layout.scratch_log_path(run_id, name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path

    def get_log(self, run_id: str, name: str) -> str | None:
        """
        Retrieve a log file.

        Searches packed log outputs for the run id.
        """
        rows = self._latest_output_rows("logs", run_id, key_field="name")
        row = rows.get(name)
        content = row.get("content") if row else None
        return content if isinstance(content, str) else None

    def list_logs(self, run_id: str) -> list[str]:
        """
        List available log names for a run.

        Returns a list of log names (e.g., ["run", "stdout", "stderr"]).
        """
        return sorted(self._latest_output_rows("logs", run_id, key_field="name"))

    # =========================================================================
    # Capability: SupportsExperimentManifests
    # =========================================================================

    def put_experiment_manifest(
        self,
        experiment_id: str,
        manifest: dict[str, Any],
        timestamp: str | None = None,
    ) -> None:
        """Append an experiment submission manifest."""
        from datetime import datetime

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        row = {
            "experiment_id": experiment_id,
            "submitted_at": manifest.get("submitted_at") or datetime.now().isoformat(),
            "timestamp": timestamp,
            "job_id": manifest.get("job_id"),
            "executor_type": manifest.get("executor_type"),
            "manifest": manifest,
        }
        self._append_json_row(self._layout.submissions_path(), row)
        logger.debug(f"Appended submission manifest to {self._layout.submissions_path()}")

    def get_experiment_manifest(self, experiment_id: str) -> dict[str, Any] | None:
        """
        Retrieve experiment manifest by ID (most recent submission).
        """
        latest: dict[str, Any] | None = None
        for row in self._iter_ndjson(self._layout.submissions_path()):
            manifest = row.get("manifest")
            row_experiment_id = row.get("experiment_id")
            if row_experiment_id != experiment_id or not isinstance(manifest, dict):
                continue
            latest = manifest
        return latest

    # =========================================================================
    # Utility methods
    # =========================================================================

    def delete_run(self, run_id: str) -> None:
        """Delete non-canonical payload/scratch files for a run.

        Canonical append-only shard entries are immutable. Deleting a canonical
        run would require a future tombstone/compaction operation.
        """
        with self._run_lock(run_id):
            artifact_dir = self._layout.artifact_dir(run_id)
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            scratch_log = self._layout.scratch_log_path(run_id, "run")
            if scratch_log.exists():
                scratch_log.unlink()

        # Delete lock file
        lock_path = self._layout.lock_path(run_id)
        if lock_path.exists():
            lock_path.unlink()
