"""
Store transfer: Export and import between different store backends.

Provides:
- export_store: Copy data from source store to destination store
- import_to_postgres: Import FileStore data into PostgresStore
- export_to_filestore: Export PostgresStore data to FileStore

This enables:
- Postgres → FileStore export for offline analysis/Atlas compatibility
- FileStore → Postgres import for backfilling existing data
- Migration between different storage backends
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metalab.schema import dump_run_record
from metalab.store.locator import create_store, parse_locator

if TYPE_CHECKING:
    from metalab.store.base import Store
    from metalab.store.file import FileStore
    from metalab.store.postgres import PostgresStore
    from metalab.types import RunRecord

logger = logging.getLogger(__name__)


def export_store(
    source: str | "Store",
    destination: str | "Store",
    *,
    experiment_id: str | None = None,
    include_derived: bool = True,
    include_logs: bool = True,
    include_artifacts: bool = False,  # Artifacts are usually file-backed
    overwrite: bool = False,
    progress_callback: "callable[[int, int], None] | None" = None,
) -> dict[str, int]:
    """
    Export data from source store to destination store.
    
    Args:
        source: Source store locator or instance.
        destination: Destination store locator or instance.
        experiment_id: Optional filter by experiment.
        include_derived: Include derived metrics.
        include_logs: Include log files.
        include_artifacts: Include artifact files (usually skipped).
        overwrite: Overwrite existing records in destination.
        progress_callback: Optional callback(current, total) for progress.
    
    Returns:
        Dict with counts: {"runs": N, "derived": N, "logs": N, "artifacts": N}
    """
    # Resolve stores
    if isinstance(source, str):
        source = create_store(source)
    if isinstance(destination, str):
        destination = create_store(destination)
    
    counts = {"runs": 0, "derived": 0, "logs": 0, "artifacts": 0}
    
    # Get run records from source
    records = source.list_run_records(experiment_id)
    total = len(records)
    
    logger.info(f"Exporting {total} runs from {type(source).__name__} to {type(destination).__name__}")
    
    for i, record in enumerate(records):
        run_id = record.run_id
        
        # Check if exists in destination
        if not overwrite and destination.run_exists(run_id):
            logger.debug(f"Skipping existing run: {run_id}")
            continue
        
        # Export run record
        destination.put_run_record(record)
        counts["runs"] += 1
        
        # Export derived metrics
        if include_derived:
            derived = source.get_derived(run_id)
            if derived:
                destination.put_derived(run_id, derived)
                counts["derived"] += 1
        
        # Export logs
        if include_logs:
            # List logs if the store supports it
            if hasattr(source, "list_logs"):
                log_names = source.list_logs(run_id)
                for log_name in log_names:
                    content = source.get_log(run_id, log_name)
                    if content:
                        destination.put_log(run_id, log_name, content)
                        counts["logs"] += 1
            else:
                # Try common log names
                for log_name in ["run", "stdout", "stderr"]:
                    content = source.get_log(run_id, log_name)
                    if content:
                        destination.put_log(run_id, log_name, content)
                        counts["logs"] += 1
        
        # Export artifacts (usually skipped since they're file-backed)
        if include_artifacts and hasattr(source, "list_artifacts"):
            artifacts = source.list_artifacts(run_id)
            for artifact in artifacts:
                try:
                    data = source.get_artifact(artifact.uri)
                    destination.put_artifact(data, artifact)
                    counts["artifacts"] += 1
                except Exception as e:
                    logger.warning(f"Failed to export artifact {artifact.name}: {e}")
        
        # Progress callback
        if progress_callback:
            progress_callback(i + 1, total)
    
    logger.info(f"Export complete: {counts}")
    return counts


def export_to_filestore(
    source: str | "PostgresStore",
    destination: str | Path,
    *,
    experiment_id: str | None = None,
    include_manifests: bool = True,
    progress_callback: "callable[[int, int], None] | None" = None,
) -> dict[str, int]:
    """
    Export PostgresStore data to FileStore layout.
    
    Creates the standard FileStore layout:
        {root}/
        ├── runs/{run_id}.json
        ├── derived/{run_id}.json
        ├── logs/{run_id}_{name}.log
        ├── experiments/{exp_id}_{timestamp}.json
        └── _meta.json
    
    Args:
        source: PostgresStore locator or instance.
        destination: FileStore path or locator.
        experiment_id: Optional filter by experiment.
        include_manifests: Include experiment manifests.
        progress_callback: Optional callback(current, total) for progress.
    
    Returns:
        Dict with counts.
    """
    from metalab.schema import SCHEMA_VERSION
    from metalab.store.file import FileStore
    
    # Resolve stores
    if isinstance(source, str):
        source = create_store(source)
    if isinstance(destination, (str, Path)):
        dest_path = Path(destination)
        destination = FileStore(dest_path)
    
    counts = export_store(
        source,
        destination,
        experiment_id=experiment_id,
        include_derived=True,
        include_logs=True,
        include_artifacts=False,  # Artifacts are file-backed
        progress_callback=progress_callback,
    )
    
    # Export experiment manifests
    if include_manifests and hasattr(source, "_conn"):
        with source._conn() as conn:
            with conn.cursor() as cur:
                where = ""
                params: list[Any] = []
                if experiment_id:
                    where = "WHERE experiment_id = %s"
                    params = [experiment_id]
                
                cur.execute(f"""
                    SELECT experiment_id, timestamp, manifest_json
                    FROM {source._table('experiment_manifests')}
                    {where}
                """, params)
                
                exp_dir = destination.root / "experiments"
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                for exp_id, timestamp, manifest_json in cur.fetchall():
                    data = manifest_json if isinstance(manifest_json, dict) else json.loads(manifest_json)
                    safe_id = exp_id.replace(":", "_")
                    manifest_path = exp_dir / f"{safe_id}_{timestamp}.json"
                    manifest_path.write_text(json.dumps(data, indent=2))
                    counts["manifests"] = counts.get("manifests", 0) + 1
    
    logger.info(f"Exported to FileStore at {destination.root}: {counts}")
    return counts


def import_from_filestore(
    source: str | Path,
    destination: str | "PostgresStore",
    *,
    experiment_id: str | None = None,
    include_manifests: bool = True,
    update_field_catalog: bool = True,
    progress_callback: "callable[[int, int], None] | None" = None,
) -> dict[str, int]:
    """
    Import FileStore data into PostgresStore.
    
    Reads from the standard FileStore layout and imports into Postgres.
    
    Args:
        source: FileStore path or locator.
        destination: PostgresStore locator or instance.
        experiment_id: Optional filter by experiment.
        include_manifests: Include experiment manifests.
        update_field_catalog: Update field catalog for Atlas.
        progress_callback: Optional callback(current, total) for progress.
    
    Returns:
        Dict with counts.
    """
    from metalab.store.file import FileStore
    
    # Resolve stores
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        # Check if it's a file:// URL
        if isinstance(source, str) and source.startswith("file://"):
            source_path = Path(source.replace("file://", ""))
        source = FileStore(source_path)
    
    if isinstance(destination, str):
        destination = create_store(destination)
    
    counts = export_store(
        source,
        destination,
        experiment_id=experiment_id,
        include_derived=True,
        include_logs=True,
        include_artifacts=False,
        progress_callback=progress_callback,
    )
    
    # Import experiment manifests
    if include_manifests:
        exp_dir = source.root / "experiments"
        if exp_dir.exists():
            for manifest_file in exp_dir.glob("*.json"):
                try:
                    data = json.loads(manifest_file.read_text())
                    exp_id = data.get("experiment_id")
                    
                    # Filter by experiment_id if specified
                    if experiment_id and exp_id != experiment_id:
                        continue
                    
                    # Extract timestamp from filename
                    # Format: {safe_id}_{timestamp}.json
                    name_parts = manifest_file.stem.rsplit("_", 1)
                    timestamp = name_parts[-1] if len(name_parts) > 1 else datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if hasattr(destination, "put_experiment_manifest"):
                        destination.put_experiment_manifest(exp_id, data, timestamp)
                        counts["manifests"] = counts.get("manifests", 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to import manifest {manifest_file}: {e}")
    
    # Update field catalog for Atlas
    if update_field_catalog and hasattr(destination, "update_field_catalog"):
        logger.info("Updating field catalog...")
        destination.update_field_catalog()
    
    logger.info(f"Imported from FileStore at {source.root}: {counts}")
    return counts


class FallbackStore:
    """
    Store wrapper that falls back to a secondary store when primary fails.
    
    Useful for maintaining FileStore compatibility when Postgres is unavailable.
    Write operations go to primary (with fallback to secondary).
    Read operations try primary first, then secondary.
    """
    
    def __init__(
        self,
        primary: "Store",
        fallback: "Store",
        *,
        write_to_both: bool = False,
    ) -> None:
        """
        Initialize FallbackStore.
        
        Args:
            primary: Primary store (usually PostgresStore).
            fallback: Fallback store (usually FileStore).
            write_to_both: If True, write to both stores.
        """
        self._primary = primary
        self._fallback = fallback
        self._write_to_both = write_to_both
        self._primary_available = True
    
    def _check_primary(self) -> bool:
        """Check if primary store is available."""
        try:
            # Try a simple operation
            if hasattr(self._primary, "run_exists"):
                self._primary.run_exists("__health_check__")
            self._primary_available = True
            return True
        except Exception:
            self._primary_available = False
            logger.warning("Primary store unavailable, using fallback")
            return False
    
    @property
    def locator(self) -> str:
        """The store locator URI (primary's locator)."""
        if hasattr(self._primary, "locator"):
            return self._primary.locator
        if hasattr(self._fallback, "locator"):
            return self._fallback.locator
        return ""
    
    # Run record operations
    
    def put_run_record(self, record: "RunRecord") -> None:
        """Persist a run record."""
        wrote_to_primary = False
        
        if self._primary_available or self._check_primary():
            try:
                self._primary.put_run_record(record)
                wrote_to_primary = True
            except Exception as e:
                logger.warning(f"Primary store write failed: {e}")
                self._primary_available = False
        
        if self._write_to_both or not wrote_to_primary:
            self._fallback.put_run_record(record)
    
    def get_run_record(self, run_id: str) -> "RunRecord | None":
        """Retrieve a run record by ID."""
        if self._primary_available or self._check_primary():
            try:
                result = self._primary.get_run_record(run_id)
                if result is not None:
                    return result
            except Exception:
                self._primary_available = False
        
        return self._fallback.get_run_record(run_id)
    
    def list_run_records(self, experiment_id: str | None = None) -> "list[RunRecord]":
        """List run records."""
        if self._primary_available or self._check_primary():
            try:
                return self._primary.list_run_records(experiment_id)
            except Exception:
                self._primary_available = False
        
        return self._fallback.list_run_records(experiment_id)
    
    def run_exists(self, run_id: str) -> bool:
        """Check if a run record exists."""
        if self._primary_available or self._check_primary():
            try:
                if self._primary.run_exists(run_id):
                    return True
            except Exception:
                self._primary_available = False
        
        return self._fallback.run_exists(run_id)
    
    # Artifact operations
    
    def put_artifact(self, data: "bytes | Path", descriptor: "ArtifactDescriptor") -> "ArtifactDescriptor":
        """Store an artifact."""
        wrote_to_primary = False
        result = descriptor
        
        if self._primary_available or self._check_primary():
            try:
                result = self._primary.put_artifact(data, descriptor)
                wrote_to_primary = True
            except Exception as e:
                logger.warning(f"Primary store artifact write failed: {e}")
                self._primary_available = False
        
        if self._write_to_both or not wrote_to_primary:
            result = self._fallback.put_artifact(data, descriptor)
        
        return result
    
    def get_artifact(self, uri: str) -> bytes:
        """Retrieve artifact data."""
        if self._primary_available or self._check_primary():
            try:
                return self._primary.get_artifact(uri)
            except FileNotFoundError:
                pass  # Try fallback
            except Exception:
                self._primary_available = False
        
        return self._fallback.get_artifact(uri)
    
    def list_artifacts(self, run_id: str) -> "list[ArtifactDescriptor]":
        """List artifacts for a run."""
        if self._primary_available or self._check_primary():
            try:
                return self._primary.list_artifacts(run_id)
            except Exception:
                self._primary_available = False
        
        return self._fallback.list_artifacts(run_id)
    
    # Derived metrics operations
    
    def put_derived(self, run_id: str, derived: "dict[str, Metric]") -> None:
        """Persist derived metrics."""
        wrote_to_primary = False
        
        if self._primary_available or self._check_primary():
            try:
                self._primary.put_derived(run_id, derived)
                wrote_to_primary = True
            except Exception as e:
                logger.warning(f"Primary store derived write failed: {e}")
                self._primary_available = False
        
        if self._write_to_both or not wrote_to_primary:
            self._fallback.put_derived(run_id, derived)
    
    def get_derived(self, run_id: str) -> "dict[str, Metric] | None":
        """Retrieve derived metrics."""
        if self._primary_available or self._check_primary():
            try:
                result = self._primary.get_derived(run_id)
                if result is not None:
                    return result
            except Exception:
                self._primary_available = False
        
        return self._fallback.get_derived(run_id)
    
    def derived_exists(self, run_id: str) -> bool:
        """Check if derived metrics exist."""
        if self._primary_available or self._check_primary():
            try:
                if self._primary.derived_exists(run_id):
                    return True
            except Exception:
                self._primary_available = False
        
        return self._fallback.derived_exists(run_id)
    
    # Log operations
    
    def put_log(self, run_id: str, name: str, content: str, label: str | None = None) -> None:
        """Store a log file."""
        wrote_to_primary = False
        
        if self._primary_available or self._check_primary():
            try:
                # Try with label first, fall back to without
                try:
                    self._primary.put_log(run_id, name, content, label)
                except TypeError:
                    self._primary.put_log(run_id, name, content)
                wrote_to_primary = True
            except Exception as e:
                logger.warning(f"Primary store log write failed: {e}")
                self._primary_available = False
        
        if self._write_to_both or not wrote_to_primary:
            # Try with label first, fall back to without
            try:
                self._fallback.put_log(run_id, name, content, label)
            except TypeError:
                self._fallback.put_log(run_id, name, content)
    
    def get_log(self, run_id: str, name: str) -> str | None:
        """Retrieve a log file."""
        if self._primary_available or self._check_primary():
            try:
                result = self._primary.get_log(run_id, name)
                if result is not None:
                    return result
            except Exception:
                self._primary_available = False
        
        return self._fallback.get_log(run_id, name)
    
    # Manifest operations
    
    def get_experiment_manifest(self, experiment_id: str) -> "dict[str, Any] | None":
        """Retrieve experiment manifest."""
        if self._primary_available or self._check_primary():
            try:
                result = self._primary.get_experiment_manifest(experiment_id)
                if result is not None:
                    return result
            except Exception:
                self._primary_available = False
        
        return self._fallback.get_experiment_manifest(experiment_id)
