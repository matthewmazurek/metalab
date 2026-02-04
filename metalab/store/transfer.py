"""
Store transfer: Export and import between different store backends.

Provides:

- export_store: Copy data from source store to destination store

This enables:

- Store-to-store migration
- Backup and archival
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metalab.store.capabilities import SupportsLogListing
from metalab.store.locator import create_store

if TYPE_CHECKING:
    from metalab.store.base import Store

logger = logging.getLogger(__name__)


def export_store(
    source: str | "Store",
    destination: str | "Store",
    *,
    experiment_id: str | None = None,
    include_derived: bool = True,
    include_logs: bool = True,
    include_artifacts: bool = False,
    overwrite: bool = False,
    progress_callback: "callable[[int, int], None] | None" = None,
    **kwargs: Any,
) -> dict[str, int]:
    """
    Export data from source store to destination store.

    Args:
        source: Source store locator or instance.
        destination: Destination store locator or instance.
        experiment_id: Optional filter by experiment.
        include_derived: Include derived metrics.
        include_logs: Include log files.
        include_artifacts: Include artifact files (usually skipped for file-backed).
        overwrite: Overwrite existing records in destination.
        progress_callback: Optional callback(current, total) for progress.
        **kwargs: Additional arguments for store creation.

    Returns:
        Dict with counts: {"runs": N, "derived": N, "logs": N, "artifacts": N}
    """
    # Resolve stores
    if isinstance(source, str):
        source = create_store(source, **kwargs)
    if isinstance(destination, str):
        destination = create_store(destination, **kwargs)

    counts = {"runs": 0, "derived": 0, "logs": 0, "artifacts": 0}

    # Get run records from source
    records = source.list_run_records(experiment_id)
    total = len(records)

    logger.info(
        f"Exporting {total} runs from {type(source).__name__} to {type(destination).__name__}"
    )

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
            if isinstance(source, SupportsLogListing):
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
        if include_artifacts:
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
