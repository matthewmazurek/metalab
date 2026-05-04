#!/usr/bin/env python3
"""One-off migration from the legacy flat FileStore to the v4 HPC run store.

This intentionally lives outside the metalab CLI. It is for rescuing old
experiments so the clean-break runner can resume them safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SHARD_COUNT = 64
LAYOUT_VERSION = 4
OUTPUTS_LAYOUT = "hash-mod-sharded-ndjson"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shard_id(run_id: str, shard_count: int = DEFAULT_SHARD_COUNT) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return f"{value % shard_count:04d}"


def _read_top_level_json_string(path: Path, field: str) -> str | None:
    """Return a simple top-level string field without decoding the full record.

    Legacy records can contain large artifact metadata. Dry runs only need a
    few scalar fields for counting, so avoid paying the full JSON parse cost.
    """

    pattern = re.compile(
        rb'"' + re.escape(field.encode("utf-8")) + rb'"\s*:\s*"((?:\\.|[^"\\])*)"'
    )
    try:
        with path.open("rb") as handle:
            if path.stat().st_size == 0:
                return None
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                match = pattern.search(mapped)
                if match is None:
                    return None
                value = bytes(match.group(1))
    except OSError:
        return None
    try:
        return json.loads(b'"' + value + b'"')
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{datetime.now().timestamp()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    tmp = path.with_suffix(f".tmp.{datetime.now().timestamp()}")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def _json_line(data: dict[str, Any]) -> bytes:
    return (
        json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_record_shards(
    output_root: Path,
    records: list[dict[str, Any]],
    *,
    shard_count: int,
) -> None:
    rows_by_shard: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        rows_by_shard.setdefault(_shard_id(record["run_id"], shard_count), []).append(record)

    shard_map_rows: list[dict[str, Any]] = []
    sequence = 0
    for shard_id, rows in sorted(rows_by_shard.items()):
        data_path = output_root / "records" / f"{shard_id}.ndjson"
        idx_path = output_root / "records" / f"{shard_id}.idx"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        data_lines: list[bytes] = []
        idx_rows: list[dict[str, Any]] = []
        for record in rows:
            sequence += 1
            line = _json_line(record)
            idx_row = {
                "run_id": record["run_id"],
                "shard_id": shard_id,
                "offset": offset,
                "length": len(line),
                "status": record["status"],
                "sequence": sequence,
            }
            data_lines.append(line)
            idx_rows.append(idx_row)
            shard_map_rows.append(idx_row)
            offset += len(line)
        tmp = data_path.with_suffix(f".tmp.{datetime.now().timestamp()}")
        tmp.write_bytes(b"".join(data_lines))
        tmp.rename(data_path)
        _write_ndjson(idx_path, idx_rows)

    _write_ndjson(output_root / ".metalab" / "index" / "shard-map.ndjson", shard_map_rows)
    _write_json(
        output_root / "records" / "manifest.json",
        {
            "layout_version": LAYOUT_VERSION,
            "outputs_layout": OUTPUTS_LAYOUT,
            "shard_hash": "sha256",
            "shard_count": shard_count,
            "record_schema_version": "2",
            "shards": [
                f"records/{idx:04d}.ndjson" for idx in range(shard_count)
            ],
        },
    )


def _write_output_shards(
    output_root: Path,
    kind: str,
    rows: list[dict[str, Any]],
    *,
    shard_count: int,
) -> None:
    rows_by_shard: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_shard.setdefault(_shard_id(row["run_id"], shard_count), []).append(row)
    for shard_id, shard_rows in sorted(rows_by_shard.items()):
        if kind == "artifacts":
            path = output_root / "outputs" / "artifacts" / "metadata" / f"{shard_id}.ndjson"
        else:
            path = output_root / "outputs" / kind / f"{shard_id}.ndjson"
        _write_ndjson(path, shard_rows)


def _parse_time(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return fallback
    return fallback


def _duration_ms(data: dict[str, Any], started_at: datetime, finished_at: datetime) -> int:
    value = data.get("duration_ms")
    if isinstance(value, int | float):
        return int(value)
    return max(0, int((finished_at - started_at).total_seconds() * 1000))


def _infer_experiment_id(legacy_root: Path, records: list[dict[str, Any]]) -> str:
    meta_path = legacy_root / "_meta.json"
    if meta_path.exists():
        try:
            meta = _read_json(meta_path)
            if isinstance(meta.get("experiment_id"), str):
                return meta["experiment_id"]
        except Exception:
            pass

    for record in records:
        experiment_id = record.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id:
            return experiment_id

    return legacy_root.name


def _normalize_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    extra = value.get("extra")
    return {
        "code_hash": value.get("code_hash"),
        "python_version": value.get("python_version"),
        "metalab_version": value.get("metalab_version"),
        "executor_id": value.get("executor_id"),
        "host": value.get("host"),
        "extra": extra if isinstance(extra, dict) else {},
    }


def _normalize_artifacts(
    artifacts: Any,
    *,
    legacy_root: Path,
    output_root: Path,
    run_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(artifacts, list):
        return []

    normalized = []
    old_artifact_dir = (legacy_root / "artifacts" / run_id).resolve()
    new_artifact_dir = (
        output_root / "outputs" / "artifacts" / "files" / run_id[:2] / run_id
    ).resolve()
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        artifact = {
            "artifact_id": item.get("artifact_id", ""),
            "name": item.get("name", ""),
            "kind": item.get("kind", "blob"),
            "format": item.get("format", "binary"),
            "uri": item.get("uri", ""),
            "content_hash": item.get("content_hash"),
            "size_bytes": item.get("size_bytes"),
            "metadata": item.get("metadata", {}),
        }
        uri = artifact["uri"]
        if isinstance(uri, str) and uri:
            uri_path = Path(uri)
            try:
                resolved = uri_path.resolve()
                if resolved.is_relative_to(old_artifact_dir):
                    artifact["uri"] = str(new_artifact_dir / resolved.relative_to(old_artifact_dir))
                elif resolved.is_relative_to(legacy_root.resolve()):
                    artifact["uri"] = str(output_root.resolve() / resolved.relative_to(legacy_root.resolve()))
            except OSError:
                pass
        normalized.append(artifact)
    return normalized


def _run_record(
    data: dict[str, Any],
    *,
    run_id: str,
    experiment_id: str,
    status: str,
    legacy_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    fallback_time = datetime.fromtimestamp((legacy_root / "runs" / f"{run_id}.json").stat().st_mtime)
    started_at = _parse_time(data.get("started_at"), fallback_time)
    finished_at = _parse_time(data.get("finished_at"), started_at)
    return {
        "_schema_version": "2",
        "run_id": run_id,
        "experiment_id": data.get("experiment_id") or experiment_id,
        "status": status,
        "context_fingerprint": data.get("context_fingerprint", ""),
        "params_fingerprint": data.get("params_fingerprint", ""),
        "seed_fingerprint": data.get("seed_fingerprint", ""),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": _duration_ms(data, started_at, finished_at),
        "metrics": data.get("metrics", {}),
        "provenance": _normalize_provenance(data.get("provenance")),
        "error": data.get("error"),
        "params_resolved": data.get("params_resolved", {}),
        "tags": data.get("tags", []),
        "warnings": data.get("warnings", []),
        "notes": data.get("notes"),
        "artifacts": _normalize_artifacts(
            data.get("artifacts"),
            legacy_root=legacy_root,
            output_root=output_root,
            run_id=run_id,
        ),
    }


def _copy_tree(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _copy_sidecars(legacy_root: Path, output_root: Path, run_id: str) -> None:
    prefix = run_id[:2]
    _copy_tree(
        legacy_root / "artifacts" / run_id,
        output_root / "outputs" / "artifacts" / "files" / prefix / run_id,
    )


def _legacy_result_rows(legacy_root: Path, run_id: str, sequence_start: int) -> list[dict[str, Any]]:
    result_dir = legacy_root / "results" / run_id
    rows: list[dict[str, Any]] = []
    if not result_dir.exists():
        return rows
    sequence = sequence_start
    for path in sorted(result_dir.glob("*.json")):
        try:
            result = _read_json(path)
        except Exception:
            continue
        sequence += 1
        rows.append(
            {
                "run_id": run_id,
                "kind": "results",
                "name": path.stem,
                "result": result,
                "sequence": sequence,
                "written_at": datetime.now().isoformat(),
            }
        )
    return rows


def _legacy_log_rows(legacy_root: Path, run_id: str, sequence_start: int) -> list[dict[str, Any]]:
    logs_dir = legacy_root / "logs"
    rows: list[dict[str, Any]] = []
    if not logs_dir.exists():
        return rows
    sequence = sequence_start
    for log_path in sorted(logs_dir.glob(f"{run_id}_*.log")):
        sequence += 1
        rows.append(
            {
                "run_id": run_id,
                "kind": "logs",
                "name": log_path.stem[len(run_id) + 1 :],
                "content": log_path.read_text(encoding="utf-8"),
                "label": None,
                "sequence": sequence,
                "written_at": datetime.now().isoformat(),
            }
        )
    return rows


def _artifact_rows(record: dict[str, Any], sequence_start: int) -> list[dict[str, Any]]:
    rows = []
    sequence = sequence_start
    for artifact in record.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        sequence += 1
        rows.append(
            {
                "run_id": record["run_id"],
                "kind": "artifacts",
                "name": artifact.get("name", ""),
                "descriptor": artifact,
                "sequence": sequence,
                "written_at": datetime.now().isoformat(),
            }
        )
    return rows


def _fast_dry_run(
    record_paths: Iterable[Path],
    *,
    trust_success_without_done: bool,
) -> Counter[str]:
    """Count migratable legacy records without materializing run records."""

    counts: Counter[str] = Counter()
    planned = 0
    migrated = 0
    for path in record_paths:
        status = _read_top_level_json_string(path, "status")
        if status is None:
            counts["malformed_json"] += 1
            continue

        run_id = _read_top_level_json_string(path, "run_id") or path.stem
        if not run_id:
            counts["missing_run_id"] += 1
            continue

        planned += 1
        done_marker = path.with_suffix(".done").exists()
        if status == "success" and (done_marker or trust_success_without_done):
            counts["success"] += 1
            migrated += 1
        else:
            counts["left_pending"] += 1

    counts["planned"] = planned
    counts["migrated"] = migrated
    return counts


def migrate(
    legacy_root: Path,
    output_root: Path,
    *,
    experiment_id: str | None = None,
    trust_success_without_done: bool = False,
    copy_sidecars: bool = True,
    dry_run: bool = False,
    strict_dry_run: bool = False,
    shard_count: int = DEFAULT_SHARD_COUNT,
) -> Counter[str]:
    if legacy_root.resolve() == output_root.resolve():
        raise ValueError("Refusing to migrate in place; choose a fresh output directory")
    if not (legacy_root / "runs").exists():
        raise ValueError(f"Legacy store has no runs directory: {legacy_root / 'runs'}")
    if output_root.exists() and any(output_root.iterdir()) and not dry_run:
        raise ValueError(f"Output directory is not empty: {output_root}")

    if dry_run and not strict_dry_run:
        return _fast_dry_run(
            (legacy_root / "runs").glob("*.json"),
            trust_success_without_done=trust_success_without_done,
        )

    record_paths = sorted((legacy_root / "runs").glob("*.json"))
    records: list[tuple[Path, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    for path in record_paths:
        try:
            records.append((path, _read_json(path)))
        except Exception:
            counts["malformed_json"] += 1

    inferred_experiment_id = experiment_id or _infer_experiment_id(
        legacy_root,
        [record for _, record in records],
    )
    planned_run_ids: list[str] = []
    migrated_run_ids: list[str] = []
    migrated_records: list[dict[str, Any]] = []
    output_rows: dict[str, list[dict[str, Any]]] = {
        "results": [],
        "artifacts": [],
        "logs": [],
    }
    output_sequence = 0
    event_rows: list[dict[str, Any]] = []

    for path, data in records:
        run_id = data.get("run_id") or path.stem
        if not isinstance(run_id, str) or not run_id:
            counts["missing_run_id"] += 1
            continue
        planned_run_ids.append(run_id)

        status = data.get("status")
        done_marker = path.with_suffix(".done").exists()
        is_safe_success = status == "success" and (done_marker or trust_success_without_done)
        if not is_safe_success:
            counts["left_pending"] += 1
            continue

        record = _run_record(
            data,
            run_id=run_id,
            experiment_id=inferred_experiment_id,
            status="success",
            legacy_root=legacy_root,
            output_root=output_root,
        )
        counts["success"] += 1
        migrated_run_ids.append(run_id)
        migrated_records.append(record)
        result_rows = _legacy_result_rows(legacy_root, run_id, output_sequence)
        output_sequence += len(result_rows)
        artifact_rows = _artifact_rows(record, output_sequence)
        output_sequence += len(artifact_rows)
        log_rows = _legacy_log_rows(legacy_root, run_id, output_sequence)
        output_sequence += len(log_rows)
        output_rows["results"].extend(result_rows)
        output_rows["artifacts"].extend(artifact_rows)
        output_rows["logs"].extend(log_rows)
        event_rows.append(
            {
                "event_id": f"migration-{run_id}",
                "kind": "finished",
                "run_id": run_id,
                "experiment_id": record["experiment_id"],
                "job_id": "legacy-migration",
                "worker_id": "legacy",
                "timestamp": record["finished_at"],
                "payload": {
                    "duration_ms": record["duration_ms"],
                    "metrics": record["metrics"],
                    "params": record["params_resolved"],
                },
            }
        )

        if not dry_run:
            if copy_sidecars:
                _copy_sidecars(legacy_root, output_root, run_id)

    if not dry_run:
        _write_record_shards(output_root, migrated_records, shard_count=shard_count)
        for kind, rows in output_rows.items():
            _write_output_shards(output_root, kind, rows, shard_count=shard_count)
        for path in [
            output_root / "outputs" / "results",
            output_root / "outputs" / "logs",
            output_root / "outputs" / "artifacts" / "metadata",
            output_root / "outputs" / "artifacts" / "files",
        ]:
            path.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_root / "outputs" / "manifest.json",
            {
                "layout_version": LAYOUT_VERSION,
                "outputs_layout": OUTPUTS_LAYOUT,
                "shard_hash": "sha256",
                "shard_count": shard_count,
                "kinds": ["results", "artifacts", "logs"],
            },
        )
        _write_json(
            output_root / ".metalab" / "meta.json",
            {
                "created_by": "metalab",
                "layout_version": LAYOUT_VERSION,
                "outputs_layout": OUTPUTS_LAYOUT,
                "shard_hash": "sha256",
                "shard_count": shard_count,
                "record_schema_version": "2",
                "schema_version": "2",
                "migration_source": str(legacy_root),
            },
        )
        _write_json(
            output_root / "manifest.json",
            {
                "layout_version": LAYOUT_VERSION,
                "outputs_layout": OUTPUTS_LAYOUT,
                "shard_hash": "sha256",
                "shard_count": shard_count,
                "record_schema_version": "2",
                "experiment_id": inferred_experiment_id,
                "expected_run_count": len(planned_run_ids),
                "expected_run_ids_inline": False,
                "expected_run_ids_path": ".metalab/index/planned-runs/{prefix}.ndjson",
                "executor_type": "legacy-migration",
                "job_id": "legacy-migration",
                "created_at": datetime.now().isoformat(),
                "migration_source": str(legacy_root),
                "migrated_success_count": len(migrated_run_ids),
            },
        )

        shards: dict[str, list[dict[str, str]]] = {}
        for run_id in planned_run_ids:
            shards.setdefault(run_id[:2], []).append(
                {"run_id": run_id, "shard_id": _shard_id(run_id, shard_count)}
            )
        for prefix, rows in shards.items():
            _write_ndjson(
                output_root / ".metalab" / "index" / "planned-runs" / f"{prefix}.ndjson",
                rows,
            )
        _write_ndjson(
            output_root / ".metalab" / "events" / "legacy-migration" / "legacy.ndjson",
            event_rows,
        )

    counts["planned"] = len(planned_run_ids)
    counts["migrated"] = len(migrated_run_ids)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy flat FileStore successes to a v4 hash-sharded HPC store."
    )
    parser.add_argument("legacy_store", type=Path)
    parser.add_argument("output_store", type=Path)
    parser.add_argument("--experiment-id", help="Override experiment id for manifest/records missing it.")
    parser.add_argument(
        "--trust-success-without-done",
        action="store_true",
        help="Treat legacy status=success records without .done markers as complete.",
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not copy artifact payloads into the v4 layout.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    parser.add_argument(
        "--strict-dry-run",
        action="store_true",
        help="With --dry-run, fully parse and normalize records instead of using the fast count pass.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=DEFAULT_SHARD_COUNT,
        help="Number of hash-mod metadata shards to create.",
    )
    args = parser.parse_args()

    counts = migrate(
        args.legacy_store,
        args.output_store,
        experiment_id=args.experiment_id,
        trust_success_without_done=args.trust_success_without_done,
        copy_sidecars=not args.no_sidecars,
        dry_run=args.dry_run,
        strict_dry_run=args.strict_dry_run,
        shard_count=args.shard_count,
    )
    print(
        "legacy migration: "
        f"planned={counts['planned']} migrated_success={counts['migrated']} "
        f"left_pending={counts['left_pending']} malformed={counts['malformed_json']}"
    )
    if args.dry_run:
        print("dry run: no files written")
    else:
        print(f"wrote v4 hash-sharded store: {args.output_store.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
