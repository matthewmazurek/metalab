#!/usr/bin/env python3
"""One-off migration from the legacy flat FileStore to the v2 HPC run store.

This intentionally lives outside the metalab CLI. It is for rescuing old
experiments so the clean-break runner can resume them safely.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    new_artifact_dir = (output_root / "artifacts" / run_id[:2] / run_id).resolve()
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


def _v2_record(
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
        output_root / "artifacts" / prefix / run_id,
    )
    _copy_tree(
        legacy_root / "results" / run_id,
        output_root / "results" / prefix / run_id,
    )

    derived = legacy_root / "derived" / f"{run_id}.json"
    if derived.exists():
        dst = output_root / "derived" / prefix / f"{run_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived, dst)

    logs_dir = legacy_root / "logs"
    if logs_dir.exists():
        for log_path in logs_dir.glob(f"{run_id}_*.log"):
            dst = output_root / "logs" / prefix / log_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, dst)


def migrate(
    legacy_root: Path,
    output_root: Path,
    *,
    experiment_id: str | None = None,
    trust_success_without_done: bool = False,
    copy_sidecars: bool = True,
    dry_run: bool = False,
) -> Counter[str]:
    if legacy_root.resolve() == output_root.resolve():
        raise ValueError("Refusing to migrate in place; choose a fresh output directory")
    if not (legacy_root / "runs").exists():
        raise ValueError(f"Legacy store has no runs directory: {legacy_root / 'runs'}")
    if output_root.exists() and any(output_root.iterdir()) and not dry_run:
        raise ValueError(f"Output directory is not empty: {output_root}")

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

        record = _v2_record(
            data,
            run_id=run_id,
            experiment_id=inferred_experiment_id,
            status="success",
            legacy_root=legacy_root,
            output_root=output_root,
        )
        counts["success"] += 1
        migrated_run_ids.append(run_id)
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
            _write_json(output_root / "runs" / run_id[:2] / f"{run_id}.json", record)
            if copy_sidecars:
                _copy_sidecars(legacy_root, output_root, run_id)

    if not dry_run:
        _write_json(
            output_root / "_meta.json",
            {
                "created_by": "metalab",
                "layout_version": 2,
                "schema_version": "2",
                "migration_source": str(legacy_root),
            },
        )
        _write_json(
            output_root / "manifest.json",
            {
                "layout_version": 2,
                "experiment_id": inferred_experiment_id,
                "expected_run_count": len(planned_run_ids),
                "expected_run_ids_inline": False,
                "expected_run_ids_path": "index/planned-runs/{prefix}.ndjson",
                "executor_type": "legacy-migration",
                "job_id": "legacy-migration",
                "created_at": datetime.now().isoformat(),
                "migration_source": str(legacy_root),
                "migrated_success_count": len(migrated_run_ids),
            },
        )

        shards: dict[str, list[dict[str, str]]] = {}
        for run_id in planned_run_ids:
            shards.setdefault(run_id[:2], []).append({"run_id": run_id})
        for prefix, rows in shards.items():
            _write_ndjson(output_root / "index" / "planned-runs" / f"{prefix}.ndjson", rows)
        _write_ndjson(output_root / "events" / "legacy-migration" / "legacy.ndjson", event_rows)

    counts["planned"] = len(planned_run_ids)
    counts["migrated"] = len(migrated_run_ids)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy flat FileStore successes to a v2 sharded HPC store."
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
        help="Do not copy artifacts/results/derived/logs into the v2 layout.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing files.")
    args = parser.parse_args()

    counts = migrate(
        args.legacy_store,
        args.output_store,
        experiment_id=args.experiment_id,
        trust_success_without_done=args.trust_success_without_done,
        copy_sidecars=not args.no_sidecars,
        dry_run=args.dry_run,
    )
    print(
        "legacy migration: "
        f"planned={counts['planned']} migrated_success={counts['migrated']} "
        f"left_pending={counts['left_pending']} malformed={counts['malformed_json']}"
    )
    if args.dry_run:
        print("dry run: no files written")
    else:
        print(f"wrote v2 store: {args.output_store.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
