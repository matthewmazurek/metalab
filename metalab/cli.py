"""Command-line interface for the filesystem-native metalab runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


def _load_experiment(target: str) -> Any:
    path = Path(target)
    if path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import experiment script: {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent.resolve()))
        spec.loader.exec_module(module)
    else:
        module_name, _, attr = target.partition(":")
        module = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            obj = getattr(module, attr)
            return obj() if callable(obj) and not hasattr(obj, "experiment_id") else obj

    for name in ("experiment", "exp"):
        if hasattr(module, name):
            return getattr(module, name)
    if hasattr(module, "get_experiment"):
        return module.get_experiment()
    raise ValueError(
        "Experiment target must expose `experiment`, `exp`, or `get_experiment()`."
    )


def _print_status(status: Any, *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        return
    print(
        f"total={status.total} success={status.success} failed={status.failed} "
        f"skipped={status.skipped} running={status.running} pending={status.pending} "
        f"stale_workers={status.stale_workers}"
    )


def _require_store_path(store: str) -> str:
    if not store.strip():
        raise ValueError(
            "Store path is empty. If you used $STORE, define it in this terminal "
            "or pass the path explicitly."
        )
    return store


def _handle_run(args: argparse.Namespace) -> int:
    import metalab
    from metalab.executor.thread import ThreadExecutor

    exp = _load_experiment(args.target)
    store_path = _require_store_path(args.store)
    executor_factories = {
        "local": lambda: ThreadExecutor(max_workers=args.workers),
        "slurm": lambda: metalab.SlurmExecutor(metalab.SlurmConfig()),
    }
    executor = executor_factories[args.executor]()
    handle = metalab.run(
        exp,
        store=store_path,
        executor=executor,
        resume=True,
    )

    if handle.can_reconnect:
        print(f"submitted {args.executor} job_id={handle.job_id}")
        print(f"store={store_path}")
        print(f"observe: metalab observe {store_path}")
        return 0

    results = handle.result()
    print(f"completed runs={len(results)}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    from metalab.status import read_status

    _print_status(read_status(_require_store_path(args.store)), json_output=args.json_output)
    return 0


def _handle_observe(args: argparse.Namespace) -> int:
    from metalab.observe import (
        FIELD_PRESETS,
        append_fields,
        field_label,
        format_row,
        merge_run_row,
        observe_events,
        parse_fields,
        project_event,
        read_new_events,
        shorten_value,
    )
    from metalab.status import read_status

    pretty = args.pretty or not (args.plain or args.json_output)
    by_run = args.by_run or not args.events
    if args.events and args.by_run:
        raise ValueError("--events and --by-run cannot be used together")
    if pretty and args.json_output:
        raise ValueError("--pretty and --json cannot be used together")

    if args.view not in FIELD_PRESETS:
        raise ValueError(f"Unknown observe view {args.view!r}. Choose from: {', '.join(FIELD_PRESETS)}")
    fields = parse_fields(args.view)
    extras: list[str] = []
    for value in args.fields or []:
        extras.extend(parse_fields(value))
    for group in args.field or []:
        for value in group:
            extras.extend(parse_fields(value))
    fields = append_fields(fields, extras)

    if pretty:
        try:
            from rich.console import Console, Group
            from rich.live import Live
            from rich.table import Table
            from rich.text import Text
        except ImportError as e:
            raise RuntimeError(
                "Pretty observe output requires Rich. Install with `uv sync --extra rich` "
                "or omit --pretty."
            ) from e

        console = Console()
        rows: list[dict[str, Any]] = []
        by_run_rows: dict[str, dict[str, Any]] = {}
        offsets: dict[str, int] = {}

        def render_header() -> Text:
            status = read_status(_require_store_path(args.store))
            complete = status.success + status.failed + status.skipped
            pct = (complete / status.total * 100) if status.total else 0.0
            summary = Text()
            summary.append("metalab observe", style="bold")
            summary.append(f"  {_require_store_path(args.store)}", style="dim")
            summary.append("\n")
            summary.append(f"{complete}/{status.total}", style="bold white")
            summary.append(f" {pct:5.1f}%", style="white")
            summary.append("   ")
            summary.append(
                f"running {status.running}",
                style="cyan" if status.running else "dim",
            )
            summary.append("   ")
            summary.append(
                f"ok {status.success}",
                style="green" if status.success else "dim",
            )
            summary.append("   ")
            summary.append(
                f"failed {status.failed}",
                style="red" if status.failed else "dim",
            )
            summary.append("   ")
            summary.append(
                f"pending {status.pending}",
                style="white" if status.pending else "dim",
            )
            if status.skipped:
                summary.append("   ")
                summary.append(f"skipped {status.skipped}", style="dim yellow")
            if status.stale_workers:
                summary.append("   ")
                summary.append(f"stale workers {status.stale_workers}", style="yellow")
            return summary

        def render_table() -> Table:
            table = Table(
                box=None,
                caption="active and recent runs" if by_run else "recent events",
                caption_style="dim",
                expand=False,
                header_style="bold dim",
                pad_edge=False,
                show_edge=False,
            )
            for field in fields:
                justify = "right" if field == "duration_ms" or field.startswith("metrics.") else "left"
                table.add_column(field_label(field), overflow="ellipsis", justify=justify)
            visible_rows = (
                list(by_run_rows.values())[-args.limit :]
                if by_run
                else rows[-args.limit :]
            )
            for row in visible_rows:
                style = {
                    "started": "cyan",
                    "finished": "green",
                    "failed": "red",
                    "skipped": "yellow",
                }.get(str(row.get("kind")), "")
                table.add_row(
                    *[shorten_value(field, row.get(field)) for field in fields],
                    style=style,
                )
            return table

        def render_view() -> Any:
            if not by_run:
                return render_table()

            return Group(render_header(), Text(""), render_table())

        try:
            with Live(render_view(), console=console, refresh_per_second=4) as live:
                while True:
                    changed = False
                    for event in read_new_events(_require_store_path(args.store), offsets):
                        if by_run:
                            run_id = event.get("run_id")
                            if run_id:
                                by_run_rows[run_id] = merge_run_row(
                                    by_run_rows.get(run_id),
                                    event,
                                )
                                changed = True
                        else:
                            rows.append(project_event(event, fields))
                            changed = True
                    if changed or by_run:
                        live.update(render_view())
                    if args.once:
                        return 0
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
        return 0

    try:
        for row in observe_events(
            _require_store_path(args.store),
            fields=fields,
            interval=args.interval,
            once=args.once,
        ):
            if args.json_output:
                print(json.dumps(row, sort_keys=True), flush=True)
            else:
                print(format_row(row), flush=True)
    except KeyboardInterrupt:
        return 0
    return 0


def _handle_index(args: argparse.Namespace) -> int:
    from metalab.index import rebuild_index

    path = rebuild_index(_require_store_path(args.store), force=args.force)
    print(f"index rebuilt: {path}")
    return 0


def _handle_summary(args: argparse.Namespace) -> int:
    from metalab.index import summary

    rows = summary(
        _require_store_path(args.store),
        group_by=args.group_by,
        metric=args.metric,
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _handle_export(args: argparse.Namespace) -> int:
    from metalab.index import export

    out = export(_require_store_path(args.store), fmt=args.format, out=args.out)
    print(f"exported: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="metalab",
        description="metalab: filesystem-native HPC experiment runner",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment script/module")
    run_p.add_argument("target", help="Python script or module[:attr] exposing an Experiment")
    run_p.add_argument("--store", required=True, help="Filesystem run-store path")
    run_p.add_argument("--executor", choices=["local", "slurm"], default="local")
    run_p.add_argument("--workers", type=int, default=1, help="Local worker threads")
    run_p.set_defaults(func=_handle_run)

    status_p = sub.add_parser("status", help="Show run-store status")
    status_p.add_argument("store")
    status_p.add_argument("--json", action="store_true", dest="json_output")
    status_p.set_defaults(func=_handle_status)

    observe_p = sub.add_parser("observe", help="Print selected live event fields")
    observe_p.add_argument("store")
    observe_p.add_argument(
        "--fields",
        action="append",
        help=(
            "Additional comma-separated fields or aliases, e.g. "
            "params.x,metrics.loss"
        ),
    )
    observe_p.add_argument(
        "-f",
        "--field",
        action="append",
        nargs="+",
        help="Additional fields, e.g. -f params.x metrics.loss",
    )
    observe_p.add_argument(
        "--view",
        default="default",
        help="Field preset: default, basic, timing, smoke, errors",
    )
    observe_p.add_argument("--interval", type=float, default=2.0)
    observe_p.add_argument("--once", action="store_true", help="Read current events and exit")
    observe_p.add_argument("--json", action="store_true", dest="json_output")
    observe_p.add_argument(
        "--pretty",
        action="store_true",
        help="Render a live Rich table (default unless --plain or --json is used)",
    )
    observe_p.add_argument("--plain", action="store_true", help="Print key=value rows")
    observe_p.add_argument(
        "--events",
        action="store_true",
        help="Show an append-only event stream instead of the by-run dashboard",
    )
    observe_p.add_argument(
        "--by-run",
        action="store_true",
        help="Update one row per run (default unless --events is used)",
    )
    observe_p.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum rows to keep in pretty mode",
    )
    observe_p.set_defaults(func=_handle_observe)

    index_p = sub.add_parser("index", help="Manage sidecar indexes")
    index_sub = index_p.add_subparsers(dest="index_command", required=True)
    rebuild_p = index_sub.add_parser("rebuild", help="Rebuild DuckDB sidecar index")
    rebuild_p.add_argument("store")
    rebuild_p.add_argument("--force", action="store_true")
    rebuild_p.set_defaults(func=_handle_index)

    summary_p = sub.add_parser("summary", help="Summarize indexed runs")
    summary_p.add_argument("store")
    summary_p.add_argument("--group-by")
    summary_p.add_argument("--metric")
    summary_p.set_defaults(func=_handle_summary)

    export_p = sub.add_parser("export", help="Export indexed runs")
    export_p.add_argument("store")
    export_p.add_argument("--format", choices=["csv", "parquet", "jsonl"], required=True)
    export_p.add_argument("--out", required=True)
    export_p.set_defaults(func=_handle_export)

    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
