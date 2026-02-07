"""
MetaLab CLI: Command-line interface for metalab utilities.

Provides commands for:
- postgres: Manage PostgreSQL services
- store: Transfer data between stores, list experiments
- env: Manage environment profiles
- atlas: Provision and manage services
- tunnel: Open SSH tunnels to services
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="metalab",
        description="MetaLab: A general experiment runner",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Postgres commands
    postgres_parser = subparsers.add_parser(
        "postgres",
        help="Manage PostgreSQL service",
    )
    postgres_subparsers = postgres_parser.add_subparsers(
        dest="postgres_command",
        help="Postgres commands",
    )

    # postgres start
    start_parser = postgres_subparsers.add_parser(
        "start",
        help="Start PostgreSQL service",
    )
    start_parser.add_argument(
        "--store",
        "-s",
        help="Store root directory (for SLURM service discovery). Defaults to --file-root.",
    )
    start_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)",
    )
    start_parser.add_argument(
        "--database",
        "-d",
        default="metalab",
        help="Database name (default: metalab)",
    )
    start_parser.add_argument(
        "--user",
        "-u",
        help="Database user (default: current user)",
    )
    start_parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit as SLURM job",
    )
    start_parser.add_argument(
        "--slurm-partition",
        default="default",
        help="SLURM partition (default: default)",
    )
    start_parser.add_argument(
        "--slurm-time",
        default="24:00:00",
        help="SLURM walltime (default: 24:00:00)",
    )
    start_parser.add_argument(
        "--slurm-memory",
        default="4G",
        help="SLURM memory (default: 4G)",
    )
    start_parser.add_argument(
        "--data-dir",
        help="PostgreSQL data directory (PGDATA)",
    )
    start_parser.add_argument(
        "--auth-method",
        choices=["trust", "scram-sha-256"],
        default="trust",
        help="Authentication method (default: trust)",
    )
    start_parser.add_argument(
        "--password",
        help="PostgreSQL password (used with scram-sha-256)",
    )
    start_parser.add_argument(
        "--file-root",
        help="File root for artifacts/logs (also used for SLURM service discovery if --store is omitted)",
    )
    start_parser.add_argument(
        "--schema",
        help="PostgreSQL schema for PostgresStore (default: public)",
    )
    start_parser.add_argument(
        "--print-store-locator",
        action="store_true",
        help="Print PostgresStore locator for this service",
    )
    start_parser.add_argument(
        "--service-id",
        default="default",
        help="Service identifier for local mode (default: default)",
    )
    start_parser.add_argument(
        "--listen-addresses",
        default="localhost",
        help="Addresses to listen on (default: localhost, use '*' for all)",
    )
    start_parser.add_argument(
        "--max-connections",
        type=int,
        default=100,
        help="Maximum concurrent connections (default: 100)",
    )

    # postgres status
    status_parser = postgres_subparsers.add_parser(
        "status",
        help="Check PostgreSQL service status",
    )
    status_parser.add_argument(
        "--store",
        "-s",
        help="Store root directory (defaults to --file-root)",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    status_parser.add_argument(
        "--file-root",
        help="File root for artifacts/logs (also used for service discovery if --store is omitted)",
    )
    status_parser.add_argument(
        "--schema",
        help="PostgreSQL schema for PostgresStore (default: public)",
    )
    status_parser.add_argument(
        "--store-locator",
        action="store_true",
        help="Include PostgresStore locator in output",
    )
    status_parser.add_argument(
        "--service-id",
        default="default",
        help="Service identifier for local mode (default: default)",
    )

    # postgres stop
    stop_parser = postgres_subparsers.add_parser(
        "stop",
        help="Stop PostgreSQL service",
    )
    stop_parser.add_argument(
        "--store",
        "-s",
        help="Store root directory (defaults to --file-root)",
    )
    stop_parser.add_argument(
        "--file-root",
        help="File root for artifacts/logs (also used for service discovery if --store is omitted)",
    )
    stop_parser.add_argument(
        "--service-id",
        default="default",
        help="Service identifier for local mode (default: default)",
    )

    # Store commands
    store_parser = subparsers.add_parser(
        "store",
        help="Store operations (transfer, list)",
    )
    store_subparsers = store_parser.add_subparsers(
        dest="store_command",
        help="Store commands",
    )

    # store export
    export_parser = store_subparsers.add_parser(
        "export",
        help="Export data from source store to destination",
    )
    export_parser.add_argument(
        "--from",
        "-f",
        required=True,
        dest="source",
        help="Source store locator (e.g., postgresql://localhost/db)",
    )
    export_parser.add_argument(
        "--to",
        "-t",
        required=True,
        dest="destination",
        help="Destination store locator (e.g., file:///path/to/store)",
    )
    export_parser.add_argument(
        "--experiment",
        "-e",
        help="Filter by experiment ID",
    )
    export_parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact files (usually skipped for file-backed stores)",
    )
    export_parser.add_argument(
        "--include-derived",
        action="store_true",
        default=True,
        help="Include derived metrics (default: True)",
    )
    export_parser.add_argument(
        "--no-include-derived",
        action="store_false",
        dest="include_derived",
        help="Exclude derived metrics",
    )
    export_parser.add_argument(
        "--include-logs",
        action="store_true",
        default=True,
        help="Include log files (default: True)",
    )
    export_parser.add_argument(
        "--no-include-logs",
        action="store_false",
        dest="include_logs",
        help="Exclude log files",
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing records in destination",
    )

    # store import (alias for export, since export_store is bidirectional)
    import_parser = store_subparsers.add_parser(
        "import",
        help="Import data from source store to destination (alias for export)",
    )
    import_parser.add_argument(
        "--from",
        "-f",
        required=True,
        dest="source",
        help="Source store locator (e.g., file:///path/to/store)",
    )
    import_parser.add_argument(
        "--to",
        "-t",
        required=True,
        dest="destination",
        help="Destination store locator (e.g., postgresql://localhost/db)",
    )
    import_parser.add_argument(
        "--experiment",
        "-e",
        help="Filter by experiment ID",
    )
    import_parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact files (usually skipped for file-backed stores)",
    )
    import_parser.add_argument(
        "--include-derived",
        action="store_true",
        default=True,
        help="Include derived metrics (default: True)",
    )
    import_parser.add_argument(
        "--no-include-derived",
        action="store_false",
        dest="include_derived",
        help="Exclude derived metrics",
    )
    import_parser.add_argument(
        "--include-logs",
        action="store_true",
        default=True,
        help="Include log files (default: True)",
    )
    import_parser.add_argument(
        "--no-include-logs",
        action="store_false",
        dest="include_logs",
        help="Exclude log files",
    )
    import_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing records in destination",
    )

    # store list
    list_parser = store_subparsers.add_parser(
        "list",
        help="List experiments in a store",
    )
    list_parser.add_argument(
        "--store",
        "-s",
        required=True,
        help="Store locator (e.g., file:///path/to/store or postgresql://localhost/db)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )

    # Env commands
    env_parser = subparsers.add_parser(
        "env",
        help="Manage environment profiles",
    )
    env_subparsers = env_parser.add_subparsers(
        dest="env_command",
        help="Environment commands",
    )

    # env list
    env_subparsers.add_parser(
        "list",
        help="List all environment profiles",
    )

    # env show
    env_show_parser = env_subparsers.add_parser(
        "show",
        help="Show resolved config for an environment profile",
    )
    env_show_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Environment profile name (default: project default)",
    )

    # Atlas commands
    atlas_parser = subparsers.add_parser(
        "atlas",
        help="Provision and manage services",
    )
    atlas_subparsers = atlas_parser.add_subparsers(
        dest="atlas_command",
        help="Atlas commands",
    )

    # atlas up
    atlas_up_parser = atlas_subparsers.add_parser(
        "up",
        help="Provision services per the selected environment profile",
    )
    atlas_up_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    atlas_up_parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Also open a tunnel after provisioning",
    )

    # atlas down
    atlas_down_parser = atlas_subparsers.add_parser(
        "down",
        help="Stop all services and clean up",
    )
    atlas_down_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )

    # atlas status
    atlas_status_parser = atlas_subparsers.add_parser(
        "status",
        help="Check health of running services",
    )
    atlas_status_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    atlas_status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )

    # atlas logs
    atlas_logs_parser = atlas_subparsers.add_parser(
        "logs",
        help="Show service logs",
    )
    atlas_logs_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    atlas_logs_parser.add_argument(
        "service",
        nargs="?",
        default=None,
        help="Service name (e.g. atlas, postgres). Omit for all services.",
    )
    atlas_logs_parser.add_argument(
        "-n", "--tail",
        type=int,
        default=0,
        help="Show only the last N lines (0 = all)",
    )

    # Tunnel command
    tunnel_parser = subparsers.add_parser(
        "tunnel",
        help="Open a managed SSH tunnel to running services",
    )
    tunnel_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    tunnel_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Local port for the tunnel",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "postgres":
        return handle_postgres(args)
    elif args.command == "store":
        return handle_store(args)
    elif args.command == "env":
        return handle_env(args)
    elif args.command == "atlas":
        return handle_atlas(args)
    elif args.command == "tunnel":
        return handle_tunnel(args)
    else:
        parser.print_help()
        return 0


def handle_postgres(args: argparse.Namespace) -> int:
    """Handle postgres subcommands."""
    from metalab.services.postgres import (
        PostgresServiceConfig,
        build_store_locator,
        get_service_info,
        start_postgres_local,
        start_postgres_slurm,
        stop_postgres,
    )

    def _resolve_file_root() -> Path | None:
        """Resolve file_root from args, preferring explicit file_root over store."""
        if getattr(args, "file_root", None):
            return Path(args.file_root)
        if getattr(args, "store", None):
            return Path(args.store)
        return None

    def _resolve_store_root() -> Path | None:
        """Resolve store_root from args, preferring explicit store over file_root."""
        if getattr(args, "store", None):
            return Path(args.store)
        if getattr(args, "file_root", None):
            return Path(args.file_root)
        return None

    if args.postgres_command == "start":
        # Build config with all available options
        config_kwargs: dict = {
            "port": args.port,
            "database": args.database,
            "auth_method": args.auth_method,
            "listen_addresses": args.listen_addresses,
            "max_connections": args.max_connections,
        }
        if args.data_dir:
            config_kwargs["data_dir"] = Path(args.data_dir)
        if args.password:
            config_kwargs["password"] = args.password
        if args.user:
            config_kwargs["user"] = args.user

        config = PostgresServiceConfig(**config_kwargs)

        try:
            if args.slurm:
                store_root = _resolve_store_root()
                if store_root is None:
                    print(
                        "Error: --file-root (or --store) required for SLURM mode",
                        file=sys.stderr,
                    )
                    return 1
                service = start_postgres_slurm(
                    config,
                    store_root=store_root,
                    slurm_partition=args.slurm_partition,
                    slurm_time=args.slurm_time,
                    slurm_memory=args.slurm_memory,
                )
            else:
                service = start_postgres_local(
                    config,
                    service_id=args.service_id,
                )

            print("PostgreSQL started:")
            print(f"  Connection: {service.connection_string}")
            print(f"  Host: {service.host}")
            print(f"  Port: {service.port}")
            print(f"  Database: {service.database}")
            if service.slurm_job_id:
                print(f"  SLURM Job: {service.slurm_job_id}")
            if args.print_store_locator:
                file_root = _resolve_file_root()
                if file_root is None:
                    print(
                        "Error: --file-root (or --store in SLURM) required for store locator",
                        file=sys.stderr,
                    )
                    return 1
                locator = build_store_locator(
                    service,
                    file_root=file_root,
                    schema=args.schema,
                )
                print(f"  Store locator: {locator}")
            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.postgres_command == "status":
        store_root = _resolve_store_root()
        service = get_service_info(
            store_root=store_root,
            service_id=args.service_id,
        )

        if service is None:
            if args.json_output:
                print(json.dumps({"running": False}))
            else:
                print("PostgreSQL service is not running")
            return 1

        if args.json_output:
            data = service.to_dict()
            data["running"] = True
            if args.store_locator:
                file_root = _resolve_file_root()
                if file_root is None:
                    print(
                        "Error: --file-root (or --store) required for store locator",
                        file=sys.stderr,
                    )
                    return 1
                data["store_locator"] = build_store_locator(
                    service,
                    file_root=file_root,
                    schema=args.schema,
                )
            print(json.dumps(data, indent=2))
        else:
            print("PostgreSQL service is running:")
            print(f"  Connection: {service.connection_string}")
            print(f"  Host: {service.host}")
            print(f"  Port: {service.port}")
            print(f"  Database: {service.database}")
            if service.slurm_job_id:
                print(f"  SLURM Job: {service.slurm_job_id}")
            if service.started_at:
                print(f"  Started: {service.started_at}")
            if args.store_locator:
                file_root = _resolve_file_root()
                if file_root is None:
                    print(
                        "Error: --file-root (or --store) required for store locator",
                        file=sys.stderr,
                    )
                    return 1
                locator = build_store_locator(
                    service,
                    file_root=file_root,
                    schema=args.schema,
                )
                print(f"  Store locator: {locator}")

        return 0

    elif args.postgres_command == "stop":
        store_root = _resolve_store_root()

        if stop_postgres(
            store_root=store_root,
            service_id=args.service_id,
        ):
            print("PostgreSQL service stopped")
            return 0
        else:
            print("PostgreSQL service was not running")
            return 0

    else:
        print("Usage: metalab postgres {start|status|stop}", file=sys.stderr)
        return 1


def handle_store(args: argparse.Namespace) -> int:
    """Handle store subcommands."""
    from metalab.store.transfer import export_store

    if args.store_command == "export":
        try:

            def progress(current: int, total: int) -> None:
                print(f"\rExporting: {current}/{total}", end="", flush=True)

            counts = export_store(
                args.source,
                args.destination,
                experiment_id=args.experiment,
                include_artifacts=args.include_artifacts,
                include_derived=args.include_derived,
                include_logs=args.include_logs,
                overwrite=args.overwrite,
                progress_callback=progress,
            )

            print()  # Newline after progress
            print("Export complete:")
            for key, value in counts.items():
                print(f"  {key}: {value}")

            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.store_command == "import":
        # Import is an alias for export (export_store is bidirectional)
        try:

            def progress(current: int, total: int) -> None:
                print(f"\rImporting: {current}/{total}", end="", flush=True)

            counts = export_store(
                args.source,
                args.destination,
                experiment_id=args.experiment,
                include_artifacts=args.include_artifacts,
                include_derived=args.include_derived,
                include_logs=args.include_logs,
                overwrite=args.overwrite,
                progress_callback=progress,
            )

            print()  # Newline after progress
            print("Import complete:")
            for key, value in counts.items():
                print(f"  {key}: {value}")

            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.store_command == "list":
        try:
            from metalab.store.locator import parse_to_config

            config = parse_to_config(args.store)

            # Check if config supports listing experiments
            if hasattr(config, "list_experiments"):
                experiments = config.list_experiments()
            else:
                # Fall back to connecting and listing run records
                store = config.connect()
                records = store.list_run_records()
                experiments = sorted(set(r.experiment_id for r in records))

            if args.json_output:
                print(json.dumps({"experiments": experiments}))
            else:
                if experiments:
                    print("Experiments:")
                    for exp in experiments:
                        print(f"  {exp}")
                else:
                    print("No experiments found")

            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    else:
        print("Usage: metalab store {export|import|list}", file=sys.stderr)
        return 1


def handle_env(args: argparse.Namespace) -> int:
    """Handle env subcommands."""
    from metalab.config import ProjectConfig

    try:
        config = ProjectConfig.load()
    except FileNotFoundError:
        print("No .metalab.toml found. Create one in your project root.")
        print("See: https://metalab.readthedocs.io/en/latest/services/")
        return 1

    if args.env_command == "list":
        envs = config.list_environments()
        default = config.project.default_env
        if not envs:
            print("No environments defined in .metalab.toml")
            return 0
        h = "─"
        print(f"  {'NAME':<20} {'TYPE':<12} {'DEFAULT'}")
        print(f"  {h * 20} {h * 12} {h * 7}")
        for name in envs:
            profile = config.environments[name]
            marker = "*" if name == default else ""
            print(f"  {name:<20} {profile.type:<12} {marker}")
        return 0

    elif args.env_command == "show":
        try:
            resolved = config.resolve(args.name)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Environment: {resolved.env_name}")
        print(f"  Type: {resolved.env_type}")
        if resolved.file_root:
            print(f"  File root: {resolved.file_root}")
        if resolved.env_config:
            print(f"  Config:")
            for k, v in resolved.env_config.items():
                print(f"    {k}: {v}")
        if resolved.services:
            print(f"  Services:")
            for svc_name, svc_config in resolved.services.items():
                print(f"    {svc_name}: {svc_config}")
        return 0

    else:
        print("Usage: metalab env {list|show}", file=sys.stderr)
        return 1


def handle_atlas(args: argparse.Namespace) -> int:
    """Handle atlas subcommands."""
    import json as json_mod

    from metalab.config import ProjectConfig
    from metalab.environment.orchestrator import ServiceOrchestrator

    try:
        config = ProjectConfig.load()
    except FileNotFoundError:
        print("No .metalab.toml found. Create one in your project root.")
        print("See: https://metalab.readthedocs.io/en/latest/services/")
        return 1

    env_name = getattr(args, "env", None) or os.environ.get("METALAB_ENV")

    try:
        resolved = config.resolve(env_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    orch = ServiceOrchestrator(resolved)

    if args.atlas_command == "up":
        try:
            tunnel = getattr(args, "tunnel", False)
            bundle = orch.up(tunnel=tunnel)
            print(f"Services started ({resolved.env_name}):")
            for name, handle in bundle.services.items():
                print(f"  {name}: {handle.host}:{handle.port}")
            if bundle.store_locator:
                print(f"  Store: {bundle.store_locator}")

            atlas = bundle.get("atlas")
            if atlas:
                print(f"\n  Atlas UI: http://{atlas.host}:{atlas.port}")

            if bundle.tunnel_targets and not tunnel:
                print(f"\nTo access locally, run:")
                print(f"  metalab tunnel")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.atlas_command == "down":
        try:
            orch.down()
            print("All services stopped.")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.atlas_command == "status":
        try:
            status = orch.status()
            if not status.bundle_found:
                print("No services running.")
                return 0
            use_json = getattr(args, "json_output", False)
            if use_json:
                print(json_mod.dumps(status.services, indent=2))
            else:
                for name, info in status.services.items():
                    symbol = "\u2713" if info["available"] else "\u2717"
                    print(
                        f"  {symbol} {name}: {info['host']}:{info['port']} ({info['status']})"
                    )
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.atlas_command == "logs":
        try:
            service = getattr(args, "service", None)
            tail = getattr(args, "tail", 0)
            logs = orch.logs(service_name=service, tail=tail)
            if not logs:
                print("No service logs found.", file=sys.stderr)
                return 1
            for name, content in logs.items():
                if len(logs) > 1:
                    print(f"=== {name} ===")
                print(content)
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    else:
        print("Usage: metalab atlas {up|down|status|logs}", file=sys.stderr)
        return 1


def handle_tunnel(args: argparse.Namespace) -> int:
    """Handle tunnel command."""
    import signal

    from metalab.config import ProjectConfig
    from metalab.environment.orchestrator import ServiceOrchestrator

    try:
        config = ProjectConfig.load()
    except FileNotFoundError:
        print("No .metalab.toml found. Create one in your project root.")
        print("See: https://metalab.readthedocs.io/en/latest/services/")
        return 1

    env_name = getattr(args, "env", None) or os.environ.get("METALAB_ENV")

    try:
        resolved = config.resolve(env_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    orch = ServiceOrchestrator(resolved)

    try:
        handle = orch.tunnel()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if handle is None:
        print("No tunnel needed (services are local)")
        return 0

    print(f"Tunnel established: {handle.local_url}")
    print("Press Ctrl+C to close.")

    try:
        signal.pause()
    except KeyboardInterrupt:
        print("\nClosing tunnel...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
