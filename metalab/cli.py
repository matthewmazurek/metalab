"""
MetaLab CLI: Command-line interface for metalab utilities.

Provides commands for:
- store: Transfer data between stores, list experiments
- env: Manage environment profiles
- services: Provision and manage services
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

    # Services commands
    services_parser = subparsers.add_parser(
        "services",
        help="Provision and manage services",
    )
    services_subparsers = services_parser.add_subparsers(
        dest="services_command",
        help="Service commands",
    )

    # services up
    services_up_parser = services_subparsers.add_parser(
        "up",
        help="Provision services per the selected environment profile",
    )
    services_up_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    services_up_parser.add_argument(
        "--tunnel",
        action="store_true",
        help="Also open a tunnel after provisioning",
    )

    # services down
    services_down_parser = services_subparsers.add_parser(
        "down",
        help="Stop all services and clean up",
    )
    services_down_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )

    # services status
    services_status_parser = services_subparsers.add_parser(
        "status",
        help="Check health of running services",
    )
    services_status_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    services_status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )

    # services logs
    services_logs_parser = services_subparsers.add_parser(
        "logs",
        help="Show service logs",
    )
    services_logs_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
    )
    services_logs_parser.add_argument(
        "service",
        nargs="?",
        default=None,
        help="Service name (e.g. atlas, postgres). Omit for all services.",
    )
    services_logs_parser.add_argument(
        "-n", "--tail",
        type=int,
        default=0,
        help="Show only the last N lines (0 = all)",
    )

    # services rebuild-index
    services_rebuild_parser = services_subparsers.add_parser(
        "rebuild-index",
        help="Rebuild the Postgres query index from the FileStore",
    )
    services_rebuild_parser.add_argument(
        "--env",
        default=None,
        help="Environment profile name (default: project default, or METALAB_ENV)",
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

    if args.command == "store":
        return handle_store(args)
    elif args.command == "env":
        return handle_env(args)
    elif args.command == "services":
        return handle_services(args)
    elif args.command == "tunnel":
        return handle_tunnel(args)
    else:
        parser.print_help()
        return 0


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


def _print_tunnel_hint(
    bundle: "ServiceBundle",
    resolved: "ResolvedConfig",
) -> None:
    """Print a copy-paste SSH tunnel command for remote services."""
    import shlex

    from metalab.environment.connector import ConnectionTarget
    from metalab.environment.ssh_tunnel import build_ssh_command

    target_info = bundle.tunnel_targets[0]
    target = ConnectionTarget(
        remote_host=target_info["host"],
        remote_port=target_info["remote_port"],
        local_port=target_info.get("local_port", target_info["remote_port"]),
        gateway=resolved.env_config.get("gateway"),
        user=resolved.env_config.get("user"),
        ssh_key=resolved.env_config.get("ssh_key"),
    )

    cmd = build_ssh_command(target)
    print(f"\nTunnel from your workstation:")
    print(f"  {shlex.join(cmd)}")
    print(f"\nThen open: http://localhost:{target.local_port}")


def handle_services(args: argparse.Namespace) -> int:
    """Handle services subcommands."""
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

    if args.services_command == "up":
        try:
            tunnel = getattr(args, "tunnel", False)
            bundle = orch.up(tunnel=tunnel)
            print(f"Services started ({resolved.env_name}):")
            for name, handle in bundle.services.items():
                print(f"  {name}: {handle.host}:{handle.port}")
            if bundle.store_locator:
                print(f"  Store: {bundle.store_locator}")

            if bundle.tunnel_targets and not tunnel:
                _print_tunnel_hint(bundle, resolved)
            else:
                atlas = bundle.get("atlas")
                if atlas:
                    print(f"\n  Atlas UI: http://{atlas.host}:{atlas.port}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.services_command == "down":
        try:
            orch.down()
            print("All services stopped.")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    elif args.services_command == "status":
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

    elif args.services_command == "logs":
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

    elif args.services_command == "rebuild-index":
        try:
            from metalab.environment import ServiceBundle
            from metalab.store.locator import create_store
            from metalab.store.postgres import PostgresStore

            if not orch._bundle_path.exists():
                print(
                    "No running services found. Start them with 'metalab services up' first.",
                    file=sys.stderr,
                )
                return 1

            bundle = ServiceBundle.load(orch._bundle_path)
            locator = bundle.store_locator
            if not locator:
                print(
                    "No store locator in service bundle. "
                    "Is Postgres configured in your environment?",
                    file=sys.stderr,
                )
                return 1

            store = create_store(locator)
            if not isinstance(store, PostgresStore):
                print(
                    f"rebuild-index requires a PostgresStore, but got {type(store).__name__}.",
                    file=sys.stderr,
                )
                return 1

            print("Rebuilding Postgres index from FileStore...")
            count = store.rebuild_index()
            print(f"Done. Indexed {count} records.")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    else:
        print("Usage: metalab services {up|down|status|logs|rebuild-index}", file=sys.stderr)
        return 1


def handle_tunnel(args: argparse.Namespace) -> int:
    """Handle tunnel command — prints the SSH tunnel command for remote services."""
    from metalab.config import ProjectConfig
    from metalab.environment.bundle import ServiceBundle
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

    if not orch._bundle_path.exists():
        print("No service bundle found. Run 'metalab services up' first.")
        return 1

    try:
        bundle = ServiceBundle.load(orch._bundle_path)
    except Exception as e:
        print(f"Error loading bundle: {e}", file=sys.stderr)
        return 1

    if not bundle.tunnel_targets:
        print("No tunnel needed (services are local).")
        return 0

    # Allow --port to override the local port
    local_port_override = getattr(args, "port", None)
    if local_port_override is not None:
        bundle.tunnel_targets[0]["local_port"] = local_port_override

    _print_tunnel_hint(bundle, resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
