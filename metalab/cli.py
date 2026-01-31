"""
MetaLab CLI: Command-line interface for metalab utilities.

Provides commands for:
- postgres: Manage PostgreSQL services
- store: Transfer data between stores
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="metalab",
        description="MetaLab: A general experiment runner",
    )
    parser.add_argument(
        "-v", "--verbose",
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
        "--store", "-s",
        help="Store root directory (for SLURM service discovery)",
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=5432,
        help="PostgreSQL port (default: 5432)",
    )
    start_parser.add_argument(
        "--database", "-d",
        default="metalab",
        help="Database name (default: metalab)",
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
    
    # postgres status
    status_parser = postgres_subparsers.add_parser(
        "status",
        help="Check PostgreSQL service status",
    )
    status_parser.add_argument(
        "--store", "-s",
        help="Store root directory",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    
    # postgres stop
    stop_parser = postgres_subparsers.add_parser(
        "stop",
        help="Stop PostgreSQL service",
    )
    stop_parser.add_argument(
        "--store", "-s",
        help="Store root directory",
    )
    
    # Store commands
    store_parser = subparsers.add_parser(
        "store",
        help="Store transfer operations",
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
        "--from", "-f",
        required=True,
        dest="source",
        help="Source store locator (e.g., postgresql://localhost/db)",
    )
    export_parser.add_argument(
        "--to", "-t",
        required=True,
        dest="destination",
        help="Destination store locator (e.g., file:///path/to/store)",
    )
    export_parser.add_argument(
        "--experiment", "-e",
        help="Filter by experiment ID",
    )
    export_parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include artifact files (usually skipped)",
    )
    
    # store import
    import_parser = store_subparsers.add_parser(
        "import",
        help="Import data from source store to destination",
    )
    import_parser.add_argument(
        "--from", "-f",
        required=True,
        dest="source",
        help="Source store locator (e.g., file:///path/to/store)",
    )
    import_parser.add_argument(
        "--to", "-t",
        required=True,
        dest="destination",
        help="Destination store locator (e.g., postgresql://localhost/db)",
    )
    import_parser.add_argument(
        "--experiment", "-e",
        help="Filter by experiment ID",
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
    else:
        parser.print_help()
        return 0


def handle_postgres(args: argparse.Namespace) -> int:
    """Handle postgres subcommands."""
    from metalab.services.postgres import (
        PostgresServiceConfig,
        get_service_info,
        start_postgres_local,
        start_postgres_slurm,
        stop_postgres,
    )
    
    if args.postgres_command == "start":
        config = PostgresServiceConfig(
            port=args.port,
            database=args.database,
            data_dir=Path(args.data_dir) if args.data_dir else None,
        )
        
        try:
            if args.slurm:
                if not args.store:
                    print("Error: --store required for SLURM mode", file=sys.stderr)
                    return 1
                
                service = start_postgres_slurm(
                    config,
                    store_root=Path(args.store),
                    slurm_partition=args.slurm_partition,
                    slurm_time=args.slurm_time,
                    slurm_memory=args.slurm_memory,
                )
            else:
                service = start_postgres_local(config)
            
            print(f"PostgreSQL started:")
            print(f"  Connection: {service.connection_string}")
            print(f"  Host: {service.host}")
            print(f"  Port: {service.port}")
            print(f"  Database: {service.database}")
            if service.slurm_job_id:
                print(f"  SLURM Job: {service.slurm_job_id}")
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.postgres_command == "status":
        store_root = Path(args.store) if args.store else None
        service = get_service_info(store_root=store_root)
        
        if service is None:
            if args.json_output:
                print(json.dumps({"running": False}))
            else:
                print("PostgreSQL service is not running")
            return 1
        
        if args.json_output:
            data = service.to_dict()
            data["running"] = True
            print(json.dumps(data, indent=2))
        else:
            print(f"PostgreSQL service is running:")
            print(f"  Connection: {service.connection_string}")
            print(f"  Host: {service.host}")
            print(f"  Port: {service.port}")
            print(f"  Database: {service.database}")
            if service.slurm_job_id:
                print(f"  SLURM Job: {service.slurm_job_id}")
            if service.started_at:
                print(f"  Started: {service.started_at}")
        
        return 0
    
    elif args.postgres_command == "stop":
        store_root = Path(args.store) if args.store else None
        
        if stop_postgres(store_root=store_root):
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
    from metalab.store.transfer import (
        export_store,
        export_to_filestore,
        import_from_filestore,
    )
    
    if args.store_command == "export":
        try:
            def progress(current: int, total: int) -> None:
                print(f"\rExporting: {current}/{total}", end="", flush=True)
            
            counts = export_store(
                args.source,
                args.destination,
                experiment_id=args.experiment,
                include_artifacts=args.include_artifacts,
                progress_callback=progress,
            )
            
            print()  # Newline after progress
            print(f"Export complete:")
            for key, value in counts.items():
                print(f"  {key}: {value}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    elif args.store_command == "import":
        try:
            def progress(current: int, total: int) -> None:
                print(f"\rImporting: {current}/{total}", end="", flush=True)
            
            counts = import_from_filestore(
                args.source,
                args.destination,
                experiment_id=args.experiment,
                progress_callback=progress,
            )
            
            print()  # Newline after progress
            print(f"Import complete:")
            for key, value in counts.items():
                print(f"  {key}: {value}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    else:
        print("Usage: metalab store {export|import}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
