#!/usr/bin/env python3
"""
Run the gene perturbation experiment.

Usage (from repo root):
    # Light test (quick verification)
    uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity light

    # Medium experiment
    uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity medium --workers 4

    # Heavy experiment (many perturbation values)
    uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity heavy --workers 8

    # Single gene investigation
    uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --gene KLF1

Note:
    This example uses dynamo-release which has deprecated dependencies. It runs in
    an isolated workspace environment to avoid polluting the main metalab deps.

This script demonstrates:
- In silico gene perturbation using dynamo
- Artifact generation (plots, transition matrices)
- Progress tracking with metalab.progress API
- Result analysis with metalab.display_results
"""

from __future__ import annotations

import argparse
from datetime import datetime

import metalab
from experiment import (
    build_experiment,
    build_single_gene_experiment,
)

# Check for rich availability
try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def count_grid_runs(exp: metalab.Experiment) -> int:
    """Count total number of runs in an experiment."""
    n_params = sum(1 for _ in exp.params)
    n_seeds = sum(1 for _ in exp.seeds)
    return n_params * n_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gene perturbation experiment")
    parser.add_argument(
        "--intensity",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Experiment intensity (default: light)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    parser.add_argument(
        "--store",
        default="./runs/gene_perturbation",
        help="Storage directory (default: ./runs/gene_perturbation)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-run all)",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich progress bar (use simple output)",
    )
    parser.add_argument(
        "--gene",
        choices=["KLF1", "SPI1", "GATA1"],
        help="Run targeted experiment for a single gene",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        help="Override number of replicates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count runs without executing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for large experiments",
    )

    args = parser.parse_args()

    # Setup console for output
    console = Console() if HAS_RICH and not args.no_rich else None

    def print_msg(msg: str, style: str = "") -> None:
        if console:
            console.print(msg)
        else:
            # Strip rich markup for plain output
            import re

            plain = re.sub(r"\[/?[^\]]+\]", "", msg)
            print(plain)

    # Check for dynamo
    try:
        import dynamo as dyn

        print_msg(f"[green]✓[/green] dynamo version: {dyn.__version__}")
    except ImportError:
        print_msg("[red]✗[/red] dynamo not installed!")
        print_msg("Install with: [cyan]pip install dynamo-release[/cyan]")
        return

    # Build experiment
    if args.gene:
        print_msg(f"Building targeted experiment for gene: [cyan]{args.gene}[/cyan]")
        exp = build_single_gene_experiment(
            gene=args.gene,
            replicates=args.replicates or 3,
        )
    else:
        print_msg(f"Building experiment with intensity: [cyan]{args.intensity}[/cyan]")
        exp = build_experiment(intensity=args.intensity)

    # Count runs
    n_runs = count_grid_runs(exp)
    print_msg(f"Experiment: [bold]{exp.name}[/bold] v{exp.version}")
    print_msg(f"Total runs: [bold]{n_runs:,}[/bold]")

    if args.dry_run:
        print_msg("\n[yellow][Dry run - not executing][/yellow]")
        print_msg("\nSample parameter combinations:")
        for i, param_case in enumerate(exp.params):
            if i >= 5:
                remaining = n_runs // max(1, sum(1 for _ in exp.seeds)) - 5
                print_msg(f"  ... and {remaining} more")
                break
            print_msg(f"  {param_case.params}")
        return

    # Warn about large experiments
    if n_runs > 50 and not args.yes:
        print_msg(f"\n[yellow]WARNING: This will execute {n_runs:,} runs![/yellow]")
        print_msg("Consider using --intensity light for initial testing.")
        print_msg("Use --yes to skip this prompt.")
        try:
            response = input("Continue? [y/N] ")
            if response.lower() != "y":
                print_msg("Aborted.")
                return
        except EOFError:
            print_msg("\nNo TTY available. Use --yes to skip confirmation.")
            return

    print_msg(f"\nStarting with [cyan]{args.workers}[/cyan] workers...")
    print_msg(f"Store: {args.store}")
    print_msg(f"Resume: {not args.no_resume}")

    # Create progress tracker using metalab API
    progress_style = "simple" if args.no_rich else "auto"
    tracker = metalab.create_progress_tracker(
        total=n_runs,
        title=f"{exp.name} ({args.intensity})" if not args.gene else f"{exp.name}",
        style=progress_style,
        display_metrics=["gene", "perturbation_value:>8.0f", "n_groups"],
    )

    # Run experiment with progress tracker
    with tracker:
        result = metalab.run(
            exp,
            store=args.store,
            executor="threads",
            max_workers=args.workers,
            resume=not args.no_resume,
            progress=True,
            on_event=tracker,
        )

    # Display results using metalab API
    metalab.display_results(
        result,
        console=console,
        show_by_group=["gene", "perturbation_value"],
    )

    # Save results table
    df = result.table(as_dataframe=True)
    if not df.empty:
        csv_path = (
            f"{args.store}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(csv_path, index=False)
        print_msg(f"\n[dim]Results saved to: {csv_path}[/dim]")

        # Show summary statistics
        print_msg("\n[bold]Summary by Gene:[/bold]")
        if "gene" in df.columns:
            for gene in df["gene"].unique():
                gene_df = df[df["gene"] == gene]
                n_success = (gene_df["status"] == "success").sum()
                n_total = len(gene_df)
                print_msg(f"  {gene}: {n_success}/{n_total} successful runs")


if __name__ == "__main__":
    main()
